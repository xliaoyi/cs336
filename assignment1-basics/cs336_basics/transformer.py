import torch
import math

from torch import nn, Tensor
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from typing import Optional
from collections.abc import Callable, Iterable
from .tokenizer import Tokenizer


class Linear(nn.Module):
    def __init__(
        self, 
        in_features, out_features, device=None, dtype=None
    ):
        super().__init__()

        self.W = nn.Parameter(
            torch.empty(out_features, in_features, dtype = dtype, device = device)
        )

        params_std = torch.sqrt(torch.tensor(2/(in_features + out_features)))
        torch.nn.init.trunc_normal_(
            self.W, mean = 0, std = params_std,
            a = -3.0*params_std, b = 3.0*params_std
        )

    def forward(self, x):
        # return x @ self.W
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings, embedding_dim, device=None, dtype=None
    ):
        super().__init__()

        self.e = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype = dtype, device = device)
        )

        # Fixed GPT-2-style std (the fan-in+fan-out formula collapses to ~0.008 at
        # vocab=32000, which weakens the token signal and, under weight tying, the logit scale).
        params_std = 0.02
        torch.nn.init.trunc_normal_(
            self.e, mean = 0, std = params_std,
            a = -3.0*params_std, b = 3.0*params_std
        )

    def forward(self, token_ids):
        return self.e[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model, eps = 1e-5, device=None, dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rmsa = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1) / self.d_model + self.eps)
        rmsa = rearrange(rmsa, "... -> ... 1")
        result = (x / rmsa) * self.g

        return result.to(in_dtype)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)

    def forward(self, x):
        W1x = self.W1(x)
        silu = W1x * torch.sigmoid(W1x)
        W3x = self.W3(x)
        siluW1W3 = silu * W3x
        return self.W2(siluW1W3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x, token_positions):
        k = torch.arange(self.d_k // 2, device=x.device)
        token_positions = rearrange(token_positions, "... -> ... 1") # ... seq_len 1
        angle = token_positions / (torch.pow(self.theta, 2*k / self.d_k)) # ... seq_len d_k/2
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        
        # not building a whole dxd matrix R for saving memory
        x_even = x[..., 0::2] # ... seq_len d_k/2
        x_odd = x[..., 1::2] # ... seq_len d_k/2

        y_even = x_even * cos_angle - x_odd * sin_angle # ... seq_len d_k/2
        y_odd = x_even * sin_angle + x_odd * cos_angle # ... seq_len d_k/2

        y = torch.stack([y_even, y_odd], dim = -1) # ... seq_len d_k/2 2
        y = rearrange(y, "... even odd -> ... (even odd)")
        return y.to(x.dtype)  # keep dtype stable under autocast (rotation done in fp32)


def softmax(
    x: Float[Tensor, "..."],
    i: int,
) -> Float[Tensor, " ..."]:
    max_val = torch.amax(x, dim = i, keepdim=True)
    x = x - max_val
    expx = torch.exp(x)
    res = expx / torch.sum(expx, dim = i, keepdim=True)
    return res


def scaled_dot_product_attention(Q, K, V, mask):
    QK = einsum(
        Q, K,
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    )
    sqrt_d_k = K.shape[-1] ** 0.5
    norm_QK = QK / sqrt_d_k
    if mask is not None:
        norm_QK = norm_QK + torch.where(mask, 0.0, -torch.inf)
    scaled_QK = softmax(norm_QK, -1)
    attn = einsum(
        scaled_QK, V,
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v"
    )
    return attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.WQ = Linear(d_model, d_model)
        self.WK = Linear(d_model, d_model)
        self.WV = Linear(d_model, d_model)
        self.WO = Linear(d_model, d_model)
        # self.token_positions = token_positions
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        # split heads
        Q = rearrange(
            Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads = self.num_heads
        )
        K = rearrange(
            K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads = self.num_heads
        )
        V = rearrange(
            V, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads = self.num_heads
        )

        # RoPE
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # causal mask (hand-written attention; no torch.nn.functional)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attn = scaled_dot_product_attention(Q, K, V, mask) # ... num_heads seq_len d_k

        # concat heads
        multi_attn = rearrange(attn, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        return self.WO(multi_attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rmsnorm1 = RMSNorm(self.d_model)
        self.rmsnorm2 = RMSNorm(self.d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        y = self.rmsnorm1(x)
        token_positions = torch.arange(x.shape[-2], device = x.device)
        y = self.attn(y, token_positions)

        # residual connection
        y = y + x

        z = self.rmsnorm2(y)
        z = self.ffn(z)

        # residual connection
        z = z + y

        return z


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.emb = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)]
        )
        self.rmsnorm3 = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        # Weight tying: share the input embedding and output projection (both [vocab, d_model]).
        self.linear.W = self.emb.e

    def forward(self, x):
        x = self.emb(x) # ... seq_length d_model
        for layer in self.layers:
            x = layer(x)

        x = self.rmsnorm3(x)
        x = self.linear(x) # ... seq_length vocab_size
        # x = softmax(x, -1) # ... seq_length vocab_size, the model must return unnormalized logits for cross-entropy
        return x

def cross_entropy(o, x):
    o = o.float()  # 32000-way logsumexp in fp32 (safe under bf16 autocast; no-op in fp32 eval)
    max_o = torch.amax(o, dim = -1, keepdim=True)
    o = o - max_o
    x = rearrange(x, "... -> ... 1")
    nll = torch.log(torch.sum(torch.exp(o), keepdim=True, dim = -1)) - torch.gather(o, dim=-1, index=x)
    return torch.mean(nll)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if len(betas) != 2:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state["t"] + 1
                m = state["m"]
                v = state["v"]
                g = p.grad
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.sub_(lr * wd * p)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                state['t'] = t
                p.addcdiv_(m, torch.sqrt(v) + eps, value=-lr_t)
        return loss


def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t * alpha_max / T_w
    elif t >= T_w and t <= T_c:
        return alpha_min + 0.5 * (1+math.cos((t - T_w)*math.pi / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        return alpha_min

@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    parameters = [p for p in parameters if p.grad is not None]
    l2_norm = 0.0
    for p in parameters:
        l2_norm = l2_norm + p.grad.square().sum()
    l2_norm = l2_norm ** 0.5
    if l2_norm > max_l2_norm:
        clip = max_l2_norm / (l2_norm + 1e-6)
        for p in parameters:
            p.grad.mul_(clip)


def data_loading(x, batch_size, context_length, device):
    # x is a 1-D token tensor already resident on `device`; gather random windows
    # entirely on-device (no per-step numpy fancy-indexing or host->device copy).
    high = x.shape[0] - context_length
    samples = torch.randint(0, high, (batch_size, 1), device=device)
    context = torch.arange(context_length, device=device)
    idx0 = samples + context
    return x[idx0].long(), x[idx0 + 1].long()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str
):
    model_state = model.state_dict()
    opt_state = optimizer.state_dict()
    params = {
        'model_state': model_state,
        'opt_state': opt_state,
        'iteration': iteration
    }
    torch.save(params, out)


def load_checkpoint(src, model, optimizer):
    params = torch.load(src)
    model.load_state_dict(params['model_state'])
    optimizer.load_state_dict(params['opt_state'])
    iteration = params['iteration']
    return iteration

@torch.no_grad()
def decoding(
    model: torch.nn.Module, 
    prompt: str, 
    max_generate_tokens: int, 
    temperature: float,
    p_threshold:float,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens = ['<|endoftext|>']
):
    device = next(model.parameters()).device

    tokenizer = Tokenizer.from_files(
		vocab_filepath = vocab_filepath,
		merges_filepath = merges_filepath,
		special_tokens = special_tokens
	)

    input_token_ids = torch.tensor(tokenizer.encode(prompt),  dtype=torch.long, device=device)
    input_token_ids = rearrange(input_token_ids, "seq_len -> 1 seq_len")
    input_token_ids = input_token_ids[..., -model.context_length:]

    n_generate_tokens = 0 
    generate_token_ids = []

    special_token_ids = []
    for special_token in special_tokens:
        special_token_ids += tokenizer.encode(special_token)

    model.eval()

    sample_id = -1
    while n_generate_tokens < max_generate_tokens:
        if sample_id in special_token_ids:
            break

        logits = model(input_token_ids) # ... seq_length vocab_size
        last_token_logits = logits[..., -1, :] # ... vocab_size
        last_token_logits = last_token_logits / temperature
        last_token_prob = softmax(last_token_logits, -1) # ... vocab_size

        total_p = 0
        sorted_probs, sorted_indices = last_token_prob.sort(descending=True)
        for i in range(len(sorted_probs[-1])):
            if total_p >= p_threshold:
                break
            else:
                total_p += sorted_probs[..., i]
            k = i + 1
        
        #
        last_token_prob_flt = sorted_probs[..., :k]
        last_token_idx_flt = sorted_indices[..., :k]

        sample_id = torch.multinomial(last_token_prob_flt, num_samples=1) # ... 1
        sample_id = torch.gather(last_token_idx_flt, dim=-1, index=sample_id)

        input_token_ids = torch.cat([input_token_ids, sample_id], dim=-1)

        sample_id = sample_id.item()
        generate_token_ids.append(sample_id)

        input_token_ids = input_token_ids[..., -model.context_length:]
        n_generate_tokens += 1

    return tokenizer.decode(generate_token_ids)
    


    


    












            

        







