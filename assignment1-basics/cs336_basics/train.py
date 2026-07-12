import torch
import argparse
import numpy as np
import time

from cs336_basics.transformer import *

torch.set_float32_matmul_precision('high')  # TF32 for any fp32 matmul (e.g. eval)

# ----- fixed harness constants: do NOT change these across experiments -----
TRAIN_SECONDS = 600.0   # wall-clock training budget, excludes startup and final eval
CONTEXT_LENGTH = 512    # fixed sequence length
EVAL_SEQS = 4096        # final eval: first N non-overlapping windows of the valid set
EVAL_BATCH_SIZE = 32
PERIODIC_EVAL_SEQS = 640
PEAK_FLOPS = 756e12     # H100 PCIe BF16 dense, for MFU reporting only

parser = argparse.ArgumentParser(description="Train a Transformer language model.")
parser.add_argument("--train_path", type=str, default="owt_train.npy", help="Path to the training dataset (.npy).")
parser.add_argument("--valid_path", type=str, default="owt_valid.npy", help="Path to the validation dataset (.npy).")
parser.add_argument("--batch_size", type=int, default=96, help="Training batch size.")
parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda/cpu/mps).")
parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size.")
parser.add_argument("--num_layers", type=int, default=10, help="Number of Transformer layers.")
parser.add_argument("--d_model", type=int, default=512, help="Transformer hidden dimension.")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward hidden dimension.")
parser.add_argument("--theta", type=float, default=10000.0, help="RoPE base frequency.")
parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
parser.add_argument("--beta2", type=float, default=0.99, help="AdamW beta2.")
parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay coefficient.")
parser.add_argument("--max_l2_norm", type=float, default=2.0, help="Gradient clipping L2 norm.")
parser.add_argument("--alpha_max", type=float, default=3e-3, help="Peak AdamW learning rate (embedding + norms).")
parser.add_argument("--muon_lr", type=float, default=0.0125, help="Peak Muon learning rate (2D hidden weights).")
parser.add_argument("--alpha_min", type=float, default=3e-5, help="Minimum learning rate.")
parser.add_argument("--T_w", type=int, default=2000, help="Warmup steps.")
parser.add_argument("--T_c", type=int, default=100000, help="Cosine decay steps.")
parser.add_argument("--valid_interval", type=int, default=2500, help="Validation interval (steps).")
parser.add_argument("--train_seconds", type=float, default=TRAIN_SECONDS, help="Override budget for smoke tests only.")

args = parser.parse_args()


@torch.no_grad()
def evaluate(model, valid_data, num_seqs, device):
    # Deterministic eval on the first num_seqs non-overlapping windows of the valid set
    model.eval()
    loss_sum = 0.0
    num_batches = 0
    for start in range(0, num_seqs, EVAL_BATCH_SIZE):
        seq_ids = start + np.arange(EVAL_BATCH_SIZE)
        idx = seq_ids[:, None] * CONTEXT_LENGTH + np.arange(CONTEXT_LENGTH)[None, :]
        x = torch.from_numpy(valid_data[idx].astype(np.int64)).to(device)
        y = torch.from_numpy(valid_data[idx + 1].astype(np.int64)).to(device)
        loss_sum += cross_entropy(model(x), y).item()
        num_batches += 1
    model.train()
    return loss_sum / num_batches


def main():
    # Fixed seed (init + data-batch sampling) so experiment A/B comparisons are clean.
    torch.manual_seed(0)

    # Train data lives entirely on the GPU to avoid per-step memmap reads + H->D copy.
    # uint16 has no torch dtype; reinterpret as int16 (all ids < 32000 < 32768, so the
    # bit pattern is preserved and values stay non-negative).
    train_np = np.load(args.train_path)
    train_data = torch.from_numpy(train_np.view(np.int16)).to(args.device)
    del train_np
    valid_data = np.load(args.valid_path, mmap_mode='r')

    print(f"**Loaded training data from {args.train_path}", flush=True)
    print(f"**Loaded validation data from {args.valid_path}", flush=True)

    model = TransformerLM(
        args.vocab_size,
        CONTEXT_LENGTH,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.theta,
    ).to(args.device)

    # Optimizer split: 2D hidden weight matrices -> Muon; embedding (tied) + 1D norm
    # gains -> AdamW. Built from the uncompiled model for clean parameter identity.
    emb_param = model.emb.e
    muon_params, adam_params = [], []
    for p in model.parameters():
        if p is emb_param or p.dim() < 2:
            adam_params.append(p)
        else:
            muon_params.append(p)

    model = torch.compile(model, dynamic=False)  # static shapes; first-step compile excluded from budget

    opt_adam = AdamW(adam_params, args.alpha_max, (args.beta1, args.beta2), args.eps, args.weight_decay)
    opt_muon = Muon(muon_params, args.muon_lr, momentum=0.95, weight_decay=args.weight_decay)

    num_params = sum(p.numel() for p in model.parameters())
    num_emb_params = model.emb.e.numel()
    print(f"**num_params: {num_params / 1e6:.1f}M", flush=True)

    is_cuda = args.device.startswith("cuda")

    # Pre-warm the eval (batch 32) compile graph during startup so its one-time
    # torch.compile recompile is excluded from the training budget.
    evaluate(model, valid_data, EVAL_BATCH_SIZE, args.device)

    start_total = time.perf_counter()
    step = 0
    clock_start = None
    # Wall-clock-driven LR schedule: robust to throughput / step-count changes.
    # Warmup over the first few % of the budget, then cosine anneal to alpha_min
    # exactly at the end of the budget. The schedule clock aligns with the budget
    # clock (both start after step 1, so first-step compile/warmup is excluded).
    T_w_sec = 0.03 * args.train_seconds
    T_c_sec = args.train_seconds

    while True:
        # Shared cosine schedule multiplier (elapsed seconds); scale each optimizer's peak LR.
        elapsed_sched = 0.0 if clock_start is None else time.perf_counter() - clock_start
        lr_frac = learning_rate_schedule(
            elapsed_sched, 1.0, args.alpha_min / args.alpha_max, T_w_sec, T_c_sec
        )
        for group in opt_adam.param_groups:
            group['lr'] = lr_frac * args.alpha_max
        for group in opt_muon.param_groups:
            group['lr'] = lr_frac * args.muon_lr

        # Sample training batch
        train_input_tokens, train_next_tokens = data_loading(
            train_data, args.batch_size, CONTEXT_LENGTH, args.device
        )

        opt_adam.zero_grad()
        opt_muon.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model(train_input_tokens)
            loss = cross_entropy(y, train_next_tokens)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        opt_muon.step()
        opt_adam.step()
        step += 1

        if clock_start is None:
            # First step done: startup/compilation is over, start the budget clock
            if is_cuda:
                torch.cuda.synchronize()
            clock_start = time.perf_counter()

        elapsed = time.perf_counter() - clock_start
        if elapsed >= args.train_seconds:
            break

        if step % args.valid_interval == 0:
            valid_loss = evaluate(model, valid_data, PERIODIC_EVAL_SEQS, args.device)
            print(
                f"Iteration {step:5d} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {valid_loss:.4f} | "
                f"Elapsed: {elapsed / 60:.2f} min",
                flush=True
            )

    if is_cuda:
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - clock_start

    # Final fixed evaluation (excluded from the training budget)
    valid_loss = evaluate(model, valid_data, EVAL_SEQS, args.device)

    total_seconds = time.perf_counter() - start_total
    peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20 if is_cuda else 0.0
    tokens_per_step = args.batch_size * CONTEXT_LENGTH
    total_tokens = step * tokens_per_step
    # MFU: 6 * matmul params (embedding lookup excluded) + attention term, vs bf16 dense peak
    flops_per_token = 6 * (num_params - num_emb_params) + 12 * args.num_layers * args.d_model * CONTEXT_LENGTH
    timed_tokens = (step - 1) * tokens_per_step
    mfu = 100 * flops_per_token * timed_tokens / training_seconds / PEAK_FLOPS

    print("---", flush=True)
    print(f"valid_loss:       {valid_loss:.6f}", flush=True)
    print(f"training_seconds: {training_seconds:.1f}", flush=True)
    print(f"total_seconds:    {total_seconds:.1f}", flush=True)
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}", flush=True)
    print(f"mfu_percent:      {mfu:.2f}", flush=True)
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}", flush=True)
    print(f"num_steps:        {step}", flush=True)
    print(f"num_params_M:     {num_params / 1e6:.1f}", flush=True)
    print(f"depth:            {args.num_layers}", flush=True)


if __name__ == "__main__":
    main()
