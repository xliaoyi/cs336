import torch
import argparse
import numpy as np
import time
import wandb

from cs336_basics.transformer import *

parser = argparse.ArgumentParser(description="Train a Transformer language model.")
parser.add_argument("--train_path", type=str, default="data/train.npy", help="Path to the training dataset (.npy).")
parser.add_argument("--valid_path", type=str, default="data/valid.npy", help="Path to the validation dataset (.npy).")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length.")
parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda/cpu/mps).")
parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size.")
parser.add_argument("--num_layers", type=int, default=12, help="Number of Transformer layers.")
parser.add_argument("--d_model", type=int, default=768, help="Transformer hidden dimension.")
parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward hidden dimension.")
parser.add_argument("--theta", type=float, default=10000.0, help="RoPE base frequency.")
parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay coefficient.")
parser.add_argument("--n_iter", type=int, default=100000, help="Total training steps.")
parser.add_argument("--checkpoint_path_prefix", type=str, default="checkpoints/model", help="Checkpoint filename prefix.")
parser.add_argument("--max_l2_norm", type=float, default=1.0, help="Gradient clipping L2 norm.")
parser.add_argument("--alpha_max", type=float, default=1.0, help="Maximum LR multiplier.")
parser.add_argument("--alpha_min", type=float, default=0.1, help="Minimum LR multiplier.")
parser.add_argument("--T_w", type=int, default=2000, help="Warmup steps.")
parser.add_argument("--T_c", type=int, default=100000, help="Cosine decay steps.")
parser.add_argument("--valid_interval", type=int, default=500, help="Validation interval (steps).")
parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval (steps).")
parser.add_argument("--validation_batches", type=int, default=20, help="Number of validation batches.")

args = parser.parse_args()

def main():
    start = time.perf_counter()

    train_data = np.load(args.train_path, mmap_mode='r')

    print(f"Load training set, spent {time.perf_counter() - start}")

    valid_data = np.load(args.valid_path, mmap_mode='r')

    print(f"Load validation set, spent {time.perf_counter() - start}")

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.theta,
    ).to(args.device)

    opt = AdamW(
        model.parameters(), 
        args.lr,
        (args.beta1, args.beta2),
        args.eps,
        args.weight_decay
    )

    wandb.init(project="cs336-assignment1", config=vars(args))

    completed_steps = 0
    for t in range(args.n_iter):
        # Set scheduled learning rate
        for group in opt.param_groups:
            group['lr'] = learning_rate_schedule(
                t, args.alpha_max, args.alpha_min, args.T_w, args.T_c
            )

        # Sample training batch
        train_input_tokens, train_next_tokens = data_loading(
            train_data, args.batch_size, args.context_length, args.device
        )

        # Zero gradients
        opt.zero_grad()

        # Forward and loss
        y = model(train_input_tokens)
        loss = cross_entropy(y, train_next_tokens)

        # Backward
        loss.backward()

        # Clip gradients
        gradient_clipping(model.parameters(), args.max_l2_norm)

        # Optimizer step
        opt.step()
        completed_steps = t + 1

        # Periodically validate and checkpoint
        if completed_steps % args.valid_interval == 0:
            valid_loss_sum = 0
            model.eval()
            with torch.no_grad():
                for sample in range(args.validation_batches):                  
                    valid_input_tokens, valid_next_tokens = data_loading(
                        valid_data, args.batch_size, args.context_length, args.device
                    )
                    y_valid = model(valid_input_tokens)
                    loss_valid = cross_entropy(y_valid, valid_next_tokens)
                    valid_loss_sum += loss_valid.item()
                valid_loss_mean = valid_loss_sum / args.validation_batches
            print(
                f"Iteration {t}: Train loss: {loss.item()}, validation loss: {valid_loss_mean}, spent {time.perf_counter() - start}")
            model.train()

            wandb.log({
                "step": completed_steps,
                "train_loss": loss.item(),
                "val_loss": valid_loss_mean,
                "wall_time": time.perf_counter() - start,
            })
        
        if completed_steps % args.checkpoint_interval == 0:
            save_checkpoint(
                model, opt, completed_steps, f"{args.checkpoint_path_prefix}_iter{completed_steps}"
            )

    wandb.finish()

    # final model
    save_checkpoint(
        model, opt, completed_steps, f"{args.checkpoint_path_prefix}_final_iter{completed_steps}"
    )

if __name__ == "__main__":
    main()


