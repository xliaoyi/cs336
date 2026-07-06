import torch
import argparse
import numpy as np

from cs336_basics.transformer import *

parser = argparse.ArgumentParser(description='Train transformer')
parser.add_argument("--train_path", type = str, help = 'Path to training data, data should be saved with np')
parser.add_argument("--valid_path", type = str, help = 'Path to validation data, data should be saved with np')
parser.add_argument("--batch_size", type = int, help = 'Path to training data')
parser.add_argument("--context_length", type = int, help = 'Path to training data')
parser.add_argument("--device", type = str, help = 'Path to training data')
parser.add_argument("--vocab_size", type = int, help = 'Path to training data')
parser.add_argument("--num_layers", type = int, help = '')
parser.add_argument("--d_model", type = int, help = '')
parser.add_argument("--num_heads", type = int, help = '')
parser.add_argument("--d_ff", type = int, help = '')
parser.add_argument("--theta", type = float, help = '')
parser.add_argument("--lr", type = float, help = '')
parser.add_argument("--beta1", type = float, help = '')
parser.add_argument("--beta2", type = float, help = '')
parser.add_argument("--eps", type = float, help = '')
parser.add_argument("--weight_decay", type = float, help = '')
parser.add_argument("--n_iter", type = int, help = '')
parser.add_argument("--checkpoint_path_prefix", type = str, help = '')
parser.add_argument("--max_l2_norm", type = float, help = '')
parser.add_argument("--alpha_max", type = float, help = '')
parser.add_argument("--alpha_min", type = float, help = '')
parser.add_argument("--T_w", type = int, help = '')
parser.add_argument("--T_c", type = int, help = '')
parser.add_argument("--valid_checkpoint_interval", type = int, help = '')
parser.add_argument("--validation_batches", type = int, help = '')

args = parser.parse_args()

def main():
    train_data = np.load(args.train_path, mmap_mode='r')

    valid_data = np.load(args.valid_path, mmap_mode='r')

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

        # Periodically validate and checkpoint
        if completed_steps % args.valid_checkpoint_interval == 0:
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
            print(f"Iteration {t}: Train loss: {loss.item()}, validation loss: {valid_loss_mean}")
            save_checkpoint(model, opt, completed_steps, f"{args.checkpoint_path_prefix}_iter{completed_steps}")
            model.train()

        completed_steps = t + 1
    
    # final model
    save_checkpoint(model, opt, completed_steps, f"{args.checkpoint_path_prefix}_final_iter{completed_steps}")

if __name__ == "__main__":
    main()


