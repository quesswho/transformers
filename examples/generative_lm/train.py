"""
Generative language model trainer.

Uses the Encoder stack as a decoder-only transformer (GPT-style):
a causal mask turns bidirectional self-attention into autoregressive generation.

Run from project root:
    python examples/generative_lm/train.py --data data/shakespeare.txt --output shakespeare_model.pt
    python examples/generative_lm/train.py --data mytext.txt --output model.pt
    python examples/generative_lm/train.py  # auto-downloads tinyshakespeare

    Save a tokenizer
    python examples/generative_lm/train.py --data mytext.txt --save-tokenizer tok.json --vocab-size 2000
    Load a tokenizer
    python examples/generative_lm/train.py --data mytext.txt --tokenizer tok.json
"""

import argparse
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import DecoderOnlyTransformer
from tokenizer import SentencePieceBPE

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def save_checkpoint(path, model, optimizer, step, tokenizer, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokenizer": tokenizer.to_dict(),
        "config": config,
    }, path)


def load_text(path: str | None) -> str:
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.close()
    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(DATA_URL, tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        text = f.read()
    os.unlink(tmp.name)
    return text


class CharDataset(Dataset):
    def __init__(self, data: list[int], block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a generative language model.")
    parser.add_argument("--data", default=None, help="Path to a .txt training file (default: auto-download tinyshakespeare)")
    parser.add_argument("--output", default="model.pt", help="Path to save the checkpoint (default: model.pt)")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--block-size", type=int, default=256, help="Context window length (default: 256)")
    parser.add_argument("--d-model", type=int, default=256, help="Model embedding dimension (default: 256)")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of transformer layers (default: 8)")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward hidden dimension (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    parser.add_argument("--vocab-size", type=int, default=2000, help="SentencePiece vocab size (default: 2000)")
    parser.add_argument("--tokenizer", default=None, help="Path to a pre-trained tokenizer JSON file; skips BPE training")
    parser.add_argument("--save-tokenizer", default=None, help="Save the tokenizer to this JSON file after training/loading")
    parser.add_argument("--val-interval", type=int, default=500, help="Validate every N steps (default: 500)")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for periodic checkpoints (default: disabled)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--prompt", default=None, help="Seed text for sample generation after training")
    parser.add_argument("--generate-steps", type=int, default=500, help="Tokens to generate after training (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--vocab-only", action="store_true", help="Build and save the tokenizer then exit without training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")

    text = load_text(args.data)
    print(f"Finished loading text from {args.data}...")
    if args.tokenizer is not None:
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = SentencePieceBPE.load(args.tokenizer)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}\n")
    else:
        tokenizer = SentencePieceBPE()
        print(f"Training SentencePiece BPE tokenizer (vocab_size={args.vocab_size})...")
        tokenizer.train(text, vocab_size=args.vocab_size)
        print(f"Tokenizer trained. Vocab size: {tokenizer.vocab_size}\n")

    if args.save_tokenizer is not None:
        tokenizer.save(args.save_tokenizer)
        print(f"Tokenizer saved → {args.save_tokenizer}\n")

    if args.vocab_only:
        if args.save_tokenizer is None:
            print("Warning: --vocab-only set but --save-tokenizer not specified; tokenizer was not saved.")
        return

    data = tokenizer.encode(text)
    vocab_size = tokenizer.vocab_size

    split = int(0.9 * len(data))
    train_dataset = CharDataset(data[:split], args.block_size)
    val_dataset = CharDataset(data[split:], args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Vocab size: {vocab_size}  |  Train tokens: {len(data[:split]):,}  |  Val tokens: {len(data[split:]):,}\n")

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.block_size,
    ).to(device)

    counts = model.count_parameters()
    total = sum(counts.values())
    print(f"{'Component':<25} {'Params':>12}  {'%':>6}")
    for name, val in counts.items():
        print(f"  {name:<23} {val:>12,}  {val/total*100:>5.1f}%")
    print(f"  {'TOTAL':<23} {total:>12,}\n")

    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "block_size": args.block_size,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    start_step = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "step" in ckpt:
            start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}\n")

    step = start_step - 1
    train_iter = iter(train_loader)
    for step in range(start_step, args.steps + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.val_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)
                    val_losses.append(criterion(vlogits.view(-1, vocab_size), vy.view(-1)).item())
                    if len(val_losses) >= 20:
                        break
            val_loss = sum(val_losses) / len(val_losses)
            print(f"Step {step:5d}/{args.steps}  train={loss.item():.4f}  val={val_loss:.4f}")
            if args.checkpoint_dir is not None:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                save_checkpoint(os.path.join(args.checkpoint_dir, "latest.pt"), model, optimizer, step, tokenizer, config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(os.path.join(args.checkpoint_dir, "best.pt"), model, optimizer, step, tokenizer, config)
                    print(f"  → new best checkpoint (val={val_loss:.4f})")

    save_checkpoint(args.output, model, optimizer, step, tokenizer, config)
    print(f"\nModel saved → {args.output}")

    if args.prompt is not None:
        print(f"\n--- Generated text (temperature={args.temperature}) ---\n")
        ctx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
        out = model.generate(ctx, max_new_tokens=args.generate_steps, temperature=args.temperature)
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
