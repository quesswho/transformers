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

from transformer import Encoder
from tokenizer import SentencePieceBPE

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        block_size: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers, d_ff, dropout, block_size)
        self.head = nn.Linear(d_model, vocab_size)
        causal = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("causal_mask", causal.unsqueeze(0).unsqueeze(0))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self.causal_mask[:, :, :T, :T]
        return self.head(self.encoder(x, mask))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, steps: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        block_size = self.causal_mask.size(-1)
        for _ in range(steps):
            ctx = idx[:, -block_size:]
            logits = self(ctx)[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a generative language model.")
    parser.add_argument("--data", default=None, help="Path to a .txt training file (default: auto-download tinyshakespeare)")
    parser.add_argument("--output", default="model.pt", help="Path to save the checkpoint (default: model.pt)")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--block-size", type=int, default=256, help="Context window length (default: 256)")
    parser.add_argument("--d-model", type=int, default=256, help="Model embedding dimension (default: 256)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward hidden dimension (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    parser.add_argument("--vocab-size", type=int, default=2000, help="SentencePiece vocab size (default: 2000)")
    parser.add_argument("--tokenizer", default=None, help="Path to a pre-trained tokenizer JSON file; skips BPE training")
    parser.add_argument("--save-tokenizer", default=None, help="Save the tokenizer to this JSON file after training/loading")
    parser.add_argument("--val-interval", type=int, default=500, help="Validate every N steps (default: 500)")
    parser.add_argument("--prompt", default=None, help="Seed text for sample generation after training")
    parser.add_argument("--generate-steps", type=int, default=500, help="Tokens to generate after training (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--vocab-only", action="store_true", help="Build and save the tokenizer then exit without training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")

    text = load_text(args.data)
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

    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        block_size=args.block_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
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

    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "block_size": args.block_size,
        },
    }, args.output)
    print(f"\nModel saved → {args.output}")

    if args.prompt is not None:
        print(f"\n--- Generated text (temperature={args.temperature}) ---\n")
        ctx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
        out = model.generate(ctx, steps=args.generate_steps, temperature=args.temperature)
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
