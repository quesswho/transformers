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
import hashlib
import os
import random
import sys
import tempfile
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import DecoderOnlyTransformer, load_model_state_dict
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


def _tokenizer_fingerprint(tokenizer) -> str:
    d = tokenizer.to_dict()
    raw = str(sorted(d.get("vocab", {}).items())) + str(d.get("merges", []))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_tokens(text: str, tokenizer, data_path: str | None) -> np.ndarray:
    cache_path = None
    if data_path is not None:
        stat = os.stat(data_path)
        key = f"{stat.st_size}_{int(stat.st_mtime)}_{_tokenizer_fingerprint(tokenizer)}"
        cache_path = data_path + f".{key}.tokens.npy"
        if os.path.exists(cache_path):
            print(f"Loading cached tokens from {cache_path}...")
            return np.load(cache_path, mmap_mode="r")

    print("Encoding corpus...")
    tokens = tokenizer.encode(text)
    arr = np.asarray(tokens, dtype=np.int32)

    if cache_path is not None:
        np.save(cache_path, arr)
        print(f"Token cache written → {cache_path}")

    return arr


class CharDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = random.randint(0, len(self.data) - self.block_size - 1)
        chunk = self.data[i : i + self.block_size + 1]
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
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers (default: 8)")
    parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward hidden dimension (default: 8/3 * d_model)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    parser.add_argument("--vocab-size", type=int, default=2000, help="SentencePiece vocab size (default: 2000)")
    parser.add_argument("--tokenizer", default=None, help="Path to a pre-trained tokenizer JSON file; skips BPE training")
    parser.add_argument("--save-tokenizer", default=None, help="Save the tokenizer to this JSON file after training/loading")
    parser.add_argument("--val-interval", type=int, default=500, help="Validate every N steps (default: 500)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log training speed every N steps (default: 100)")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for periodic checkpoints (default: disabled)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--vocab-only", action="store_true", help="Build and save the tokenizer then exit without training")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model before training (slow first step, faster afterwards)")
    args = parser.parse_args()

    # We scale the feed-forward dimension
    # as suggested in https://arxiv.org/pdf/2002.05202
    if args.d_ff is None:
        args.d_ff = round(8 / 3 * args.d_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")

    # Allow TF32 matmuls for the ops that remain in fp32 (free speedup on Ampere+).
    torch.set_float32_matmul_precision("high")
    # bf16 autocast: We use a float type with the same exponent range as fp32 
    # but with less precision.
    # Master weights and the optimizer stay in fp32; only the forward/backward
    # compute runs in bf16.
    # This gives a whopping 79% speedup (69k tokens/sec -> 120k tokens/sec) in our tests on an RTX 3060
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_amp:
        print("Using bf16 mixed precision\n")

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
        save_dir = os.path.dirname(args.save_tokenizer)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(args.save_tokenizer)
        print(f"Tokenizer saved → {args.save_tokenizer}\n")

    if args.vocab_only:
        if args.save_tokenizer is None:
            print("Warning: --vocab-only set but --save-tokenizer not specified; tokenizer was not saved.")
        return

    data = _load_tokens(text, tokenizer, args.data)
    vocab_size = tokenizer.vocab_size

    split = int(0.9 * len(data))
    train_dataset = CharDataset(data[:split], args.block_size)
    val_dataset = CharDataset(data[split:], args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Vocab size: {vocab_size}  |  Train tokens: {split:,}  |  Val tokens: {len(data) - split:,}\n")

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
        load_model_state_dict(model, ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                print("Warning: optimizer state skipped (parameter layout changed); optimizer restarted.\n")
        if "step" in ckpt:
            start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}\n")

    if args.compile:
        model = torch.compile(model)

    step = start_step - 1
    train_iter = iter(train_loader)

    # Speed measurement: GPU work is async, so the clock is only read after a
    # synchronize. The first window (warmup: cuDNN autotune, allocator growth,
    # torch.compile) is excluded from the reported average, as is validation.
    if device.type == "cuda":
        torch.cuda.synchronize()
    window_start = time.perf_counter()
    window_tokens = 0
    window_steps = 0
    warmup_window = True
    measured_time = 0.0
    measured_tokens = 0

    for step in range(start_step, args.steps + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        window_tokens += x.numel()
        window_steps += 1

        if step % args.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - window_start
            tok_per_s = window_tokens / dt
            ms_per_step = dt / window_steps * 1000
            note = "  (warmup, excluded from avg)" if warmup_window else ""
            print(f"Step {step:5d}/{args.steps}  train={loss.item():.4f}  "
                  f"{tok_per_s:>10,.0f} tok/s  {ms_per_step:7.1f} ms/step{note}")
            if warmup_window:
                warmup_window = False
            else:
                measured_time += dt
                measured_tokens += window_tokens

        if step % args.val_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        vlogits = model(vx)
                        vloss = criterion(vlogits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(vloss.item())
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

        # Restart the measurement window after any pause (logging sync,
        # validation, checkpointing) so only pure training steps are timed.
        if step % args.log_interval == 0 or step % args.val_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            window_start = time.perf_counter()
            window_tokens = 0
            window_steps = 0

    if measured_tokens > 0:
        print(f"\nAvg training speed: {measured_tokens / measured_time:,.0f} tok/s "
              f"({measured_tokens:,} tokens in {measured_time:.1f}s, warmup & validation excluded)")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:,.0f} MiB")

    save_checkpoint(args.output, model, optimizer, step, tokenizer, config)
    print(f"\nModel saved → {args.output}")


if __name__ == "__main__":
    main()
