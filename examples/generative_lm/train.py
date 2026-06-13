"""
Generative language model trainer.

Uses the TransformerStack as a decoder-only transformer (GPT-style):
a causal mask turns bidirectional self-attention into autoregressive generation.

Run from project root:
    python examples/generative_lm/train.py --data mytext.txt --output model.pt

    Save a tokenizer
    python examples/generative_lm/train.py --data mytext.txt --save-tokenizer tok.json --vocab-size 2000
    Load a tokenizer
    python examples/generative_lm/train.py --data mytext.txt --tokenizer tok.json

    build babylm_strict_small model
    python examples/generative_lm/train.py --output models/sslm7M.pt --checkpoint-dir models/chk --data data/babylm_strict_small.txt --save-tokenizer data/vocab/ssbby10k.json --vocab-size 10000 --steps 10000
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn

from transformer import DecoderOnlyTransformer, ModelConfig
from tokenizer import Tokenizer
from training import (
    Prefetcher,
    ThroughputMeter,
    get_batch,
    load_checkpoint,
    print_param_table,
    restore_training_state,
    save_checkpoint,
)

from data import load_text, load_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a generative language model.")
    parser.add_argument("--data", required=True, help="Path to a .txt training file")
    parser.add_argument("--output", default="model.pt", help="Path to save the checkpoint (default: model.pt)")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps (default: 5000)")
    parser.add_argument("--max-epochs", type=int, default=None, help="Stop after this many epochs (one full pass over the training tokens). Caps --steps when set (default: disabled)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--block-size", type=int, default=256, help="Context window length (default: 256)")
    parser.add_argument("--d-model", type=int, default=256, help="Model embedding dimension (default: 256)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 8)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers (default: 8)")
    parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward hidden dimension (default: 8/3 * d_model)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW peak learning rate (default: 1e-3)")
    parser.add_argument("--warmup-frac", type=float, default=0.03, help="Fraction of total steps spent linearly warming up the LR (default: 0.03)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final LR as a fraction of peak after cosine decay (default: 0.1)")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Unigram vocab size (default: 2000)")
    parser.add_argument("--tokenizer", default=None, help="Path to a pre-trained tokenizer.json file; skips tokenizer training")
    parser.add_argument("--save-tokenizer", default=None, help="Save the tokenizer to this JSON file after training/loading")
    parser.add_argument("--val-interval", type=int, default=500, help="Validate every N steps (default: 500)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log training speed every N steps (default: 100)")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for periodic checkpoints (default: disabled)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--vocab-only", action="store_true", help="Build and save the tokenizer then exit without training")
    parser.add_argument("--compile", default=True, action="store_true", help="torch.compile the model before training (slow first step, faster afterwards)")
    args = parser.parse_args()

    # We scale the feed-forward dimension
    # as suggested in https://arxiv.org/pdf/2002.05202
    if args.d_ff is None:
        args.d_ff = round(8 / 3 * args.d_model)
    return args


def cosine_lr(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    """LR multiplier (relative to peak): linear warmup, then cosine decay to
    min_ratio. Used as the LambdaLR schedule so the constant LR is replaced by a
    warmup + decay curve."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_tokenizer(args: argparse.Namespace, text: str) -> Tokenizer:
    if args.tokenizer is not None:
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = Tokenizer.load(args.tokenizer)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}\n")
    else:
        print(f"Training Unigram tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = Tokenizer.train(text, vocab_size=args.vocab_size)
        print(f"Tokenizer trained. Vocab size: {tokenizer.vocab_size}\n")

    if args.save_tokenizer is not None:
        save_dir = os.path.dirname(args.save_tokenizer)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(args.save_tokenizer)
        print(f"Tokenizer saved → {args.save_tokenizer}\n")
    return tokenizer


def evaluate(model, val_data, block_size, batch_size, criterion, vocab_size, device, use_amp) -> float:
    """Mean validation loss over a fixed number of random batches.

    The loss is accumulated on-device and read back with a single .item() after
    the loop. Calling .item() per batch would force a CUDA sync each iteration,
    stalling the GPU between forwards instead of letting the validation batches
    queue back-to-back.
    """
    model.eval()
    val_batches = 20
    val_loss_sum = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(val_batches):
            vx, vy = get_batch(val_data, block_size, batch_size, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                vlogits = model(vx)
                vloss = criterion(vlogits.view(-1, vocab_size), vy.view(-1))
            val_loss_sum += vloss.detach()
    return (val_loss_sum / val_batches).item()


def main() -> None:
    args = parse_args()

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
    tokenizer = build_tokenizer(args, text)

    if args.vocab_only:
        if args.save_tokenizer is None:
            print("Warning: --vocab-only set but --save-tokenizer not specified; tokenizer was not saved.")
        return

    data = load_tokens(text, tokenizer, args.data)
    vocab_size = tokenizer.vocab_size

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    print(f"Vocab size: {vocab_size}  |  Train tokens: {split:,}  |  Val tokens: {len(data) - split:,}\n")

    ckpt = None
    if args.resume:
        # The checkpoint's config is the source of truth for the architecture;
        # CLI model arguments are ignored when resuming.
        ckpt = load_checkpoint(args.resume, map_location=device)
        config = ModelConfig.from_dict(ckpt["config"])
        if config.vocab_size != vocab_size:
            sys.exit(
                f"Error: checkpoint was trained with vocab_size={config.vocab_size} "
                f"but the current tokenizer has vocab_size={vocab_size}."
            )
    else:
        config = ModelConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.block_size,
        )
    block_size = config.max_seq_len

    # Training samples random batches, so an "epoch" is defined as having seen
    # roughly as many tokens as the training set holds. --max-epochs caps the
    # step count accordingly; --steps remains the upper bound.
    total_steps = args.steps
    if args.max_epochs is not None:
        steps_per_epoch = max(1, len(train_data) // (args.batch_size * block_size))
        epoch_limit = args.max_epochs * steps_per_epoch
        total_steps = min(args.steps, epoch_limit)
        print(f"Epoch limit: {args.max_epochs} epochs × {steps_per_epoch:,} steps/epoch "
              f"= {epoch_limit:,} steps  →  training for {total_steps:,} steps\n")

    model = DecoderOnlyTransformer(config).to(device)
    print_param_table(model.count_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, fused=True)
    criterion = nn.CrossEntropyLoss()

    start_step = 1
    best_val_loss = float("inf")
    if ckpt is not None:
        start_step = restore_training_state(model, optimizer, ckpt)
        print(f"Resumed from {args.resume} at step {start_step}\n")

    # Warmup + cosine-decay LR schedule. The curve is a pure function of the
    # step and args.steps, so on resume we just fast-forward it to start_step
    # rather than persisting scheduler state (also robust if --steps changes).
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_lr(s, total_steps, warmup_steps, args.min_lr_ratio),
    )
    for _ in range(start_step - 1):
        scheduler.step()

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    step = start_step - 1

    # On CUDA, overlap the next batch's gather + H2D copy with the current
    # step's compute. The CPU path falls back to a synchronous get_batch.
    prefetcher = Prefetcher(train_data, block_size, args.batch_size, device) if device.type == "cuda" else None
    meter = ThroughputMeter(device)

    for step in range(start_step, total_steps + 1):
        model.train()
        x, y = prefetcher.next() if prefetcher is not None else get_batch(train_data, block_size, args.batch_size, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        meter.record(x.numel())

        if step % args.log_interval == 0:
            tok_per_s, ms_per_step, warmup = meter.flush()
            note = "  (warmup, excluded from avg)" if warmup else ""
            print(f"Step {step:5d}/{total_steps}  train={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"{tok_per_s:>10,.0f} tok/s  {ms_per_step:7.1f} ms/step{note}")

        if step % args.val_interval == 0:
            val_loss = evaluate(model, val_data, block_size, args.batch_size, criterion, vocab_size, device, use_amp)
            print(f"Step {step:5d}/{total_steps}  train={loss.item():.4f}  val={val_loss:.4f}")
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
            meter.reset_window()

    summary = meter.summary()
    if summary is not None:
        print(f"\n{summary}")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:,.0f} MiB")

    save_checkpoint(args.output, model, optimizer, step, tokenizer, config)
    print(f"\nModel saved → {args.output}")


if __name__ == "__main__":
    main()
