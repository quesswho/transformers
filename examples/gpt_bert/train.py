"""
GPT-BERT trainer (Charpentier & Samuel, 2024).

Trains a single transformer jointly as a GPT and a BERT by mixing two
objectives at the batch level:

  * causal next-token prediction  -- a causal-masked window, standard LM loss.
  * masked-next-token prediction (MNTP) -- a bidirectional, mask-corrupted
    window whose labels are shifted by one so the masked-token prediction is
    read from the previous position's hidden state, matching the causal output
    offset. One shared output head therefore serves both objectives.

The model is a `GPTBERT`, structurally identical to the decoder-only model, so
its checkpoints load directly into `examples/generative_lm/generate.py`, the HF
export, and the BabyLM eval (all of which score causally).

Run from project root:
    python examples/gpt_bert/train.py --data data/babylm_strict_small.txt \\
        --output models/gptbert.pt --save-tokenizer data/vocab/gptbert10k.json \\
        --vocab-size 10000 --steps 10000 --mlm-ratio 0.5

`--mlm-ratio` is the fraction of steps that use the MNTP objective (the rest are
causal); `--mlm-prob` is the per-token masking probability within an MNTP batch.

`--tokenizer-type` picks the subword algorithm: unigram (default) / bpe (best
BLiMP/syntax) / morph (Morfessor morphology-aware, best EWoK/entity tracking).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
# Reuse the generative_lm corpus/tokenization helpers (load_text, load_tokens).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generative_lm"))

import torch
import torch.nn as nn

from transformer import GPTBERT, ModelConfig
from tokenizer import Tokenizer
from tokenizer.hf_tokenizer import SPECIAL_TOKENS
from training import (
    Prefetcher,
    ThroughputMeter,
    build_optimizer,
    build_scheduler,
    get_batch,
    get_mntp_batch,
    load_checkpoint,
    print_param_table,
    restore_training_state,
    save_checkpoint,
    shorten_inductor_kernel_names,
)

from data import load_text, load_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-BERT language model.")
    parser.add_argument("--data", required=True, help="Path to a .txt training file")
    parser.add_argument("--output", default="model.pt", help="Path to save the checkpoint (default: model.pt)")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps (default: 5000)")
    parser.add_argument("--max-epochs", type=int, default=None, help="Stop after this many epochs (one full pass over the training tokens). Caps --steps when set (default: disabled)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--block-size", type=int, default=256, help="Context window length (default: 256)")
    parser.add_argument("--d-model", type=int, default=384, help="Model embedding dimension (default: 384)")
    parser.add_argument("--nhead", type=int, default=6, help="Number of attention heads (default: 6)")
    parser.add_argument("--num-layers", type=int, default=10, help="Number of transformer layers (default: 10)")
    parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward hidden dimension (default: 8/3 * d_model)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW peak learning rate for embeddings, LM head, biases and norms (default: 1e-3)")
    parser.add_argument("--muon", default=True, action=argparse.BooleanOptionalAction, help="Use the hybrid Muon (2D hidden weights) + AdamW optimizer; --no-muon falls back to pure AdamW (default: enabled)")
    parser.add_argument("--muon-lr", type=float, default=2e-2, help="Muon peak learning rate for 2D weight matrices (default: 2e-2)")
    parser.add_argument("--warmup-frac", type=float, default=0.03, help="Fraction of total steps spent linearly warming up the LR (default: 0.03)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final LR as a fraction of peak after cosine decay (default: 0.1)")
    parser.add_argument("--mlm-ratio", type=float, default=0.5, help="Fraction of training steps that use the masked-next-token (BERT) objective; the rest are causal (default: 0.5)")
    parser.add_argument("--mlm-prob", type=float, default=0.15, help="Per-token masking probability within an MNTP batch (default: 0.15)")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Subword vocab size (default: 2000). At BabyLM 10M-word scale keep this modest (~8k-16k) to avoid token inflation eating the budget")
    parser.add_argument("--tokenizer-type", default="unigram", choices=["unigram", "bpe", "morph"], help="Tokenizer algorithm: unigram (default, strong all-rounder), bpe (best BLiMP/syntax), morph (Morfessor morphology-aware, best EWoK/entity tracking)")
    parser.add_argument("--tokenizer", default=None, help="Path to a pre-trained tokenizer.json file (must include <mask>); skips tokenizer training")
    parser.add_argument("--save-tokenizer", default=None, help="Save the tokenizer to this JSON file after training/loading")
    parser.add_argument("--val-interval", type=int, default=500, help="Validate every N steps (default: 500)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log training speed every N steps (default: 100)")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for periodic checkpoints (default: disabled)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--vocab-only", action="store_true", help="Build and save the tokenizer then exit without training")
    # The per-step causal/bidirectional switch is a Python bool, so Dynamo
    # compiles exactly two graphs (one per value) and reuses them -- it does NOT
    # recompile on every switch. Both shapes are identical, so there are no
    # shape-driven recompiles either. We use the default inductor mode rather
    # than "reduce-overhead": cudagraphs re-record across the objective
    # alternation (and on the freshly allocated MNTP inputs), which is slower and
    # flakier than plain inductor fusion here.
    parser.add_argument("--compile", default=True, action=argparse.BooleanOptionalAction, help="torch.compile the model in default inductor mode (use --no-compile to disable)")
    args = parser.parse_args()

    # We scale the feed-forward dimension as suggested in
    # https://arxiv.org/pdf/2002.05202
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
        print(f"Training {args.tokenizer_type} tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = Tokenizer.train(text, vocab_size=args.vocab_size, tokenizer_type=args.tokenizer_type)
        print(f"Tokenizer trained. Vocab size: {tokenizer.vocab_size}\n")

    if tokenizer.mask_token_id is None:
        sys.exit(
            "Error: the tokenizer has no <mask> token, which GPT-BERT's MNTP "
            "objective requires. Train a fresh tokenizer (omit --tokenizer) or "
            "supply one built with the current tokenizer code."
        )

    # Fertility (tokens/word) over a corpus sample: the key token-inflation
    # diagnostic — fewer tokens/word at equal vocab means better use of the
    # fixed BabyLM word budget.
    tok_per_word, unk_frac = tokenizer.fertility(text[:1_000_000])
    print(f"Fertility: {tok_per_word:.3f} tokens/word  |  <unk>: {unk_frac:.2%}\n")

    if args.save_tokenizer is not None:
        save_dir = os.path.dirname(args.save_tokenizer)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(args.save_tokenizer)
        print(f"Tokenizer saved -> {args.save_tokenizer}\n")
    return tokenizer


def evaluate(model, val_data, args, tokenizer, vocab_size, device, use_amp) -> tuple[float, float]:
    """Mean validation loss for both objectives over a fixed number of batches.

    Losses accumulate on-device and are read back with a single .item() after the
    loop so validation batches queue back-to-back instead of syncing per batch.
    """
    model.eval()
    val_batches = 20
    block_size = args.block_size
    causal_sum = torch.zeros((), device=device)
    mlm_sum = torch.zeros((), device=device)
    causal_crit = nn.CrossEntropyLoss()
    mlm_crit = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for _ in range(val_batches):
            vx, vy = get_batch(val_data, block_size, args.batch_size, device)
            mx, ml = get_mntp_batch(
                val_data, block_size, args.batch_size, device,
                mask_id=tokenizer.mask_token_id, vocab_size=vocab_size,
                n_special=len(SPECIAL_TOKENS), mask_prob=args.mlm_prob,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                causal_sum += causal_crit(model(vx, is_causal=True).view(-1, vocab_size), vy.view(-1)).detach()
                mlm_sum += mlm_crit(model(mx, is_causal=False).view(-1, vocab_size), ml.view(-1)).detach()
    return (causal_sum / val_batches).item(), (mlm_sum / val_batches).item()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")

    # Allow TF32 matmuls for the ops that remain in fp32 (free speedup on Ampere+).
    torch.set_float32_matmul_precision("high")
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
              f"= {epoch_limit:,} steps  ->  training for {total_steps:,} steps\n")

    model = GPTBERT(config).to(device)
    print_param_table(model.count_parameters())
    print(f"Objective mix: {1 - args.mlm_ratio:.0%} causal / {args.mlm_ratio:.0%} MNTP "
          f"(mask prob {args.mlm_prob:.0%})\n")

    optimizer = build_optimizer(model, muon=args.muon, muon_lr=args.muon_lr, adam_lr=args.lr, weight_decay=0.1)
    print(f"Optimizer: hybrid Muon (lr={args.muon_lr:.0e}) + AdamW (lr={args.lr:.0e})\n" if args.muon
          else f"Optimizer: AdamW (lr={args.lr:.0e})\n")
    causal_crit = nn.CrossEntropyLoss()
    mlm_crit = nn.CrossEntropyLoss(ignore_index=-100)

    start_step = 1
    best_val_loss = float("inf")
    if ckpt is not None:
        start_step = restore_training_state(model, optimizer, ckpt)
        print(f"Resumed from {args.resume} at step {start_step}\n")

    # Warmup + cosine-decay LR schedule. The curve is a pure function of the step
    # and total_steps, so on resume we just fast-forward it to start_step.
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = build_scheduler(
        optimizer,
        lambda s: cosine_lr(s, total_steps, warmup_steps, args.min_lr_ratio),
    )
    for _ in range(start_step - 1):
        scheduler.step()

    if args.compile:
        # On Windows, keep Inductor's Triton cache paths under MAX_PATH (260) so
        # long fused-kernel names don't break compilation; no-op elsewhere.
        shorten_inductor_kernel_names()
        # Default inductor mode (not reduce-overhead): compiles a causal and a
        # bidirectional graph once each, then reuses them across the objective mix.
        model = torch.compile(model)

    step = start_step - 1

    # The causal path uses the Prefetcher (overlaps gather + H2D with compute);
    # MNTP batches are built synchronously since they need CPU-side corruption.
    prefetcher = Prefetcher(train_data, block_size, args.batch_size, device) if device.type == "cuda" else None
    meter = ThroughputMeter(device)

    for step in range(start_step, total_steps + 1):
        model.train()
        is_mlm = torch.rand(()).item() < args.mlm_ratio
        if is_mlm:
            x, target = get_mntp_batch(
                train_data, block_size, args.batch_size, device,
                mask_id=tokenizer.mask_token_id, vocab_size=vocab_size,
                n_special=len(SPECIAL_TOKENS), mask_prob=args.mlm_prob,
            )
        else:
            x, target = prefetcher.next() if prefetcher is not None else get_batch(train_data, block_size, args.batch_size, device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x, is_causal=not is_mlm)
            if is_mlm:
                loss = mlm_crit(logits.view(-1, vocab_size), target.view(-1))
            else:
                loss = causal_crit(logits.view(-1, vocab_size), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        meter.record(x.numel())

        if step % args.log_interval == 0:
            tok_per_s, ms_per_step, warmup = meter.flush()
            note = "  (warmup, excluded from avg)" if warmup else ""
            obj = "mntp " if is_mlm else "causal"
            print(f"Step {step:5d}/{total_steps}  {obj} loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"{tok_per_s:>10,.0f} tok/s  {ms_per_step:7.1f} ms/step{note}")

        if step % args.val_interval == 0:
            causal_val, mlm_val = evaluate(model, val_data, args, tokenizer, vocab_size, device, use_amp)
            print(f"Step {step:5d}/{total_steps}  val causal={causal_val:.4f}  val mntp={mlm_val:.4f}")
            if args.checkpoint_dir is not None:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                save_checkpoint(os.path.join(args.checkpoint_dir, "latest.pt"), model, optimizer, step, tokenizer, config)
                # Track the best causal checkpoint: it is what downstream
                # generation and (causal) evaluation actually use.
                if causal_val < best_val_loss:
                    best_val_loss = causal_val
                    save_checkpoint(os.path.join(args.checkpoint_dir, "best.pt"), model, optimizer, step, tokenizer, config)
                    print(f"  -> new best checkpoint (val causal={causal_val:.4f})")

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
    print(f"\nModel saved -> {args.output}")


if __name__ == "__main__":
    main()
