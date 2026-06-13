"""Export a training checkpoint (.pt) to a self-contained HuggingFace directory.

The trainer's `save_checkpoint` bundles the native model weights, the tokenizer,
and the `ModelConfig` into a single .pt file. The BabyLM evaluation harness, on
the other hand, calls `AutoModelForCausalLM.from_pretrained(path)` /
`AutoTokenizer.from_pretrained(path)` in its own process. This script bridges the
two: it rebuilds the HF wrapper, loads the trained weights, and writes a directory
that loads standalone via `trust_remote_code` (config + weights + bundled
modeling/config .py + tokenizer).

Run from project root:
    python examples/generative_lm/export_hf.py --checkpoint models/chk/best.pt --output exported/baby-strict-small

Then verify it loads from a *fresh* process (one that never imports `src/hf`):
    python examples/generative_lm/export_hf.py --checkpoint models/chk/best.pt --output exported/baby-strict-small --verify

Push to the Hub (required for an actual BabyLM submission):
    python examples/generative_lm/export_hf.py --checkpoint models/chk/best.pt --output exported/baby-strict-small --push-to-hub <user>/baby-strict-small
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from transformer import ModelConfig
from tokenizer import Tokenizer
from training import load_checkpoint
from hf import BabyTransformerConfig, BabyTransformerForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a .pt checkpoint to a HuggingFace model directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trainer .pt checkpoint")
    parser.add_argument("--output", required=True, help="Directory to write the HF model + tokenizer to")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Also push the exported model + tokenizer to this HF Hub repo id (e.g. user/baby-strict-small)")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private (default: public)")
    parser.add_argument("--verify", action="store_true",
                        help="After export, reload the directory with trust_remote_code and run a forward pass")
    return parser.parse_args()


def export(checkpoint: str, output: str) -> None:
    # Checkpoint is loaded onto CPU; export is weight-shuffling, not training.
    ckpt = load_checkpoint(checkpoint, map_location="cpu")

    model_config = ModelConfig.from_dict(ckpt["config"])
    hf_config = BabyTransformerConfig.from_model_config(model_config)

    model = BabyTransformerForCausalLM(hf_config)
    # The checkpoint holds the *inner* native model's state dict, which maps 1:1
    # onto the wrapper's `.transformer` submodule.
    model.transformer.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)

    tokenizer = Tokenizer.from_dict(ckpt["tokenizer"])
    tokenizer.save_pretrained(output)

    print(f"Exported HF model + tokenizer -> {output}")
    print(f"  vocab_size={hf_config.vocab_size}  d_model={hf_config.d_model}  "
          f"num_layers={hf_config.num_layers}  max_seq_len={hf_config.max_seq_len}")


def verify(output: str) -> None:
    """Reload the exported directory the way the eval harness does and run a
    forward pass. trust_remote_code is required because the architecture lives in
    the bundled .py files, not in `transformers` itself."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(output)
    model.eval()

    enc = tokenizer("The quick brown fox", return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape[-1] == model.config.vocab_size, "logits vocab dim mismatch"
    print(f"Verify OK: loaded with trust_remote_code, logits {tuple(out.logits.shape)}")


def main() -> None:
    args = parse_args()
    export(args.checkpoint, args.output)

    if args.verify:
        verify(args.output)

    if args.push_to_hub is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(args.output, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.output)
        model.push_to_hub(args.push_to_hub, private=args.private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.private)
        print(f"Pushed -> https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
