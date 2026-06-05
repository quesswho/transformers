"""
Load a saved generative LM checkpoint and generate text without training.

Run from project root:
    python examples/generative_lm/generate.py --model model.pt --prompt "ROMEO:" --steps 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from transformer import DecoderOnlyTransformer
from tokenizer import SentencePieceBPE


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a saved generative LM checkpoint.")
    parser.add_argument("--model", default="model.pt", help="Path to the .pt checkpoint file (default: model.pt)")
    parser.add_argument("--prompt", default="ROMEO:", help="Seed text to start generation from")
    parser.add_argument("--steps", type=int, default=500, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    tokenizer = SentencePieceBPE.from_dict(checkpoint["tokenizer"])

    model = DecoderOnlyTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["block_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokens = tokenizer.encode(args.prompt)
    if not tokens:
        print("Error: prompt encoded to empty token sequence.")
        sys.exit(1)

    ctx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=args.steps, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
