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
    parser.add_argument("--compile", action="store_true", help="torch.compile the model before generating (slow first run, faster afterwards)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = DecoderOnlyTransformer.from_checkpoint(args.model, map_location=device)
    model = model.to(device)
    tokenizer = SentencePieceBPE.from_dict(checkpoint["tokenizer"])

    if args.compile:
        model = torch.compile(model)

    tokens = tokenizer.encode(args.prompt)
    if not tokens:
        print("Error: prompt encoded to empty token sequence.")
        sys.exit(1)

    ctx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=args.steps, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
