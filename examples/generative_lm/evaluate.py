"""
Zero-shot evaluation of a trained checkpoint on the BabyLM tasks.

Loads a trainer ``.pt`` checkpoint (model weights + bundled tokenizer + config) and
scores it on the zero-shot minimal-pair tasks — BLiMP, the BLiMP supplement, EWoK,
COMPS, entity-tracking — printing per-paradigm and per-task accuracy, plus a reading
task that measures how well the model's word surprisal predicts human reading times.
This runs the repo's native model directly (no HuggingFace export / trust_remote_code),
so it is fast enough to use as a training-quality signal.

First fetch the eval data:
    python examples/babylm_eval_download/download.py

Then, from project root:
    python examples/generative_lm/evaluate.py --checkpoint models/chk/best.pt
    python examples/generative_lm/evaluate.py --checkpoint models/chk/best.pt --tasks blimp,comps
    python examples/generative_lm/evaluate.py --checkpoint models/chk/best.pt --limit 50 --json out.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from transformer import DecoderOnlyTransformer
from tokenizer import Tokenizer
from eval import TASKS, evaluate_reading, evaluate_task

# A regression-based task that lives outside the minimal-pair TASKS registry: it
# scores reading-time prediction rather than accuracy, so it is handled separately.
READING = "reading"
ALL_TASKS = list(TASKS) + [READING]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot eval of a checkpoint on BabyLM tasks.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trainer .pt checkpoint")
    parser.add_argument("--data-dir", default="data/eval",
                        help="Directory holding the task subdirs (default: data/eval)")
    parser.add_argument("--tasks", default="all",
                        help=f"Comma-separated subset of {{{','.join(ALL_TASKS)}}}, or 'all' (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap records read per paradigm file (for quick smoke runs; default: full set)")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Also write the full results (per-paradigm accuracies) to this JSON file")
    return parser.parse_args()


def load_model(checkpoint: str, device):
    model, ckpt = DecoderOnlyTransformer.from_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    tokenizer = Tokenizer.from_dict(ckpt["tokenizer"])
    return model, tokenizer


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tasks = ALL_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Choose from {ALL_TASKS}.")

    print(f"Loading {args.checkpoint} onto {device} ...", flush=True)
    model, tokenizer = load_model(args.checkpoint, device)

    from pathlib import Path
    data_root = Path(args.data_dir)
    print(f"Scoring tasks: {', '.join(tasks)}  (data: {data_root})\n", flush=True)

    report = {}
    accuracies = []
    missing = []
    # Score one task at a time and print its row as soon as it finishes, so a long
    # full-set run shows progress instead of sitting silent until the end.
    minimal_pair = [t for t in tasks if t != READING]
    if minimal_pair:
        print(f"{'task':<16}{'paradigms':>10}{'examples':>10}{'accuracy':>10}")
        print("-" * 46, flush=True)
    for name in minimal_pair:
        directory = data_root / TASKS[name].subdir
        if not directory.is_dir():
            missing.append(name)
            continue
        r = evaluate_task(model, tokenizer, directory, TASKS[name].adapter,
                          device=device, limit=args.limit)
        print(f"{name:<16}{len(r.uid_accuracy):>10}{r.n:>10}{r.accuracy:>9.1%}", flush=True)
        accuracies.append(r.accuracy)
        report[name] = {"accuracy": r.accuracy, "n": r.n, "paradigms": r.uid_accuracy}
    if minimal_pair:
        print("-" * 46)
    if accuracies:
        print(f"{'macro avg':<16}{'':>10}{'':>10}{sum(accuracies) / len(accuracies):>9.1%}")

    # Reading is scored by how well surprisal predicts reading times (not accuracy),
    # so it gets its own block rather than a row in the table above.
    if READING in tasks:
        csv_path = data_root / READING / "reading_data.csv"
        if not csv_path.is_file():
            missing.append(READING)
        else:
            print("\nreading  (how well surprisal predicts human reading times)", flush=True)
            rr = evaluate_reading(model, tokenizer, csv_path, device=device, limit=args.limit)
            print(f"  eye-tracking score : {rr.eye_tracking_score:6.2f}")
            print(f"  self-paced score   : {rr.self_paced_score:6.2f}")
            print(f"  surprisal correlations (n={rr.n}):")
            for measure, corr in rr.correlations.items():
                print(f"    {measure:<24}{corr:+.4f}")
            report[READING] = {
                "eye_tracking_score": rr.eye_tracking_score,
                "self_paced_score": rr.self_paced_score,
                "n": rr.n,
                "correlations": rr.correlations,
            }

    if missing:
        print(f"\nSkipped (no data under {data_root}): {', '.join(missing)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote full results -> {args.json}")


if __name__ == "__main__":
    main()
