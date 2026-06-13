"""
Download + filter EWoK into the layout the `ewok` task in src/eval expects.

EWoK (``ewok-core/ewok-core-1.0``) is gated and not redistributed, so unlike the
other zero-shot tasks it can't be pulled by examples/babylm_eval_download/download.py.
This script reproduces the official filtering step: it loads the dataset, keeps only
items whose words are all in the BabyLM vocabulary (so the model isn't judged on
words it could never have seen), and writes one JSONL per domain into
``data/eval/ewok_filtered/`` — the directory the `ewok` adapter reads.

Prerequisites:
  1. pip install datasets nltk        # EWoK ships as a gated Parquet; nltk does the
                                       # word tokenization the vocab filter expects
  2. Accept the license at https://huggingface.co/datasets/ewok-core/ewok-core-1.0
  3. Authenticate so the gated download works: set HF_TOKEN=<your token>
     (or run `huggingface-cli login`)

Run from project root:
    python examples/babylm_eval_download/download_ewok.py
    python examples/babylm_eval_download/download_ewok.py --output data/eval

Note: the official script writes each item twice (once with Context/Target swapped)
to simplify its scorer. We don't — the `ewok` adapter in src/eval/tasks.py already
scores both targets in both contexts, so one copy per item gives the same per-domain
accuracy.
"""

import argparse
import json
import urllib.request
from pathlib import Path

DATASET = "ewok-core/ewok-core-1.0"
VOCAB_URL = ("https://raw.githubusercontent.com/babylm/evaluation-pipeline-2025/"
             "main/evaluation_pipeline/ewok/vocab.txt")
FIELDS = ("Context1", "Context2", "Target1", "Target2")


def load_vocab(path: str | None) -> set[str]:
    """Return the BabyLM filter vocabulary, from a local file if given else fetched
    from the official pipeline repo."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        print(f"Fetching filter vocab from {VOCAB_URL}")
        with urllib.request.urlopen(VOCAB_URL) as resp:
            text = resp.read().decode("utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def get_tokenizer():
    """Return nltk's ``word_tokenize`` (downloading its model data on first use).

    The vocab was built against nltk's tokenization — it lists split contractions
    like ``n't`` and standalone punctuation — so matching it requires the same
    tokenizer rather than a regex approximation."""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
    except ImportError:
        raise SystemExit("EWoK filtering needs nltk. Install it: pip install nltk")
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return word_tokenize


def in_vocab(record: dict, vocab: set[str], tokenize) -> bool:
    """True iff every word of every scored field of ``record`` is in ``vocab``."""
    for field in FIELDS:
        for word in tokenize(record[field].lower()):
            if word not in vocab:
                return False
    return True


def load_dataset_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("EWoK is a gated Parquet dataset. Install the reader: pip install datasets")
    try:
        return load_dataset(DATASET, split=split)
    except Exception as e:
        raise SystemExit(
            f"Could not load {DATASET} ({e}).\n"
            f"Accept the license at https://huggingface.co/datasets/{DATASET} and set "
            f"HF_TOKEN (or run `huggingface-cli login`)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and filter EWoK into data/eval/ewok_filtered/."
    )
    root = Path(__file__).parent.parent.parent
    parser.add_argument("--output", default=str(root / "data" / "eval"),
                        help="Eval data root; data is written to <output>/ewok_filtered (default: data/eval)")
    parser.add_argument("--vocab", default=None,
                        help="Path to a local vocab.txt (default: fetch the official one)")
    parser.add_argument("--split", default="test", help="Dataset split to filter (default: test)")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    tokenize = get_tokenizer()
    dataset = load_dataset_split(args.split)

    out_dir = Path(args.output) / "ewok_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_domain: dict[str, list[dict]] = {}
    kept = 0
    for record in dataset:
        if in_vocab(record, vocab, tokenize):
            by_domain.setdefault(record["Domain"], []).append(record)
            kept += 1

    for domain, items in sorted(by_domain.items()):
        dest = out_dir / f"{domain}.jsonl"
        with dest.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"  {dest.relative_to(out_dir.parent)}  [{len(items)} items]")

    print(f"\nKept {kept}/{len(dataset)} items across {len(by_domain)} domains -> {out_dir}")
    print("Score it with: python examples/generative_lm/evaluate.py "
          f"--checkpoint <ckpt> --data-dir {args.output} --tasks ewok")


if __name__ == "__main__":
    main()
