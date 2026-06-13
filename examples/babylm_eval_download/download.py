"""
BabyLM 2026 Strict zero-shot evaluation data downloader.

Fetches the per-paradigm JSONL files for the zero-shot tasks from the official
Hugging Face dataset and lays them out the way ``examples/generative_lm/evaluate.py``
expects:

    data/eval/blimp_filtered/*.jsonl
    data/eval/supplement_filtered/*.jsonl
    data/eval/comps/*.jsonl
    data/eval/entity_tracking/*.jsonl
    data/eval/reading/reading_data.csv

  https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Evals

Uses only urllib (no extra deps), mirroring examples/babylm_download/download.py.
The file list is queried from the HF API at runtime, so new paradigms are picked
up automatically.

Note on EWoK: it is *not* redistributed in this dataset (licensing), so it is not
downloaded here. Run examples/babylm_eval_download/download_ewok.py to fetch and
filter it into data/eval/ewok_filtered/.

Run from project root:
    python examples/babylm_eval_download/download.py
    python examples/babylm_eval_download/download.py --output data/eval --split full_eval
"""

import argparse
import json
import urllib.request
from pathlib import Path

REPO = "BabyLM-community/BabyLM-2026-Strict-Evals"
API_URL = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"

# The tasks scored by src/eval. (GLUE is fine-tuning, EWoK is fetched separately —
# both are intentionally excluded here.) The minimal-pair tasks are JSONL; reading
# ships as a single CSV.
TASK_SUBDIRS = ["blimp_filtered", "supplement_filtered", "comps", "entity_tracking", "reading"]


def list_files(split: str) -> list[str]:
    """Return every JSONL/CSV path in the dataset under evaluation_data/<split>/ that
    belongs to one of TASK_SUBDIRS."""
    with urllib.request.urlopen(API_URL) as resp:
        meta = json.load(resp)
    prefix = f"evaluation_data/{split}/"
    wanted = tuple(f"{prefix}{sub}/" for sub in TASK_SUBDIRS)
    return [
        s["rfilename"]
        for s in meta["siblings"]
        if s["rfilename"].startswith(wanted) and s["rfilename"].endswith((".jsonl", ".csv"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the BabyLM 2026 Strict zero-shot eval data into a local directory."
    )
    root = Path(__file__).parent.parent.parent
    parser.add_argument("--output", default=str(root / "data" / "eval"),
                        help="Output directory for the task subdirs (default: data/eval)")
    parser.add_argument("--split", default="full_eval", choices=["full_eval", "fast_eval"],
                        help="Which eval split to fetch (default: full_eval)")
    args = parser.parse_args()

    output = Path(args.output)
    split_prefix = f"evaluation_data/{args.split}/"

    files = list_files(args.split)
    if not files:
        raise SystemExit(f"No eval files found for split {args.split!r} — check the dataset repo.")

    print(f"Downloading {len(files)} files from {REPO} ({args.split}) -> {output}")
    for rfile in files:
        # Strip the evaluation_data/<split>/ prefix so the task subdir lands at the root.
        dest = output / rfile[len(split_prefix):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"  {dest.relative_to(output)}  [cache]")
            continue
        urllib.request.urlretrieve(RESOLVE_URL.format(path=rfile), dest)
        print(f"  {dest.relative_to(output)}  [downloaded]")

    print(f"\nDone. Point evaluate.py at --data-dir {output}")


if __name__ == "__main__":
    main()
