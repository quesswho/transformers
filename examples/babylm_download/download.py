"""
BabyLM 2026 Strict-Small corpus downloader.

Downloads the six source files of the official 10M-token "Strict-Small" training
set from the Hugging Face Hub and concatenates them into a single corpus file
ready for use with train.py.

  https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small

The text is already cleaned and detoxified upstream, so no preprocessing is
applied here beyond newline normalization.

Run from project root:
    python examples/babylm_download/download.py
    python examples/babylm_download/download.py --output data/babylm_strict_small.txt
"""

import argparse
import urllib.request
from pathlib import Path

REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{name}}"

# The six source corpora that make up the 10M-token Strict-Small split.
FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the BabyLM 2026 Strict-Small corpus into a single training file."
    )
    root = Path(__file__).parent.parent.parent
    parser.add_argument("--output", default=str(root / "data" / "babylm_strict_small.txt"),
                        help="Output corpus file (default: data/babylm_strict_small.txt)")
    parser.add_argument("--cache-dir", default=str(root / "data" / "babylm"),
                        help="Directory for the individual source files (default: data/babylm)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output = Path(args.output)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output, "w", encoding="utf-8") as out:
        for i, name in enumerate(FILES):
            cache_path = cache_dir / name
            if cache_path.exists():
                label = "cache"
            else:
                url = BASE_URL.format(name=name)
                print(f"Downloading {name} ...")
                urllib.request.urlretrieve(url, cache_path)
                label = "downloaded"

            # The source files use Unix line endings; normalize defensively so
            # the concatenated corpus is consistent regardless of platform.
            text = cache_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()

            if i > 0:
                out.write("\n\n")
            out.write(text)

            size = len(text.encode("utf-8"))
            total += size
            print(f"  {name:24s} {size / (1024*1024):6.2f} MB  [{label}]")

    print(f"\nCorpus: {total / (1024*1024):.1f} MB")
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
