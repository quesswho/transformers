"""
Bulk Project Gutenberg downloader.

Downloads English plain-text books until a target corpus size is reached,
then writes a single concatenated file ready for use with train.py.

Run from project root:
    python examples/gutenberg_download/download.py
    python examples/gutenberg_download/download.py --target-mb 200
    python examples/gutenberg_download/download.py --target-mb 500 --output data/corpus.txt

Then train:
    python examples/generative_lm/train.py --data data/gutenberg_corpus.txt
"""

import argparse
import csv
import os
import re
import time
import urllib.request
from pathlib import Path

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL    = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
SEPARATOR   = "\n\n" + "=" * 72 + "\n\n"

_START_RE = re.compile(r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^\*]+\*{3}", re.IGNORECASE)
_END_RE   = re.compile(r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^\*]+\*{3}",   re.IGNORECASE)


def _strip(raw: str) -> str:
    s = _START_RE.search(raw)
    e = _END_RE.search(raw)
    if s is None or e is None:
        return raw
    return raw[s.end():e.start()].strip()


def _catalog_ids(cache_dir: Path) -> list[int]:
    path = cache_dir / "pg_catalog.csv"
    if not path.exists():
        print("Downloading Gutenberg catalog CSV...")
        urllib.request.urlretrieve(CATALOG_URL, path)
    ids = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("Type") == "Text" and "en" in row.get("Language", ""):
                try:
                    ids.append(int(row["Text#"]))
                except ValueError:
                    pass
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-download Project Gutenberg books into a training corpus.")
    root = Path(__file__).parent.parent.parent
    parser.add_argument("--target-mb",  type=float, default=500.0,
                        help="Target corpus size in MB (default: 500)")
    parser.add_argument("--output",     default=str(root / "data" / "gutenberg_corpus.txt"),
                        help="Output corpus file (default: data/gutenberg_corpus.txt)")
    parser.add_argument("--cache-dir",  default=str(root / "data" / "gutenberg"),
                        help="Directory for individual book cache (default: data/gutenberg)")
    parser.add_argument("--delay",      type=float, default=1.0,
                        help="Seconds between downloads (default: 1.0)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output    = Path(args.output)
    target    = int(args.target_mb * 1024 * 1024)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    book_ids = _catalog_ids(cache_dir)
    print(f"Catalog: {len(book_ids)} English texts found. Target: {args.target_mb:.0f} MB\n")

    total = downloaded = skipped = 0

    with open(output, "w", encoding="utf-8") as out:
        first = True
        for book_id in book_ids:
            if total >= target:
                break

            cache_path = cache_dir / f"pg{book_id}.txt"

            if cache_path.exists():
                with open(cache_path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                label = "cache"
            else:
                url = BOOK_URL.format(id=book_id)
                try:
                    time.sleep(args.delay)
                    urllib.request.urlretrieve(url, cache_path)
                except Exception as exc:
                    if cache_path.exists():
                        os.unlink(cache_path)
                    print(f"  SKIP {book_id}: {exc}")
                    skipped += 1
                    continue

                try:
                    with open(cache_path, encoding="utf-8", errors="replace") as f:
                        raw = f.read()
                except Exception as exc:
                    os.unlink(cache_path)
                    print(f"  SKIP {book_id}: {exc}")
                    skipped += 1
                    continue

                text = _strip(raw)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                label = "downloaded"
                downloaded += 1

            if not text:
                skipped += 1
                continue

            size = len(text.encode("utf-8"))
            if not first:
                out.write(SEPARATOR)
            out.write(text)
            out.flush()
            first = False
            total += size
            print(f"  {total // (1024*1024):4d}/{args.target_mb:.0f} MB  book {book_id}  {size//1024} KB  [{label}]")

    print(f"\nCorpus: {total / (1024*1024):.1f} MB  ({downloaded} downloaded, {skipped} skipped)")
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()
