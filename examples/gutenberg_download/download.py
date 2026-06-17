"""
Bulk Project Gutenberg downloader.

Downloads English plain-text books until a target corpus size is reached,
then writes a single concatenated file ready for use with train.py.

Run from project root:
    python examples/gutenberg_download/download.py
    python examples/gutenberg_download/download.py --target-mb 200
    python examples/gutenberg_download/download.py --target-mb 500 --output data/corpus.txt
"""

import argparse
import csv
import os
import re
import time
import urllib.request
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from preprocess import preprocess_text

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL    = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
SEPARATOR   = "\n\n" + "=" * 72 + "\n\n"

_START_RE = re.compile(r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^\*]+\*{3}", re.IGNORECASE)
_END_RE   = re.compile(r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^\*]+\*{3}",   re.IGNORECASE)


def _minhash(text: str, num_perm: int = 128, k: int = 5) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(max(0, len(words) - k + 1)):
        mh.update(" ".join(words[i:i + k]).encode())
    return mh


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
    parser.add_argument("--no-preprocess", action="store_true", default=False,
                        help="Skip text preprocessing (write raw stripped text to corpus)")
    parser.add_argument("--dedup-threshold", type=float, default=0.8,
                        help="Jaccard similarity threshold for near-duplicate detection (0–1, 1.0 = disable, default: 0.8)")
    parser.add_argument("--min-words", type=int, default=500,
                        help="Minimum word count per book (default: 500)")
    parser.add_argument("--min-alpha", type=float, default=0.65,
                        help="Minimum alphabetic character ratio per book (default: 0.65)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output    = Path(args.output)
    target    = int(args.target_mb * 1024 * 1024)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    book_ids = _catalog_ids(cache_dir)
    print(f"Catalog: {len(book_ids)} English texts found. Target: {args.target_mb:.0f} MB\n")

    lsh = MinHashLSH(threshold=args.dedup_threshold, num_perm=128) if args.dedup_threshold < 1.0 else None
    total = downloaded = skipped = 0

    log_path = output.with_name(output.stem + "_log.csv")
    log_file = open(log_path, "w", encoding="utf-8", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["book_id", "word_count", "alpha_ratio", "accepted", "reject_reason"])

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

            if not args.no_preprocess:
                text = preprocess_text(text)

            if not text:
                skipped += 1
                continue

            word_count = len(text.split())
            non_ws = [c for c in text if not c.isspace()]
            alpha_ratio = sum(c.isalpha() for c in non_ws) / len(non_ws) if non_ws else 0.0

            if word_count < args.min_words:
                print(f"  SKIP {book_id}: too short ({word_count} words)")
                log_writer.writerow([book_id, word_count, f"{alpha_ratio:.3f}", False, "too_short"])
                skipped += 1
                continue

            if alpha_ratio < args.min_alpha:
                print(f"  SKIP {book_id}: low alpha ratio ({alpha_ratio:.2f})")
                log_writer.writerow([book_id, word_count, f"{alpha_ratio:.3f}", False, "low_alpha"])
                skipped += 1
                continue

            if lsh is not None:
                mh = _minhash(text)
                if lsh.query(mh):
                    print(f"  SKIP {book_id}: near-duplicate")
                    log_writer.writerow([book_id, word_count, f"{alpha_ratio:.3f}", False, "near_duplicate"])
                    skipped += 1
                    continue
                lsh.insert(str(book_id), mh)

            size = len(text.encode("utf-8"))
            if not first:
                out.write(SEPARATOR)
            out.write(text)
            out.flush()
            first = False
            total += size
            log_writer.writerow([book_id, word_count, f"{alpha_ratio:.3f}", True, ""])
            print(f"  {total // (1024*1024):4d}/{args.target_mb:.0f} MB  book {book_id}  {size//1024} KB  [{label}]")

    log_file.close()
    print(f"\nCorpus: {total / (1024*1024):.1f} MB  ({downloaded} downloaded, {skipped} skipped)")
    print(f"Saved -> {output}")
    print(f"Log   -> {log_path}")


if __name__ == "__main__":
    main()
