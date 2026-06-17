"""
Text preprocessing for Project Gutenberg corpora.

Applies standard NLP preprocessing to clean raw text before training:
  - Strip HTML tags and decode HTML entities
  - Normalize line endings and strip control characters
  - Unicode NFKC normalization (NBSP, ligatures, fancy quotes -> ASCII)
  - Collapse multiple inline spaces/tabs per line (preserves leading indentation)
  - Remove decorative separator lines (---, ===, ***, ___, ~~~)
  - Cap repeated punctuation runs at 3 (e.g. "......" -> "...")
  - Collapse 3+ consecutive blank lines to 2
  - Unwrap hard-wrapped prose lines (single \\n -> space)

Can be run standalone or imported as a module:

    from preprocess import preprocess_text
    cleaned = preprocess_text(raw_book_text)

    python examples/gutenberg_download/preprocess.py --input data/corpus.txt --output data/corpus_clean.txt
"""

import argparse
import html as _html
import re
import unicodedata
from pathlib import Path

_CONTROL_RE        = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE       = re.compile(r"<[^>]{0,500}>")
_SEPARATOR_LINE_RE = re.compile(r"(?m)^[ \t]*[*\-=_~#]{3,}[ \t]*$")
_INLINE_SPACE_RE   = re.compile(r"(?<=\S)[ \t]+")   # only collapses after non-whitespace -> preserves indentation
_TRAILING_SPACE_RE = re.compile(r"(?m)[ \t]+$")
_REPEAT_PUNCT_RE   = re.compile(r"([.!?\-])\1{3,}")
_MULTI_BLANK_RE    = re.compile(r"\n{3,}")
_HARD_WRAP_RE      = re.compile(r"(?<!\n)\n(?!\n)")

_CHUNK_BYTES = 32 * 1024 * 1024  # 32 MB read buffer for CLI streaming


def preprocess_text(text: str) -> str:
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Strip HTML tags, then decode entities (e.g. &amp; -> &)
    text = _HTML_TAG_RE.sub("", text)
    text = _html.unescape(text)

    # 3. NFKC: NBSP->space, ligatures, smart quotes, wide chars -> ASCII equivalents
    text = unicodedata.normalize("NFKC", text)

    # 4. Remove non-printable control characters (keep \t and \n)
    text = _CONTROL_RE.sub("", text)

    # 5. Collapse internal whitespace runs (preserves leading indentation), strip trailing, remove decorative separators
    text = _INLINE_SPACE_RE.sub(" ", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _SEPARATOR_LINE_RE.sub("", text)

    # 6. Cap repeated punctuation runs at 3 (e.g. "------" -> "---", "......" -> "...")
    text = _REPEAT_PUNCT_RE.sub(r"\1\1\1", text)

    # 7. Collapse 3+ consecutive blank lines to 2
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    # 8. Unwrap hard-wrapped prose: single \n -> space (preserves \n\n paragraph breaks)
    text = _HARD_WRAP_RE.sub(" ", text)

    # Re-collapse any double spaces introduced by the unwrap step
    text = _INLINE_SPACE_RE.sub(" ", text)

    # 9. Strip document-level leading/trailing whitespace
    return text.strip()


def _iter_paragraphs(path: Path, encoding: str = "utf-8"):
    """Stream text from *path* in ~32 MB chunks split at paragraph boundaries."""
    buf = ""
    with path.open(encoding=encoding, errors="replace") as fh:
        while True:
            chunk = fh.read(_CHUNK_BYTES)
            if not chunk:
                break
            buf += chunk
            idx = buf.rfind("\n\n")
            if idx != -1:
                yield buf[: idx + 2]
                buf = buf[idx + 2 :]
    if buf:
        yield buf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a Gutenberg corpus .txt file to remove noise."
    )
    parser.add_argument("--input",  required=True, help="Path to input .txt file")
    parser.add_argument("--output", required=True, help="Path to write cleaned output")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    raw_bytes   = input_path.stat().st_size
    raw_mb      = raw_bytes / (1024 ** 2)

    print(f"Preprocessing {raw_mb:.1f} MB ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_bytes = 0
    with output_path.open("w", encoding="utf-8") as out:
        for chunk in _iter_paragraphs(input_path):
            cleaned_chunk = preprocess_text(chunk)
            out.write(cleaned_chunk)
            out.write("\n\n")
            clean_bytes += len(cleaned_chunk.encode("utf-8")) + 2

    clean_mb  = clean_bytes / (1024 ** 2)
    reduction = (1.0 - clean_mb / raw_mb) * 100 if raw_mb else 0.0
    print(f"{raw_mb:.1f} MB -> {clean_mb:.1f} MB ({reduction:.1f}% reduction)")
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
