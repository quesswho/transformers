"""Corpus loading and tokenization for the generative LM example.

These helpers know about the tokenizer and the text corpus, so they stay local
to the example rather than living in the model-agnostic src/training package.
"""

import hashlib
import os
import tempfile
import urllib.request

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path: str | None) -> str:
    """Read a UTF-8 text file, or auto-download tinyshakespeare when path is None."""
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.close()
    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(DATA_URL, tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        text = f.read()
    os.unlink(tmp.name)
    return text


def _tokenizer_fingerprint(tokenizer) -> str:
    d = tokenizer.to_dict()
    raw = str(sorted(d.get("vocab", {}).items())) + str(d.get("merges", []))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_tokens(text: str, tokenizer, data_path: str | None) -> np.ndarray:
    """Encode text to an int32 token array, caching the result next to data_path.

    The cache key folds in the file's size/mtime and a fingerprint of the
    tokenizer, so changing either the corpus or the tokenizer invalidates it.
    """
    cache_path = None
    if data_path is not None:
        stat = os.stat(data_path)
        key = f"{stat.st_size}_{int(stat.st_mtime)}_{_tokenizer_fingerprint(tokenizer)}"
        cache_path = data_path + f".{key}.tokens.npy"
        if os.path.exists(cache_path):
            print(f"Loading cached tokens from {cache_path}...")
            return np.load(cache_path, mmap_mode="r")

    print("Encoding corpus...")
    tokens = tokenizer.encode(text)
    arr = np.asarray(tokens, dtype=np.int32)

    if cache_path is not None:
        np.save(cache_path, arr)
        print(f"Token cache written → {cache_path}")

    return arr
