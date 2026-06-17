"""Corpus loading and tokenization for the generative LM example.

These helpers know about the tokenizer and the text corpus, so they stay local
to the example rather than living in the model-agnostic src/training package.
"""

import hashlib
import json
import os

import numpy as np


def load_text(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _tokenizer_fingerprint(tokenizer) -> str:
    # Hash the whole serialised tokenizer (not just tokenizer_json) so a morph
    # tokenizer's Morfessor model is also folded into the cache key.
    raw = json.dumps(tokenizer.to_dict(), sort_keys=True)
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
        print(f"Token cache written -> {cache_path}")

    return arr
