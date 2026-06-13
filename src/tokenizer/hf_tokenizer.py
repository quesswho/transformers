"""SentencePiece-style Unigram tokenizer built on HuggingFace `tokenizers`.

We train a Unigram model (the SentencePiece algorithm that tends to edge out
BPE on morphology-sensitive benchmarks like BLiMP) and wrap it in 
a `PreTrainedTokenizerFast` so the same tokenizer loads directly into the BabyLM 
evaluation harness via `AutoTokenizer.from_pretrained`.
"""

import numpy as np
from tokenizers import Regex, Tokenizer as HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# Ordered so the unknown token lands at id 0, matching the old tokenizer; the
# remaining specials get the next contiguous ids. Read ids back with
# `token_to_id` rather than assuming positions if you depend on them.
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


def _wrap(backend: HFTokenizer) -> PreTrainedTokenizerFast:
    """Wrap a raw `tokenizers` object in the HF fast-tokenizer interface."""
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
    )


class Tokenizer:
    """Thin wrapper presenting the small surface the rest of the repo uses
    (`train`/`encode`/`decode`/`vocab_size`/`save`/`load`/`to_dict`) over a
    HuggingFace fast tokenizer."""

    def __init__(self, fast: PreTrainedTokenizerFast) -> None:
        self._fast = fast

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "Tokenizer":
        """Train a Unigram tokenizer on `text` (one document per line)."""
        backend = HFTokenizer(models.Unigram())
        # NFKC + NMT whitespace cleanup is the standard SentencePiece front-end;
        # Metaspace marks word boundaries with ▁ (U+2581) like SentencePiece.
        backend.normalizer = normalizers.Sequence([
            normalizers.Nmt(),
            normalizers.NFKC(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ])
        backend.pre_tokenizer = pre_tokenizers.Metaspace()
        backend.decoder = decoders.Metaspace()

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token=UNK_TOKEN,
        )
        # Feed lines lazily so we never copy the whole corpus into a list.
        backend.train_from_iterator(
            (line for line in text.splitlines() if line.strip()), trainer
        )
        return cls(_wrap(backend))

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load a tokenizer from a `tokenizer.json` file."""
        return cls(_wrap(HFTokenizer.from_file(path)))

    @classmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        """Rebuild from the dict embedded in a checkpoint (see `to_dict`)."""
        return cls(_wrap(HFTokenizer.from_str(data["tokenizer_json"])))

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode to an int32 array of token ids (no special tokens added)."""
        ids = self._fast.encode(text, add_special_tokens=False)
        return np.asarray(ids, dtype=np.int32)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode to token ids plus per-token (start, end) character spans.

        The character offsets let log-likelihood scoring isolate the tokens that
        belong to a sentence's completion (e.g. the EWoK/COMPS target phrase),
        which is exactly how the official BabyLM zero-shot pipeline locates the
        scored span. No special tokens are added."""
        enc = self._fast(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def decode(self, ids) -> str:
        return self._fast.decode(list(ids), skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        # len() counts every id the model must be able to emit (base vocab plus
        # any added tokens); the model's output projection is sized from this.
        return len(self._fast)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise for embedding in a checkpoint."""
        return {"tokenizer_json": self._fast.backend_tokenizer.to_str()}

    def save(self, path: str) -> None:
        """Save a single-file `tokenizer.json`."""
        self._fast.backend_tokenizer.save(path)

    def save_pretrained(self, directory: str) -> None:
        """Write a full HF tokenizer directory (config + tokenizer.json) so the
        BabyLM eval can load it with `AutoTokenizer.from_pretrained`."""
        self._fast.save_pretrained(directory)
