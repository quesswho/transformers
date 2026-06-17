"""Configurable subword tokenizers built on HuggingFace `tokenizers`.

Three tokenizer types are supported (the 2025 BabyLM findings show tokenizer
choice is disproportionately high-leverage at the 10M-100M word scale — see
logs/2025.babylm-main.28.md §D.4):

* ``unigram`` (default) — SentencePiece-style Unigram; a strong all-rounder that
  tends to edge out BPE on morphology-sensitive benchmarks like BLiMP.
* ``bpe`` — byte-pair encoding; the findings report the strongest *syntactic*
  acceptability judgments (BLiMP) from BPE.
* ``morph`` — morphology-aware: an unsupervised Morfessor model splits words into
  morphemes before a Unigram model runs, shifting competence toward semantics /
  discourse (large EWoK and entity-tracking gains in the findings).

Each tokenizer is wrapped in a `PreTrainedTokenizerFast` so the same tokenizer
loads into the BabyLM evaluation harness via `AutoTokenizer.from_pretrained`
(``morph`` excepted — its custom pre-tokenizer is re-attached on load instead;
see `morfessor_pretok`).
"""

import json

import numpy as np
from tokenizers import Regex, Tokenizer as HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from . import morfessor_pretok

# Ordered so the unknown token lands at id 0, matching the old tokenizer; the
# remaining specials get the next contiguous ids. Read ids back with
# `token_to_id` rather than assuming positions if you depend on them.
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
MASK_TOKEN = "<mask>"
# <mask> is appended last so the existing unk/pad/bos/eos ids (0..3) are
# unchanged; it feeds the GPT-BERT masked-next-token-prediction objective.
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]

TOKENIZER_TYPES = ("unigram", "bpe", "morph")


def _normalizer() -> normalizers.Normalizer:
    """NFKC + NMT whitespace cleanup — the standard SentencePiece front-end,
    shared by all three tokenizer types so specials and offsets line up."""
    return normalizers.Sequence([
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ])


def _wrap(backend: HFTokenizer) -> PreTrainedTokenizerFast:
    """Wrap a raw `tokenizers` object in the HF fast-tokenizer interface."""
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        mask_token=MASK_TOKEN,
    )


def _detect_type(backend: HFTokenizer) -> str:
    """Infer the type of a plain (unigram/bpe) tokenizer from its model field."""
    model_type = json.loads(backend.to_str()).get("model", {}).get("type", "")
    return "bpe" if model_type.lower() == "bpe" else "unigram"


def _wrap_morph(backend: HFTokenizer, model) -> PreTrainedTokenizerFast:
    """Wrap a Unigram backend as a morph tokenizer.

    `PreTrainedTokenizerFast` deep-copies (pickles) the backend at construction,
    which a custom Python pre-tokenizer can't survive. So `backend` must carry a
    plain Metaspace pre-tokenizer when passed here; we re-attach the Morfessor
    pre-tokenizer onto the wrapped copy afterwards (encoding reads it live)."""
    fast = _wrap(backend)
    fast.backend_tokenizer.pre_tokenizer = morfessor_pretok.build_pre_tokenizer(model)
    return fast


class Tokenizer:
    """Thin wrapper presenting the small surface the rest of the repo uses
    (`train`/`encode`/`decode`/`vocab_size`/`save`/`load`/`to_dict`) over a
    HuggingFace fast tokenizer."""

    def __init__(
        self,
        fast: PreTrainedTokenizerFast,
        *,
        tokenizer_type: str = "unigram",
        morfessor_model=None,
    ) -> None:
        self._fast = fast
        self._type = tokenizer_type
        self._morfessor = morfessor_model

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, text: str, vocab_size: int, tokenizer_type: str = "unigram") -> "Tokenizer":
        """Train a tokenizer of ``tokenizer_type`` on `text` (one document/line)."""
        if tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(
                f"tokenizer_type must be one of {TOKENIZER_TYPES}, got {tokenizer_type!r}"
            )

        morfessor_model = None
        if tokenizer_type == "bpe":
            backend = HFTokenizer(models.BPE(unk_token=UNK_TOKEN))
            backend.pre_tokenizer = pre_tokenizers.Metaspace()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS
            )
        else:
            # unigram and morph both use a Unigram core; morph differs only in
            # the pre-tokenizer (morphemes vs. plain words).
            backend = HFTokenizer(models.Unigram())
            if tokenizer_type == "morph":
                morfessor_model = morfessor_pretok.train_morfessor(text)
                backend.pre_tokenizer = morfessor_pretok.build_pre_tokenizer(morfessor_model)
            else:
                backend.pre_tokenizer = pre_tokenizers.Metaspace()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=SPECIAL_TOKENS,
                unk_token=UNK_TOKEN,
            )

        backend.normalizer = _normalizer()
        backend.decoder = decoders.Metaspace()  # Metaspace marks word boundaries with ▁
        # Feed lines lazily so we never copy the whole corpus into a list.
        backend.train_from_iterator(
            (line for line in text.splitlines() if line.strip()), trainer
        )

        if tokenizer_type == "morph":
            # Swap the custom pre-tokenizer for a serialisable Metaspace one so
            # the backend can be wrapped (deep-copied), then re-attach Morfessor.
            backend.pre_tokenizer = pre_tokenizers.Metaspace()
            return cls(_wrap_morph(backend, morfessor_model),
                       tokenizer_type="morph", morfessor_model=morfessor_model)
        return cls(_wrap(backend), tokenizer_type=tokenizer_type)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load a tokenizer from a `tokenizer.json` (unigram/bpe) or a `morph`
        bundle written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") == "morph":
            return cls._from_morph(data)
        backend = HFTokenizer.from_file(path)
        return cls(_wrap(backend), tokenizer_type=_detect_type(backend))

    @classmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        """Rebuild from the dict embedded in a checkpoint (see `to_dict`)."""
        if data.get("type") == "morph":
            return cls._from_morph(data)
        backend = HFTokenizer.from_str(data["tokenizer_json"])
        return cls(_wrap(backend), tokenizer_type=_detect_type(backend))

    @classmethod
    def _from_morph(cls, data: dict) -> "Tokenizer":
        """Rebuild a morph tokenizer: load its Morfessor model and re-attach the
        custom pre-tokenizer that a plain tokenizer.json can't carry."""
        model = morfessor_pretok.load_model(data["morfessor_model"])
        # tokenizer_json was saved with a plain Metaspace pre-tokenizer; _wrap_morph
        # re-attaches the Morfessor pre-tokenizer after the wrapping deep-copy.
        backend = HFTokenizer.from_str(data["tokenizer_json"])
        return cls(_wrap_morph(backend, model), tokenizer_type="morph", morfessor_model=model)

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
    # Properties / diagnostics
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        # len() counts every id the model must be able to emit (base vocab plus
        # any added tokens); the model's output projection is sized from this.
        return len(self._fast)

    @property
    def tokenizer_type(self) -> str:
        return self._type

    @property
    def mask_token_id(self) -> int | None:
        """Id of the <mask> token, or None for tokenizers trained before it was
        added. Required by the GPT-BERT masked-next-token-prediction objective."""
        return self._fast.mask_token_id

    def fertility(self, text: str) -> tuple[float, float]:
        """Return (tokens-per-word, <unk>-fraction) over `text`.

        Fertility is the key diagnostic for "token inflation": an oversized or
        ill-fitting vocab spends more tokens per word, silently shrinking the
        fixed BabyLM word budget. Pass a representative sample for large corpora.
        """
        n_words = max(1, len(text.split()))
        ids = self._fast.encode(text, add_special_tokens=False)
        n_unk = sum(1 for i in ids if i == self._fast.unk_token_id)
        return len(ids) / n_words, n_unk / max(1, len(ids))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _core_json(self) -> str:
        """Serialise the backend, swapping morph's unserialisable custom
        pre-tokenizer for a plain Metaspace one (re-attached on load)."""
        backend = self._fast.backend_tokenizer
        if self._type != "morph":
            return backend.to_str()
        saved = backend.pre_tokenizer
        backend.pre_tokenizer = pre_tokenizers.Metaspace()
        try:
            return backend.to_str()
        finally:
            backend.pre_tokenizer = saved

    def to_dict(self) -> dict:
        """Serialise for embedding in a checkpoint. Always includes
        ``tokenizer_json``; morph additionally carries its Morfessor model."""
        out = {"tokenizer_json": self._core_json()}
        if self._type == "morph":
            out["type"] = "morph"
            out["morfessor_model"] = morfessor_pretok.dump_model(self._morfessor)
        return out

    def save(self, path: str) -> None:
        """Save the tokenizer. unigram/bpe write a single-file `tokenizer.json`;
        morph writes a JSON bundle (core + Morfessor model) reloadable by
        :meth:`load`."""
        if self._type != "morph":
            self._fast.backend_tokenizer.save(path)
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    def save_pretrained(self, directory: str) -> None:
        """Write a full HF tokenizer directory (config + tokenizer.json) so the
        BabyLM eval can load it with `AutoTokenizer.from_pretrained`.

        Not supported for ``morph``: its Morfessor pre-tokenizer is a Python
        callback that a standalone tokenizer.json cannot represent. Use
        :meth:`save` / :meth:`load` (or the embedded checkpoint tokenizer) for
        morph instead."""
        if self._type == "morph":
            raise NotImplementedError(
                "morph tokenizers can't be exported via save_pretrained (custom "
                "Morfessor pre-tokenizer is not serialisable into tokenizer.json). "
                "Use save()/load() or the checkpoint-embedded tokenizer."
            )
        self._fast.save_pretrained(directory)
