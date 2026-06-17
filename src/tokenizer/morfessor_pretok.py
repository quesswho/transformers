"""Morphology-aware pre-tokenizer backed by an unsupervised Morfessor model.

This powers the ``morph`` tokenizer type. A Morfessor Baseline model is trained
on the corpus's word-type counts (unsupervised, no external data — Strict-track
legal), then used at pre-tokenization time to split each word into morphemes
*before* the subword (Unigram) model runs. The 2025 BabyLM findings report that
morphology-aware token boundaries give large gains on semantic tasks (~+20% EWoK,
~+40% entity tracking) at the 10M-word scale (see logs/2025.babylm-main.28.md
§D.4).

The split is expressed as slices of HuggingFace's ``NormalizedString``, so
per-token character offsets are preserved — ``Tokenizer.encode_with_offsets``
(and the BabyLM zero-shot scorer in src/eval that depends on it) keep working.

A custom Python pre-tokenizer cannot be embedded in a plain ``tokenizer.json``,
so the Morfessor model is serialised separately (pickle + base64) and the custom
pre-tokenizer is re-attached when the tokenizer is loaded (see hf_tokenizer.py).
"""

import base64
import collections
import pickle
import re

from tokenizers import NormalizedString, PreTokenizedString, pre_tokenizers

# ▁ (U+2581): the Metaspace word-boundary marker. Metaspace prepends it to each
# word; we keep it attached to the first morpheme so word boundaries survive.
MARKER = "▁"

# Maximal runs of letters (Unicode-aware, no digits/underscore). Only these are
# fed to Morfessor; punctuation and mixed tokens are left for the subword model.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def train_morfessor(text: str, *, corpusweight: float = 1.0):
    """Train an unsupervised Morfessor Baseline model on word-type counts.

    Imported lazily so unigram/bpe users don't need ``morfessor`` installed.
    """
    import morfessor

    counts = collections.Counter(
        match.group().lower()
        for line in text.splitlines()
        for match in _WORD_RE.finditer(line)
    )
    model = morfessor.BaselineModel(corpusweight=corpusweight)
    # Morfessor keys compounds by their atom tuple (here, characters).
    model.load_data([(count, tuple(word)) for word, count in counts.items()])
    model.train_batch()
    return model


class MorfessorPreTokenizer:
    """HF custom pre-tokenizer: split each Metaspace word chunk into morphemes."""

    def __init__(self, model) -> None:
        self.model = model
        self._cache: dict[str, list[int]] = {}

    def _boundaries(self, token: str) -> list[int]:
        """Cut indices into ``token`` at morpheme boundaries (cached per token)."""
        cuts = self._cache.get(token)
        if cuts is not None:
            return cuts

        # The leading Metaspace marker (if present) keeps its character slot and
        # stays with the first morpheme; Morfessor segments the bare word.
        offset = 1 if token[:1] == MARKER else 0
        word = token[offset:]
        if len(word) > 1 and word.isalpha():
            # viterbi_segment returns a list of morphs, each a tuple of atoms
            # (characters); morph length therefore counts characters.
            morphs, _ = self.model.viterbi_segment(tuple(word.lower()))
            cuts, pos = [], offset
            for morph in morphs[:-1]:
                pos += len(morph)
                cuts.append(pos)
        else:
            cuts = []
        self._cache[token] = cuts
        return cuts

    def morph_split(self, _index: int, normalized: NormalizedString):
        token = str(normalized)
        cuts = self._boundaries(token)
        if not cuts:
            return [normalized]
        pieces, prev = [], 0
        for cut in (*cuts, len(token)):
            pieces.append(normalized[prev:cut])
            prev = cut
        return pieces

    def pre_tokenize(self, pretok: PreTokenizedString) -> None:
        pretok.split(self.morph_split)


def build_pre_tokenizer(model):
    """Metaspace (word boundaries) then Morfessor (morpheme boundaries)."""
    return pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(),
        pre_tokenizers.PreTokenizer.custom(MorfessorPreTokenizer(model)),
    ])


def dump_model(model) -> str:
    """Serialise a Morfessor model to a base64 string for the morph bundle."""
    return base64.b64encode(pickle.dumps(model)).decode("ascii")


def load_model(blob: str):
    """Inverse of :func:`dump_model`."""
    return pickle.loads(base64.b64decode(blob.encode("ascii")))
