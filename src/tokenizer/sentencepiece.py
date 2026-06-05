"""
SentencePiece BPE tokenizer.

Key difference from standard BPE: no pre-tokenization by whitespace.
Input is treated as a raw character stream; spaces are replaced by the
▁ (U+2581) prefix on the following character, so word boundaries are
encoded inside the token strings rather than as a split point.
"""

import heapq
import json
from collections import Counter
from typing import ClassVar

SPIECE_UNDERLINE = "▁"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_ID, BOS_ID, EOS_ID = 0, 1, 2
NUM_SPECIAL = 3


class SentencePieceBPE:
    SPECIAL_TOKENS: ClassVar[list[str]] = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.vocab_inv: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._merge_rank: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        marked = self._mark_spaces(text)

        raw_words = self._split_into_words(marked)
        word_count: Counter[tuple[str, ...]] = Counter(raw_words)
        unique_words = list(word_count.keys())
        words: list[list[str]] = [list(w) for w in unique_words]
        freqs: list[int] = [word_count[w] for w in unique_words]

        # Initial vocabulary: specials + sorted unique characters
        all_chars: list[str] = sorted({ch for w in words for ch in w})
        self.vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for ch in all_chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        print(f"Initial vocab size: {len(self.vocab)} (including {NUM_SPECIAL} special tokens)")

        # Build pair counts and reverse index (pair → set of word IDs)
        pair_counts: dict[tuple[str, str], int] = {}
        pair_to_words: dict[tuple[str, str], set[int]] = {}
        for wid, (word, freq) in enumerate(zip(words, freqs)):
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
                pair_to_words.setdefault(p, set()).add(wid)

        # Max-heap via negated counts for O(log n) best-pair lookup
        heap: list[tuple[int, tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        num_merges = vocab_size - len(self.vocab)

        for step in range(num_merges):
            # Pop until we find a heap entry whose count is still current
            best_pair: tuple[str, str] | None = None
            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                if pair_counts.get(candidate, 0) == -neg_cnt:
                    best_pair = candidate
                    break

            if best_pair is None:
                break

            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            merges.append(best_pair)

            affected = pair_to_words.pop(best_pair, set())
            del pair_counts[best_pair]

            for wid in affected:
                word = words[wid]
                freq = freqs[wid]

                # Remove all pair contributions from this word
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    if p == best_pair:
                        continue
                    new_cnt = pair_counts.get(p, 0) - freq
                    if new_cnt > 0:
                        pair_counts[p] = new_cnt
                    else:
                        pair_counts.pop(p, None)
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(wid)
                        if not s:
                            del pair_to_words[p]

                # Apply merge in-place
                new_word: list[str] = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                words[wid] = new_word

                # Add pair contributions from the new word
                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    old_cnt = pair_counts.get(p, 0)
                    new_cnt = old_cnt + freq
                    pair_counts[p] = new_cnt
                    pair_to_words.setdefault(p, set()).add(wid)
                    heapq.heappush(heap, (-new_cnt, p))

            if (step + 1) % 200 == 0 or step == num_merges - 1:
                print(f"  merge {step + 1}/{num_merges}  vocab={len(self.vocab)}")

        self.merges = merges
        self.vocab_inv = {i: s for s, i in self.vocab.items()}
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        marked = self._mark_spaces(text)
        cache: dict[tuple[str, ...], list[int]] = {}
        ids: list[int] = []
        for word in self._split_into_words(marked):
            if word not in cache:
                merged = self._merge_word(word)
                cache[word] = [self.vocab.get(tok, UNK_ID) for tok in merged]
            ids.extend(cache[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab_inv.get(i, UNK_TOKEN) for i in ids]
        text = "".join(tokens)
        return text.replace(SPIECE_UNDERLINE, " ").lstrip(" ")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "merges": list(self.merges),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "SentencePieceBPE":
        tok = cls()
        tok.vocab = data["vocab"]
        tok.vocab_inv = {int(i): s for s, i in tok.vocab.items()}
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok._merge_rank = {tuple(pair): rank for rank, pair in enumerate(tok.merges)}
        return tok

    @classmethod
    def load(cls, path: str) -> "SentencePieceBPE":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_spaces(text: str) -> str:
        """Replace spaces with ▁ prepended to the following word."""
        pieces = text.split(" ")
        if not pieces:
            return text
        return pieces[0] + "".join(
            SPIECE_UNDERLINE + p if p else SPIECE_UNDERLINE
            for p in pieces[1:]
        )

    @staticmethod
    def _split_into_words(marked: str) -> list[tuple[str, ...]]:
        """Split ▁-marked text into character-tuple 'words'."""
        parts = marked.split(SPIECE_UNDERLINE)
        words: list[tuple[str, ...]] = []
        if parts[0]:
            words.append(tuple(parts[0]))
        for p in parts[1:]:
            words.append(tuple(SPIECE_UNDERLINE + p))
        return words

    def _merge_word(self, word: tuple[str, ...]) -> tuple[str, ...]:
        """Apply all learned merge rules to a single word — O(L log L)."""
        if len(word) < 2:
            return word

        n = len(word)
        symbols: list[str | None] = list(word)
        merges_len = len(self.merges)

        # Array-based doubly linked list over symbol positions
        prev = list(range(-1, n - 1))   # prev[0] = -1
        nxt = list(range(1, n + 1))     # nxt[n-1] = n (sentinel >= n)

        # Seed the heap with all valid adjacent pairs
        heap: list[tuple[int, int]] = []
        for i in range(n - 1):
            r = self._merge_rank.get((symbols[i], symbols[i + 1]), merges_len)  # type: ignore[arg-type]
            if r < merges_len:
                heapq.heappush(heap, (r, i))

        while heap:
            rank, i = heapq.heappop(heap)

            if symbols[i] is None:
                continue
            j = nxt[i]
            if j >= n or symbols[j] is None:
                continue

            pair = (symbols[i], symbols[j])
            actual_rank = self._merge_rank.get(pair, merges_len)  # type: ignore[arg-type]
            if actual_rank != rank:
                # Stale entry; re-push with corrected rank if still mergeable
                if actual_rank < merges_len:
                    heapq.heappush(heap, (actual_rank, i))
                continue

            new_tok = symbols[i] + symbols[j]  # type: ignore[operator]
            symbols[i] = new_tok
            symbols[j] = None

            # Splice j out of the linked list
            nxt_j = nxt[j]
            nxt[i] = nxt_j
            if nxt_j < n:
                prev[nxt_j] = i

            # Push new left-neighbor pair
            pi = prev[i]
            if pi >= 0 and symbols[pi] is not None:
                r = self._merge_rank.get((symbols[pi], new_tok), merges_len)  # type: ignore[arg-type]
                if r < merges_len:
                    heapq.heappush(heap, (r, pi))

            # Push new right-neighbor pair
            ni = nxt[i]
            if ni < n and symbols[ni] is not None:
                r = self._merge_rank.get((new_tok, symbols[ni]), merges_len)  # type: ignore[arg-type]
                if r < merges_len:
                    heapq.heappush(heap, (r, i))

        return tuple(s for s in symbols if s is not None)
