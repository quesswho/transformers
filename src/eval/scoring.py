"""Per-token log-likelihood scoring for a causal LM.

Every minimal-pair task (BLiMP, EWoK, COMPS, entity-tracking) reduces to the same
question: which of two (or more) candidate strings does the model assign the higher
log-probability to? This module provides that primitive against the repo's native
``DecoderOnlyTransformer`` (whose ``forward(ids)`` returns logits ``[B, T, V]``).

The score of a span is the **sum** (not the mean) of ``log_softmax`` log-probs over
the tokens in that span, where token ``t`` is predicted from the tokens before it.
Tokens are selected by their character offset, so a "completion" (e.g. the EWoK
target) can be scored in the context of its prefix while only the completion's
tokens contribute.

The primary entry-point for bulk evaluation is ``score_spans_batched``, which packs
multiple candidates into a single forward pass (right-padding is safe for causal
models: real tokens never attend to trailing pad positions).  The single-item helpers
``score_span`` / ``sequence_logprob`` / ``completion_logprob`` are kept for
one-off use (e.g. reading-time surprisal).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def _target_logprobs(model, ids: list[int], device) -> torch.Tensor:
    """Log-prob of each token given its left context.

    Returns a 1-D tensor of length ``len(ids) - 1`` where element ``i`` is the
    model's log-probability of ``ids[i + 1]`` conditioned on ``ids[:i + 1]``."""
    x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)[0]  # [T, V]
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)  # predicts ids[1:]
    targets = torch.as_tensor(ids[1:], dtype=torch.long, device=device)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [T-1]


@torch.inference_mode()
def _batch_target_logprobs(model, ids_list: list[list[int]], device) -> list[torch.Tensor]:
    """Batch forward pass over multiple sequences.

    Sequences are right-padded to the same length; causal attention ensures that
    no real token attends to a pad position, so scores are identical to running
    each sequence individually.  Returns one [T-1] tensor per input sequence."""
    B = len(ids_list)
    lengths = [len(ids) for ids in ids_list]
    max_len = max(lengths)
    padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        padded[i, :lengths[i]] = torch.as_tensor(ids, dtype=torch.long)
    logits = model(padded)                                          # [B, T, V]
    lp_all = F.log_softmax(logits[:, :-1].float(), dim=-1)        # [B, T-1, V]
    results = []
    for i in range(B):
        L = lengths[i]
        targets = padded[i, 1:L].unsqueeze(-1)                    # [L-1, 1]
        results.append(lp_all[i, :L - 1].gather(-1, targets).squeeze(-1))  # [L-1]
    return results


def _span_sum(lp: torch.Tensor, offsets: list[tuple[int, int]], start_char: int) -> float:
    """Sum log-probs for tokens whose character end-offset is after ``start_char``."""
    total = 0.0
    for i in range(1, len(offsets)):
        if offsets[i][1] > start_char:
            total += lp[i - 1].item()
    return total


def score_span(model, tokenizer, text: str, *, start_char: int = 0, device="cpu") -> float:
    """Summed log-prob of the tokens of ``text`` whose character span ends after
    ``start_char``.

    ``start_char=0`` scores the whole string (every token except the unscoreable
    first one) — used for full-sentence BLiMP. A positive ``start_char`` restricts
    scoring to a trailing completion — used for EWoK / COMPS / entity-tracking."""
    ids, offsets = tokenizer.encode_with_offsets(text)
    if len(ids) < 2:
        return 0.0
    lp = _target_logprobs(model, ids, device)
    return _span_sum(lp, offsets, start_char)


@torch.inference_mode()
def score_spans_batched(
    model,
    tokenizer,
    items: list[tuple[str, int]],
    *,
    device="cpu",
) -> list[float]:
    """Score a list of ``(text, start_char)`` pairs in a single forward pass.

    Equivalent to calling ``score_span`` for each item individually but orders of
    magnitude faster because all sequences are packed into one batched forward call.
    Sequences with fewer than two tokens score 0.0 (nothing to predict)."""
    if not items:
        return []
    encoded = [tokenizer.encode_with_offsets(text) for text, _ in items]
    ids_list = [ids for ids, _ in encoded]
    offsets_list = [offs for _, offs in encoded]
    valid = [len(ids) >= 2 for ids in ids_list]
    scoreable = [ids for ids, ok in zip(ids_list, valid) if ok]
    if not scoreable:
        return [0.0] * len(items)
    lp_tensors = _batch_target_logprobs(model, scoreable, device)
    results: list[float] = []
    lp_iter = iter(lp_tensors)
    for i in range(len(items)):
        if not valid[i]:
            results.append(0.0)
        else:
            results.append(_span_sum(next(lp_iter), offsets_list[i], items[i][1]))
    return results


def sequence_logprob(model, tokenizer, text: str, *, device="cpu") -> float:
    """Total log-probability of a whole sentence (BLiMP / supplement)."""
    return score_span(model, tokenizer, text, start_char=0, device=device)


def completion_logprob(model, tokenizer, sentence: str, completion: str, *, device="cpu") -> float:
    """Log-probability of ``completion`` as the suffix of ``sentence``, scored in
    the context of the preceding prefix (EWoK / COMPS / entity-tracking).

    The completion is located by character length from the end
    (``start_char = len(sentence) - len(completion)``)."""
    return score_span(model, tokenizer, sentence, start_char=len(sentence) - len(completion), device=device)
