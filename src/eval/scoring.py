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

Runs at batch_size=1: the decoder-only stack is causal-only with no padding mask,
so there is nothing to gain from padding a batch here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def _target_logprobs(model, ids: list[int], device) -> torch.Tensor:
    """Log-prob of each token given its left context.

    Returns a 1-D tensor ``lp`` of length ``len(ids) - 1`` where ``lp[i]`` is the
    model's log-probability of ``ids[i + 1]`` conditioned on ``ids[:i + 1]``. The
    very first token has no context and is therefore never scored (the standard
    ``targets = tokens[1:]`` shift)."""
    x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)[0]  # [T, V]
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)  # predicts ids[1:]
    targets = torch.as_tensor(ids[1:], dtype=torch.long, device=device)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [T-1]


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
    total = 0.0
    # lp[i] corresponds to ids[i + 1]; the first token (i = -1) is unscored.
    for i in range(1, len(ids)):
        if offsets[i][1] > start_char:
            total += lp[i - 1].item()
    return total


def sequence_logprob(model, tokenizer, text: str, *, device="cpu") -> float:
    """Total log-probability of a whole sentence (BLiMP / supplement)."""
    return score_span(model, tokenizer, text, start_char=0, device=device)


def completion_logprob(model, tokenizer, sentence: str, completion: str, *, device="cpu") -> float:
    """Log-probability of ``completion`` as the suffix of ``sentence``, scored in
    the context of the preceding prefix (EWoK / COMPS / entity-tracking).

    The completion is located by character length from the end
    (``start_char = len(sentence) - len(completion)``)."""
    return score_span(model, tokenizer, sentence, start_char=len(sentence) - len(completion), device=device)
