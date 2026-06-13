"""Reading-time / eye-tracking prediction from model surprisal.

Unlike the minimal-pair tasks in ``tasks.py``, this one is *not* scored by accuracy.
The question is psycholinguistic: does the model's word-by-word **surprisal**
(``-log P(word | preceding context)``) predict how long humans look at / dwell on
each word? Surprisal is the standard linking hypothesis between a language model and
human reading effort, so this doubles as a measure of how human-like the model is.

The metric is the surprisal's *unique* contribution to predicting a reading measure,
computed with nested ordinary-least-squares regressions (NumPy only):

  * For every region (a word in a sentence) we compute the model's surprisal of the
    target word given its left context, as the full-word log-probability summed over
    the word's sub-word tokens.
  * For each dependent measure ``dv`` we fit a **baseline** OLS predicting ``dv`` from
    frequency / length / context-length (plus their pairwise interactions), then an
    **experimental** model that additionally includes surprisal (``pred``). The
    surprisal's contribution is ``ΔR² / (1 - R²_baseline)`` — the share of the
    residual variance it explains over and above the lexical controls.
  * ``eye_tracking_score`` is that contribution (×100) averaged over the eye-tracking
    reading-time measures; ``self_paced_score`` is the same for self-paced reading
    time, using a spillover baseline that also controls for the previous word's length
    and surprisal.

We also report the raw Pearson correlation between surprisal and every measure.

Input is a CSV with one row per region and these columns: ``item`` (the left-context
string), ``word`` (the target), ``prev_item`` / ``prev_word`` (the preceding region,
for spillover), the lexical controls ``Subtlex_log10`` / ``length`` /
``context_length`` / ``prev_length``, and one column per human measure listed in
``MEASURES``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scoring import completion_logprob

# Human measures to predict: eye-tracking reading times, self-paced reading time,
# and ERP components. A column is simply skipped if absent from the input CSV.
EYE_TRACKING = ["RTfirstfix", "RTfirstpass", "RTgopast", "RTrightbound"]
SELF_PACED = "self_paced_reading_time"
ERP = ["ELAN", "LAN", "N400", "P600", "EPNP", "PNP"]
MEASURES = EYE_TRACKING + [SELF_PACED] + ERP


def word_surprisal(model, tokenizer, prefix: str, word: str, *, device="cpu") -> float:
    """Surprisal ``-log P(word | prefix)`` of ``word`` continuing ``prefix``.

    This is the negated full-word log-probability, summed over the word's sub-word
    tokens. ``prefix`` is the running sentence context up to but excluding the target
    word."""
    word = word.strip()
    if not word:
        return float("nan")
    prefix = (prefix or "").strip()
    text = f"{prefix} {word}" if prefix else word
    return -completion_logprob(model, tokenizer, text, word, device=device)


def _to_float(value: str) -> float:
    """Parse a CSV cell to float, mapping blanks / non-numeric to NaN so that the
    regressions can drop those rows (complete-case analysis)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ols_r2(y: np.ndarray, predictors: list[np.ndarray]) -> float:
    """R² of an OLS fit of ``y`` on ``predictors`` (plus an intercept).

    Rows where the response or any predictor is non-finite are dropped first
    (complete-case analysis). Returns NaN when too few complete rows remain to
    identify the coefficients."""
    design = np.column_stack(predictors + [np.ones_like(y)])
    mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    y, design = y[mask], design[mask]
    if y.shape[0] <= design.shape[1]:
        return float("nan")
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    ss_res = float(residual @ residual)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _contribution(y: np.ndarray, baseline: list[np.ndarray], pred: np.ndarray) -> float:
    """Surprisal's share of residual variance: ``(R²_exp - R²_base)/(1 - R²_base)``,
    as a percentage. ``baseline`` are the control predictors; the experimental model
    adds ``pred``. Each R² is fit on its own complete-case rows."""
    r2_base = _ols_r2(y, baseline)
    r2_exp = _ols_r2(y, baseline + [pred])
    if not np.isfinite(r2_base) or not np.isfinite(r2_exp) or r2_base >= 1.0:
        return float("nan")
    return (r2_exp - r2_base) / (1.0 - r2_base) * 100.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the rows where both ``a`` and ``b`` are finite."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


@dataclass
class ReadingResult:
    n: int = 0
    correlations: dict[str, float] = field(default_factory=dict)
    eye_tracking_score: float = float("nan")
    self_paced_score: float = float("nan")


def evaluate_reading(model, tokenizer, csv_path, *, device="cpu", limit=None) -> ReadingResult:
    """Score a model on the reading dataset at ``csv_path``.

    Computes each region's model surprisal, then the surprisal→reading-time
    correlations and the eye-tracking / self-paced predictive-power scores. Measures
    absent from the CSV are skipped. ``limit`` caps the number of regions processed
    (for quick smoke runs); ``None`` uses all."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Reading data file not found: {csv_path}")

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        present = [m for m in MEASURES if m in (reader.fieldnames or [])]
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            row["pred"] = word_surprisal(model, tokenizer, row["item"], row["word"], device=device)
            prev_word = row.get("prev_word", "")
            row["prev_pred"] = (
                word_surprisal(model, tokenizer, row.get("prev_item", ""), prev_word, device=device)
                if prev_word else float("nan")
            )
            rows.append(row)

    col = lambda name: np.array([_to_float(r.get(name, "")) for r in rows])
    pred = col("pred")
    subtlex, length, ctx = col("Subtlex_log10"), col("length"), col("context_length")
    prev_len, prev_pred = col("prev_length"), col("prev_pred")

    correlations = {m: _pearson(pred, col(m)) for m in present}

    # Eye-tracking baseline: frequency, length, context length and their pairwise
    # interactions. The experimental model adds surprisal.
    eye_base = [subtlex, length, ctx, subtlex * length, subtlex * ctx, length * ctx]
    eye_scores = [_contribution(col(dv), eye_base, pred) for dv in EYE_TRACKING if dv in present]
    eye_scores = [s for s in eye_scores if np.isfinite(s)]
    eye_tracking_score = float(np.mean(eye_scores)) if eye_scores else float("nan")

    # Self-paced baseline adds spillover controls (the previous word's length and
    # surprisal) and all pairwise interactions among the five controls.
    if SELF_PACED in present:
        sp_main = [subtlex, length, ctx, prev_len, prev_pred]
        sp_inter = [a * b for i, a in enumerate(sp_main) for b in sp_main[i + 1:]]
        self_paced_score = _contribution(col(SELF_PACED), sp_main + sp_inter, pred)
    else:
        self_paced_score = float("nan")

    return ReadingResult(
        n=len(rows),
        correlations=correlations,
        eye_tracking_score=eye_tracking_score,
        self_paced_score=self_paced_score,
    )
