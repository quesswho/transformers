"""Run minimal-pair accuracy for a task and aggregate the results.

For each ``Example`` we score every candidate and the model is "correct" when the
labelled candidate (index 0 in BabyLM data) scores strictly highest. Results are
collected per ``uid`` (paradigm/subset); the headline task accuracy is the macro
average over uids — matching how the official pipeline averages subdomain
accuracies — and we also report the micro count.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .data import iter_records
from .scoring import completion_logprob, sequence_logprob
from .tasks import TASKS, Candidate, Example


def _score_candidate(model, tokenizer, candidate: Candidate, device) -> float:
    kind = candidate[0]
    if kind == "seq":
        return sequence_logprob(model, tokenizer, candidate[1], device=device)
    if kind == "comp":
        return completion_logprob(model, tokenizer, candidate[1], candidate[2], device=device)
    raise ValueError(f"Unknown candidate kind: {kind!r}")


def _is_correct(scores: list[float], label: int) -> bool:
    """Correct only when ``label`` is the unique maximum. Ties count as wrong,
    which is the conventional (chance-averse) BLiMP scoring."""
    best = max(scores)
    return scores[label] == best and sum(s == best for s in scores) == 1


@dataclass
class TaskResult:
    correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, uid: str, correct: bool) -> None:
        self.total[uid] += 1
        self.correct[uid] += int(correct)

    @property
    def uid_accuracy(self) -> dict[str, float]:
        return {uid: self.correct[uid] / self.total[uid] for uid in sorted(self.total)}

    @property
    def accuracy(self) -> float:
        """Macro average over paradigms (mean of per-uid accuracies)."""
        accs = self.uid_accuracy
        return sum(accs.values()) / len(accs) if accs else 0.0

    @property
    def n(self) -> int:
        return sum(self.total.values())


def evaluate_task(model, tokenizer, directory, adapter, *, device="cpu", limit=None) -> TaskResult:
    """Score every example produced by ``adapter`` over the JSONL files in
    ``directory``. ``limit`` caps the number of records read *per paradigm file*
    (so a capped run still samples every paradigm — handy for smoke tests and
    training-time checks); ``None`` runs the full set."""
    result = TaskResult()
    seen: dict[str, int] = defaultdict(int)
    for stem, record in iter_records(directory):
        if limit is not None and seen[stem] >= limit:
            continue
        seen[stem] += 1
        for example in adapter(record, stem):
            scores = [_score_candidate(model, tokenizer, c, device) for c in example.candidates]
            result.add(example.uid, _is_correct(scores, example.label))
    return result


def evaluate_tasks(model, tokenizer, data_root, tasks=None, *, device="cpu", limit=None) -> dict[str, TaskResult]:
    """Evaluate several tasks rooted at ``data_root`` (one subdirectory each).

    ``tasks`` is a list of names from ``TASKS`` (default: all). Tasks whose data
    directory is missing are skipped silently so a partial download still works."""
    from pathlib import Path

    data_root = Path(data_root)
    names = tasks if tasks is not None else list(TASKS)
    results: dict[str, TaskResult] = {}
    for name in names:
        spec = TASKS[name]
        directory = data_root / spec.subdir
        if not directory.is_dir():
            continue
        results[name] = evaluate_task(
            model, tokenizer, directory, spec.adapter, device=device, limit=limit
        )
    return results
