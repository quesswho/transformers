"""Run minimal-pair accuracy for a task and aggregate the results.

For each ``Example`` we score every candidate and the model is "correct" when the
labelled candidate scores strictly highest. Results are collected per ``uid``
(paradigm/subset); the headline task accuracy is the macro average over uids (the
conventional way to average subdomain accuracies), and we also report the micro
count.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .data import iter_records
from .scoring import score_spans_batched, score_spans_batched_mlm
from .tasks import TASKS, Candidate, Example


def _count_records(directory, limit) -> int:
    """Number of records ``evaluate_task`` will process, for the progress total.

    Counts non-blank lines per ``.jsonl`` file (matching ``iter_records``) and
    applies the per-file ``limit`` the same way the evaluation loop does."""
    total = 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        total += n if limit is None else min(n, limit)
    return total


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


def evaluate_task(
    model, tokenizer, directory, adapter, *, device="cpu", limit=None, batch_size: int = 256,
    mask_id: int | None = None,
) -> TaskResult:
    """Score every example produced by ``adapter`` over the JSONL files in
    ``directory``.

    Candidates are packed into batches of ``batch_size`` and scored with a single
    forward pass each, which gives a large speedup over the previous one-at-a-time
    approach.  ``limit`` caps the number of records read *per paradigm file* (so a
    capped run still samples every paradigm — handy for smoke tests and
    training-time checks); ``None`` runs the full set.

    When ``mask_id`` is given, candidates are scored through GPT-BERT's bidirectional
    masked-next-token path (pseudo-log-likelihood) instead of the causal path — the
    mode GPT-BERT scores strongest in; the model must accept ``forward(.., is_causal=
    False)`` and the tokenizer must have a ``<mask>`` token."""
    result = TaskResult()
    seen: dict[str, int] = defaultdict(int)

    pending_examples: list[Example] = []
    pending_items: list[tuple[str, int]] = []
    pending_map: list[tuple[int, int]] = []
    pending_records = 0  # records represented in the current pending batch

    pbar = tqdm(total=_count_records(directory, limit), unit="rec",
                leave=False, dynamic_ncols=True)

    def flush() -> None:
        nonlocal pending_records
        if not pending_items:
            return
        if mask_id is None:
            scores_flat = score_spans_batched(model, tokenizer, pending_items, device=device)
        else:
            scores_flat = score_spans_batched_mlm(
                model, tokenizer, pending_items, mask_id=mask_id, device=device, batch_size=batch_size
            )
        per_example: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for k, (ex_idx, cand_idx) in enumerate(pending_map):
            per_example[ex_idx].append((cand_idx, scores_flat[k]))
        for ex_idx, example in enumerate(pending_examples):
            pairs = sorted(per_example[ex_idx])
            result.add(example.uid, _is_correct([s for _, s in pairs], example.label))
        pbar.update(pending_records)
        pbar.set_postfix(acc=f"{result.accuracy:.3f}", n=result.n)
        pending_examples.clear()
        pending_items.clear()
        pending_map.clear()
        pending_records = 0

    for stem, record in iter_records(directory):
        if limit is not None and seen[stem] >= limit:
            continue
        seen[stem] += 1
        pending_records += 1
        for example in adapter(record, stem):
            ex_idx = len(pending_examples)
            pending_examples.append(example)
            for cand_idx, candidate in enumerate(example.candidates):
                kind = candidate[0]
                if kind == "seq":
                    pending_items.append((candidate[1], 0))
                elif kind == "comp":
                    pending_items.append((candidate[1], len(candidate[1]) - len(candidate[2])))
                else:
                    raise ValueError(f"Unknown candidate kind: {kind!r}")
                pending_map.append((ex_idx, cand_idx))
            if len(pending_items) >= batch_size:
                flush()

    flush()
    pbar.close()
    return result


def evaluate_tasks(
    model, tokenizer, data_root, tasks=None, *, device="cpu", limit=None, batch_size: int = 256,
    mask_id: int | None = None,
) -> dict[str, TaskResult]:
    """Evaluate several tasks rooted at ``data_root`` (one subdirectory each).

    ``tasks`` is a list of names from ``TASKS`` (default: all). Tasks whose data
    directory is missing are skipped silently so a partial download still works.
    ``mask_id`` selects GPT-BERT's masked (bidirectional) scoring — see
    ``evaluate_task``."""
    from pathlib import Path

    data_root = Path(data_root)
    names = [n for n in (tasks if tasks is not None else list(TASKS))
             if (data_root / TASKS[n].subdir).is_dir()]
    results: dict[str, TaskResult] = {}
    for name in tqdm(names, desc="tasks", unit="task", dynamic_ncols=True):
        spec = TASKS[name]
        directory = data_root / spec.subdir
        results[name] = evaluate_task(
            model, tokenizer, directory, spec.adapter,
            device=device, limit=limit, batch_size=batch_size, mask_id=mask_id,
        )
    return results
