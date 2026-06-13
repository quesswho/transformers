"""Zero-shot evaluation for BabyLM tasks (BLiMP, supplement, EWoK, COMPS,
entity-tracking) scored directly against the repo's native causal LM.

All tasks reduce to minimal-pair log-likelihood comparison; see ``scoring.py`` for
the primitive, ``tasks.py`` for per-task adapters, and ``minimal_pairs.py`` for the
accuracy runner."""

from .minimal_pairs import TaskResult, evaluate_task, evaluate_tasks
from .scoring import completion_logprob, score_span, sequence_logprob
from .tasks import TASKS, Example, TaskSpec

__all__ = [
    "TASKS",
    "Example",
    "TaskSpec",
    "TaskResult",
    "evaluate_task",
    "evaluate_tasks",
    "score_span",
    "sequence_logprob",
    "completion_logprob",
]
