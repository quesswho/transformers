"""Zero-shot evaluation framework scored directly against the repo's native causal LM.

The framework is generic: any task that reduces to a minimal-pair log-likelihood
comparison (pick the candidate string the model assigns the higher log-prob) can be
plugged in as an adapter. See ``scoring.py`` for the scoring primitive,
``minimal_pairs.py`` for the accuracy runner, and ``tasks.py`` for the registry of
bundled tasks (currently BLiMP, supplement, EWoK, COMPS, entity-tracking)."""

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
