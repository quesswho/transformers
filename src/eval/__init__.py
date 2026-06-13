"""Zero-shot evaluation framework scored directly against the repo's native causal LM.

Most tasks reduce to a minimal-pair log-likelihood comparison (pick the candidate
string the model assigns the higher log-prob) and plug in as an adapter — see
``scoring.py`` for the scoring primitive, ``minimal_pairs.py`` for the accuracy
runner, and ``tasks.py`` for the registry of bundled tasks (currently BLiMP,
supplement, EWoK, COMPS, entity-tracking). ``reading.py`` is a separate, regression-
based task: it predicts human reading times from the model's word surprisal."""

from .minimal_pairs import TaskResult, evaluate_task, evaluate_tasks
from .reading import ReadingResult, evaluate_reading, word_surprisal
from .scoring import completion_logprob, score_span, sequence_logprob
from .tasks import TASKS, Example, TaskSpec

__all__ = [
    "TASKS",
    "Example",
    "TaskSpec",
    "TaskResult",
    "evaluate_task",
    "evaluate_tasks",
    "ReadingResult",
    "evaluate_reading",
    "word_surprisal",
    "score_span",
    "sequence_logprob",
    "completion_logprob",
]
