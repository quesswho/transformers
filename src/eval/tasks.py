"""Task adapters: map a raw JSONL record to scoreable minimal-pair candidates.

Each adapter turns one record into one or more ``Example``s. An ``Example`` is a
list of candidate strings to score plus the index of the correct one (the bundled
adapters all list the good/acceptable variant first, so ``label=0``) and a ``uid``
used to group results into paradigms/subsets for reporting.

A candidate is one of:
  ``("seq", text)``                 -> score the whole sentence (BLiMP / supplement)
  ``("comp", sentence, completion)``-> score only ``completion`` as the suffix of
                                       ``sentence``, in the context of its prefix
                                       (EWoK / COMPS / entity-tracking)

To add a new task, write an adapter ``(record, stem) -> list[Example]`` and register
it in ``TASKS`` below with the subdirectory its JSONL data lives in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Candidate = tuple  # ("seq", text) | ("comp", sentence, completion)


@dataclass
class Example:
    candidates: list[Candidate]
    label: int
    uid: str


def _blimp(record: dict, stem: str) -> list[Example]:
    """BLiMP and the BLiMP supplement: a good/bad sentence pair, full-sentence
    scored. Supplement files carry no ``UID``, so fall back to the paradigm stem."""
    good, bad = record["sentence_good"], record["sentence_bad"]
    return [Example(
        candidates=[("seq", good), ("seq", bad)],
        label=0,
        uid=record.get("UID", stem),
    )]


def _ewok(record: dict, stem: str) -> list[Example]:
    """EWoK: a target is more likely under its matching context than the opposing
    one. Each record yields two examples (one per target), completion-scored on the
    target only."""
    c1, c2 = record["Context1"], record["Context2"]
    t1, t2 = record["Target1"], record["Target2"]
    uid = record.get("Domain", stem)
    return [
        Example(
            candidates=[("comp", f"{c1} {t1}", f" {t1}"), ("comp", f"{c2} {t1}", f" {t1}")],
            label=0, uid=uid,
        ),
        Example(
            candidates=[("comp", f"{c2} {t2}", f" {t2}"), ("comp", f"{c1} {t2}", f" {t2}")],
            label=0, uid=uid,
        ),
    ]


def _comps(record: dict, stem: str) -> list[Example]:
    """COMPS: a property phrase is more likely after the acceptable concept's
    prefix than the unacceptable one's. Completion = the shared property phrase."""
    prop = record["property_phrase"]
    accept = f'{record["prefix_acceptable"]} {prop}'
    unaccept = f'{record["prefix_unacceptable"]} {prop}'
    return [Example(
        candidates=[("comp", accept, prop), ("comp", unaccept, prop)],
        label=0, uid=stem,
    )]


def _entity_tracking(record: dict, stem: str) -> list[Example]:
    """Entity tracking: of several continuations of a box-contents prefix, the
    correct option (listed first) should score highest. Completion = each option."""
    prefix = record["input_prefix"]
    options = record["options"]
    uid = f'{stem}_{record["numops"]}_ops'
    return [Example(
        candidates=[("comp", prefix + opt, opt) for opt in options],
        label=0, uid=uid,
    )]


@dataclass
class TaskSpec:
    subdir: str                                   # directory name under the data root
    adapter: Callable[[dict, str], list[Example]]


# Registry: task name -> where its data lives + how to read it. Add new tasks here.
TASKS: dict[str, TaskSpec] = {
    "blimp": TaskSpec("blimp_filtered", _blimp),
    "supplement": TaskSpec("supplement_filtered", _blimp),
    "ewok": TaskSpec("ewok_filtered", _ewok),
    "comps": TaskSpec("comps", _comps),
    "entity_tracking": TaskSpec("entity_tracking", _entity_tracking),
}
