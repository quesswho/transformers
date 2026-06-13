"""Loading BabyLM zero-shot task data from local JSONL files.

The official eval data ships as one directory per task, each holding one ``.jsonl``
file per paradigm (e.g. ``blimp_filtered/adjunct_island.jsonl``). Each line is one
record; the fields differ per task and are interpreted by the adapters in
``tasks.py``. This module just walks the directory and yields ``(stem, record)``
pairs, where ``stem`` is the paradigm name (the filename without ``.jsonl``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_records(directory: str | Path) -> Iterator[tuple[str, dict]]:
    """Yield ``(paradigm_stem, record)`` for every line of every ``.jsonl`` file
    in ``directory`` (sorted for stable ordering). Blank lines are skipped."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Eval data directory not found: {directory}")
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield path.stem, json.loads(line)
