"""Locate the memllm checkout and put it on the path.

memllm is a sibling repo, not a published package, so there is nothing to pip
install. It supplies the cost ledger, the audited grader, and the lift/McNemar
statistics -- reusing them is the point, because a result from this repo is only
worth anything if it is measured the same way memllm's numbers were.

Resolution order: $MEMLLM_PATH, a sibling directory, then $HOME/memllm. Kaggle
clones both repos side by side, which the sibling case covers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent


def memllm_root() -> Path:
    candidates = []
    if os.environ.get("MEMLLM_PATH"):
        candidates.append(Path(os.environ["MEMLLM_PATH"]).expanduser())
    candidates += [_HERE.parent / "memllm", Path.home() / "memllm"]
    for c in candidates:
        if (c / "memllm" / "eval" / "grade.py").is_file():
            return c.resolve()
    raise ImportError(
        "cannot find the memllm checkout. Clone it next to this repo, or set "
        "MEMLLM_PATH=/path/to/memllm. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def ensure_on_path() -> Path:
    root = memllm_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


ensure_on_path()
