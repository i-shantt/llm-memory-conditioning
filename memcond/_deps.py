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


def require_dataset(path: str | Path) -> Path:
    """Return `path` if the benchmark split is there, or exit explaining how.

    LongMemEval is a 265 MB download and is deliberately not vendored, so the
    first command a reader runs is also the first one that can fail on a clean
    clone. `load_examples` would raise a bare FileNotFoundError, which says
    nothing about where the file comes from.
    """
    p = Path(path)
    if p.is_file():
        return p
    raise SystemExit(
        f"LongMemEval is not downloaded -- no file at {p}\n\n"
        f"It is not vendored in this repo. Fetch the split into the memllm\n"
        f"checkout with:\n\n"
        f"    pip install huggingface_hub\n"
        f"    python -c \"from huggingface_hub import hf_hub_download; \\\n"
        f"hf_hub_download('xiaowu0162/longmemeval', '{p.name}', \\\n"
        f"repo_type='dataset', local_dir='{p.parent}')\"\n\n"
        f"Nothing in tests/ needs it, and neither do\n"
        f"scripts/compare_conditioners.py or scripts/test_sort_router.py --\n"
        f"those reproduce this repo's tables from the stored arms alone."
    )


ensure_on_path()
