"""Tests for locating the two external things this repo does not vendor.

Both failures here happen on someone's first command against a clean clone, so
the message matters more than usual: memllm is a sibling checkout and
LongMemEval is a 265 MB download, and neither is guessable from a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import memcond._deps as deps  # noqa: E402


def test_require_dataset_returns_the_path_when_the_split_is_there(tmp_path):
    split = tmp_path / "longmemeval_s"
    split.write_text("[]")
    assert deps.require_dataset(split) == split
    assert deps.require_dataset(str(split)) == split


def test_a_missing_dataset_exits_with_the_download_command_not_a_traceback(tmp_path):
    """`load_examples` would raise a bare FileNotFoundError, which does not tell
    a first-time reader that the file is a HuggingFace download."""
    missing = tmp_path / "longmemeval_s"
    with pytest.raises(SystemExit) as e:
        deps.require_dataset(missing)

    msg = str(e.value)
    assert "hf_hub_download" in msg, "must name the way to get the file"
    assert "xiaowu0162/longmemeval" in msg, "must name the dataset"
    assert missing.name in msg, "must name the split being fetched"
    assert str(missing.parent) in msg, "must say where to put it"
    assert "tests/" in msg, "must say what still works without it"


def test_a_directory_is_not_mistaken_for_the_split(tmp_path):
    """The split is a single file. Pointing --data at data/raw/ is an easy slip
    and must fail loudly rather than at some later open()."""
    with pytest.raises(SystemExit):
        deps.require_dataset(tmp_path)


def test_memllm_root_names_every_place_it_looked(monkeypatch):
    """The sibling checkout is the single most common setup mistake."""
    monkeypatch.delenv("MEMLLM_PATH", raising=False)
    monkeypatch.setattr(deps, "_HERE", Path("/nonexistent/repo/pkg"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/nonexistent")))

    with pytest.raises(ImportError) as e:
        deps.memllm_root()

    msg = str(e.value)
    assert "MEMLLM_PATH" in msg, "must name the escape hatch"
    assert "Looked in:" in msg and "/nonexistent" in msg, "must show the paths tried"
