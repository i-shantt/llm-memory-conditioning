"""Pre-flight for the GPU arm, with a stub model. CPU-only, seconds to run.

The point is to never discover a schema mistake an hour into a Kaggle run. This
exercises the real script end to end -- real data, real retriever, real
conditioner, real grading, real payload -- and replaces only the thing that
needs a GPU.

Skips when the benchmark is not downloaded, following memllm's convention.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import memcond._deps  # noqa: E402,F401

DATA = memcond._deps.memllm_root() / "data/raw/longmemeval_oracle"
pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason=f"{DATA} not downloaded")

import run_conditioned_eval as rce  # noqa: E402
from memllm.eval.ablation import arm_from_payload, compute_lift  # noqa: E402


class _StubGen:
    def __init__(self, text: str, prompt_tokens: int) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = len(text.split())


class _StubBackend:
    """Echoes the gold-ish answer for half the questions, refuses for the rest,
    so accuracy is neither 0 nor 1 and the aggregation is actually exercised."""

    name = "stub:test"

    def __init__(self) -> None:
        self.n = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int | None = None):
        self.prompts.append(prompt)
        self.n += 1
        text = ("I don't know." if self.n % 2
                else "The answer is Target and it cost $800.")
        return _StubGen(text, prompt_tokens=len(prompt) // 4)


def _run(tmp: str, conditioner: str, backend: _StubBackend, limit: int = 6):
    argv = ["run_conditioned_eval.py", "--data", str(DATA), "--limit", str(limit),
            "--retriever", "bm25", "--conditioner", conditioner,
            "--answer-backend", "stub:test", "--out-dir", tmp,
            "--tag", f"test_{conditioner.replace(':', '-')}"]
    old_argv, old_build = sys.argv, rce.build_backend
    sys.argv = argv
    rce.build_backend = lambda spec, **kw: backend
    try:
        rce.main()
    finally:
        sys.argv, rce.build_backend = old_argv, old_build
    out = Path(tmp) / f"test_{conditioner.replace(':', '-')}.json"
    return json.loads(out.read_text())


def test_payload_satisfies_the_ablation_contract():
    """ablation.py reads exactly five things. If any is missing or misnamed the
    arm is silently unusable, which is only discovered after the GPU is spent."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _run(tmp, "identity", _StubBackend())

    for key in ("config", "accuracy", "read_tokens_per_query", "records"):
        assert key in p, f"missing {key}"
    for key in ("tag", "retriever", "answer_backend"):
        assert p["config"].get(key) is not None, f"config.{key} is None"
    for r in p["records"]:
        for key in ("question_id", "question_type", "deterministic"):
            assert key in r, f"record missing {key}"

    arm = arm_from_payload(p)
    assert arm.name and arm.model and arm.retriever == "bm25"
    assert 0.0 <= arm.accuracy <= 1.0


def test_arms_are_pairable_so_mcnemar_has_something_to_compare():
    """The whole design is a paired comparison on identical question ids. If the
    conditioner changed which questions ran, the pairing would be empty."""
    with tempfile.TemporaryDirectory() as tmp:
        base = _run(tmp, "identity", _StubBackend())
        cond = _run(tmp, "all", _StubBackend())

    a, b = arm_from_payload(base), arm_from_payload(cond)
    assert set(a.graded) == set(b.graded), "arms must cover the same questions"
    assert len(a.graded) > 0
    report = compute_lift(b, [a])
    assert report.contingency["n"] == len(a.graded)


def test_the_conditioner_actually_changes_the_prompt():
    """Guards the failure where the flag is threaded through but never applied,
    which would produce a perfect null result for a trivial reason."""
    with tempfile.TemporaryDirectory() as tmp:
        b_id, b_all = _StubBackend(), _StubBackend()
        _run(tmp, "identity", b_id)
        _run(tmp, "all", b_all)

    assert b_id.prompts and len(b_id.prompts) == len(b_all.prompts)
    assert b_id.prompts != b_all.prompts, "conditioner had no effect on the prompt"
    assert any("event 1 of" in p for p in b_all.prompts)


def test_identity_prompts_match_memllm_rendering():
    """The baseline arm must be the same experiment memllm already ran."""
    from memllm.data.loader import load_examples, stratified_subset
    from memllm.cost import CostLedger
    from memllm.retrieval.bm25 import BM25Retriever

    with tempfile.TemporaryDirectory() as tmp:
        stub = _StubBackend()
        _run(tmp, "identity", stub, limit=3)

    ex = stratified_subset(load_examples(str(DATA)), 3, seed=0)[0]
    units = ex.units("turn")
    led = CostLedger()
    r = BM25Retriever()
    r.index(units, led, None)
    by_id = {u.unit_id: u for u in units}
    hits = r.search(ex.question, 10, led, ex.question_date)
    expected = "\n\n".join(f"[{by_id[uid].session_date}] {by_id[uid].text}"
                           for uid, _ in hits if uid in by_id)
    assert expected and expected in stub.prompts[0]


def test_token_cap_flag_tracks_the_configured_maximum():
    """memllm's clamped session arms were only caught because this was recorded
    per question."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _run(tmp, "identity", _StubBackend())
    assert p["config"]["max_new_tokens"] == 256, "the truncation fix must be live"
    assert all(r["hit_token_cap"] is False for r in p["records"])
