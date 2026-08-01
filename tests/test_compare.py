"""Tests for the paired-comparison analysis, on synthetic arms.

This script runs as the last cell of the GPU notebook. If it throws there, the
arms survive but the analysis does not, and the numbers have to be recomputed by
hand. Cases here are built so the contingency table is known by construction.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import compare_conditioners as cc  # noqa: E402

PY = sys.executable
TYPES = ["knowledge-update", "temporal-reasoning"]


def payload(conditioner: str, correct: list[bool], *, model: str = "ollama:qwen2.5:7b-instruct",
            tokens: float = 2600.0, prompt_tokens=None) -> dict:
    """An arm where question i is answered correctly iff correct[i]."""
    records = []
    for i, ok in enumerate(correct):
        records.append({
            "question_id": f"q{i}", "question": f"question {i}?",
            "question_type": TYPES[i % len(TYPES)],
            "gold": "g", "pred": "p", "is_abstention": False,
            "extractive": True, "deterministic": ok, "judge": None,
            "raw_verdict": "",
            "prompt_tokens": (prompt_tokens if prompt_tokens is not None
                              else 1000 + i),
            "completion_tokens": 10, "hit_token_cap": False,
        })
    return {
        "config": {"tag": f"cond_{conditioner}", "retriever": "bm25",
                   "answer_backend": model, "conditioner": conditioner,
                   "answer_backend_name": model},
        "n_examples": len(correct),
        "accuracy": sum(1 for c in correct if c) / len(correct),
        "n_graded": len(correct), "n_not_gradable": 0,
        "token_f1_mean": 0.0, "accuracy_by_question_type": {},
        "cost_total": {}, "read_tokens_per_query": tokens,
        "prompt_tokens_max": 2000, "n_hit_token_cap": 0,
        "records": records,
    }


def write(tmp: Path, name: str, p: dict) -> None:
    (tmp / f"cond_{name}.json").write_text(json.dumps(p))


def run(tmp: Path) -> tuple[str, list]:
    out = tmp / "cmp.json"
    r = subprocess.run(
        [PY, str(REPO / "scripts/compare_conditioners.py"),
         "--results", str(tmp), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, f"script failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout, json.loads(out.read_text())


def test_a_conditioner_that_fixes_questions_shows_a_positive_significant_delta():
    """20 questions: the baseline gets 2, the conditioner gets those 2 plus 12
    more and loses none. A one-sided 14-vs-0 discordant split is decisive."""
    base = [True, True] + [False] * 18
    cond = [True, True] + [True] * 12 + [False] * 6
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "identity", payload("identity", base))
        write(tmp, "temporal", payload("temporal", cond))
        stdout, reports = run(tmp)

    (rep,) = reports
    assert rep["conditioner"] == "temporal"
    assert rep["delta"] == 0.6
    assert rep["contingency"] == {"n": 20, "both": 2, "system_only": 12,
                                  "control_only": 0, "neither": 6}
    assert rep["significant"] and rep["p_value"] < 0.001
    assert rep["ci_lo"] > 0
    assert "*" in stdout


def test_a_conditioner_that_changes_nothing_is_not_significant():
    same = [True] * 5 + [False] * 15
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "identity", payload("identity", same))
        write(tmp, "temporal", payload("temporal", same))
        _, reports = run(tmp)

    (rep,) = reports
    assert rep["delta"] == 0.0
    assert not rep["significant"]
    assert rep["p_value"] == 1.0


def test_a_conditioner_that_breaks_questions_shows_a_negative_delta():
    """supersede:drop is predicted to lose. The script must be able to say so."""
    base = [True] * 10 + [False] * 10
    worse = [True] * 3 + [False] * 17
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "identity", payload("identity", base))
        write(tmp, "supersede-drop", payload("supersede:drop", worse))
        stdout, reports = run(tmp)

    (rep,) = reports
    assert rep["delta"] < 0
    assert rep["ci_hi"] < 0, "a real regression's CI should exclude zero"
    assert not rep["significant"], "significance is one-sided: a loss is not a win"
    # ...which is exactly why a regression needs its own marker, or it would be
    # reported identically to a null result.
    assert rep["significant_regression"] is True
    assert "!" in stdout


def test_token_delta_is_reported_against_the_baseline():
    """A conditioner that buys accuracy with tokens has not made anything free,
    so the cost has to appear next to every accuracy number."""
    ok = [True] * 10 + [False] * 10
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "identity", payload("identity", ok, tokens=2000.0))
        write(tmp, "temporal", payload("temporal", ok, tokens=2200.0))
        stdout, reports = run(tmp)

    assert abs(reports[0]["token_delta_pct"] - 0.10) < 1e-9
    assert "+10.0%" in stdout


def test_clamped_arms_are_skipped_rather_than_silently_compared():
    """memllm's session arms had every prompt at exactly 4098 tokens because the
    server truncated them. Comparing against that measures the truncation."""
    ok = [True] * 10 + [False] * 10
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "identity", payload("identity", ok))
        write(tmp, "clamped", payload("clamped", ok, prompt_tokens=4098))
        stdout, reports = run(tmp)

    assert reports == [], "a clamped arm must not produce a comparison"


def test_a_model_with_no_baseline_arm_is_skipped_not_crashed():
    """If the identity arm dies, the other arms for that model are unusable --
    but the arms for every other model must still be reported."""
    ok = [True] * 10 + [False] * 10
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "a", payload("temporal", ok, model="ollama:qwen2.5:1.5b-instruct"))
        write(tmp, "b", payload("identity", ok, model="ollama:qwen2.5:7b-instruct"))
        write(tmp, "c", payload("temporal", ok, model="ollama:qwen2.5:7b-instruct"))
        _, reports = run(tmp)

    assert [r["model"] for r in reports] == ["ollama:qwen2.5:7b-instruct"]


def test_per_type_deltas_are_broken_out():
    """The whole thesis is type-specific: supersession should move
    knowledge-update and temporal grounding should move temporal-reasoning."""
    base = [False] * 20
    # even indices are knowledge-update, odd are temporal-reasoning
    cond = [i % 2 == 0 for i in range(20)]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "identity", payload("identity", base))
        write(tmp, "supersede-mark", payload("supersede:mark", cond))
        stdout, reports = run(tmp)

    per = reports[0]["per_type"]
    assert per["knowledge-update"]["lift"] == 1.0
    assert per["temporal-reasoning"]["lift"] == 0.0
    assert "knowledge-update" in stdout


def test_no_arms_at_all_exits_with_a_message_not_a_traceback():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [PY, str(REPO / "scripts/compare_conditioners.py"), "--results", d,
             "--out", str(Path(d) / "x.json")],
            capture_output=True, text=True)
    assert r.returncode != 0
    assert "no cond_*.json arms" in r.stderr
    assert "Traceback" not in r.stderr


def test_load_groups_arms_by_model():
    ok = [True] * 6 + [False] * 4
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write(tmp, "a", payload("identity", ok, model="m1"))
        write(tmp, "b", payload("temporal", ok, model="m1"))
        write(tmp, "c", payload("identity", ok, model="m2"))
        grouped = cc.load(tmp)
    assert sorted(grouped) == ["m1", "m2"]
    assert len(grouped["m1"]) == 2 and len(grouped["m2"]) == 1
