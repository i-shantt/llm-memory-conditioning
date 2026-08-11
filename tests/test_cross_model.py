"""Tests for the cross-model claim, which is the most fragile one in the repo.

Saying a conditioned 1.5B reaches a 7B's accuracy invites exactly the error
memllm warns about: the containment grader rewards verbosity, and the two models
differ in it. These pin the two guards that make the claim sayable -- a paired
test rather than two point estimates, and a re-grade under answer-length caps.

Everything runs from the stored predictions, so no model and no benchmark.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "cross_model_check", REPO / "scripts" / "cross_model_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cmc = _load_script()

TREATMENT = RESULTS / "cond_1.5b_all_k10_n100.json"
BASE_7B = RESULTS / "cond_7b_identity_k10_n100.json"
BASE_1_5B = RESULTS / "cond_1.5b_identity_k10_n100.json"


def test_mcnemar_reproduces_the_published_within_model_lift():
    """The cross-model test is only trustworthy if the same code reproduces the
    headline number, which was computed by memllm's ablation module."""
    n, treat, base, _, _, p = cmc.mcnemar_exact(
        cmc.graded(TREATMENT), cmc.graded(BASE_1_5B))
    assert n == 91
    assert treat - base == pytest.approx(0.1319, abs=5e-4)
    assert p == pytest.approx(0.002, abs=5e-4)


def test_conditioned_1_5b_is_not_separable_from_the_unconditioned_7b():
    """The claim is 'a test could not tell them apart', not 'it won'. If a
    re-grade ever separates them, the README sentence has to change."""
    _, treat, base, _, _, p = cmc.mcnemar_exact(
        cmc.graded(TREATMENT), cmc.graded(BASE_7B))
    assert p > 0.05, f"7B is now separable (p={p}); the README claim is stale"
    assert abs(treat - base) <= 2 / 91, "the gap grew past two questions"


def test_the_gap_to_7b_stays_within_two_questions_at_every_length_cap():
    """A lead that decays as answers are truncated was verbosity. This gap has
    no lead to lose in either direction, at any cap -- which is the point."""
    treat, base = cmc.graded(TREATMENT), cmc.graded(BASE_7B)
    for cap in cmc.CAPS:
        gap = cmc.capped_accuracy(treat, cap) - cmc.capped_accuracy(base, cap)
        assert abs(gap) <= 2 / 91 + 1e-9, f"cap={cap} opened a gap of {gap:+.3f}"


def test_the_lead_over_3b_shrinks_under_truncation_and_is_reported_that_way():
    """Against 3B the conditioned 1.5B does lead, but roughly three quarters of
    that lead is length. The README has to say so rather than quote the full
    number alone."""
    treat, base = cmc.graded(TREATMENT), cmc.graded(RESULTS / "cond_3b_identity_k10_n100.json")
    full = cmc.capped_accuracy(treat, None) - cmc.capped_accuracy(base, None)
    tight = cmc.capped_accuracy(treat, 8) - cmc.capped_accuracy(base, 8)
    assert full > 0.09, "the uncapped lead is smaller than reported"
    assert 0 < tight < full / 2, "the decay pattern changed; re-read the README"
