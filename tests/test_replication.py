"""Tests pinning the replication failure, which is now the repo's main claim.

The README says a +13.2 point result at n=91 became +0.7 at n=446, that the
original measurement still reproduces on the questions it was measured on, and
that the discrepancy is the control arm's draw rather than the treatment's.
Those are four separate assertions and each one can rot independently -- a
regrade in memllm moves all of them at once.

Runs from the stored predictions. No model, no benchmark download.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
FULL_CONTROL = RESULTS / "cond_qwen2.5-1.5b-instruct_identity_k10_n500.json"
FULL_SYSTEM = RESULTS / "cond_qwen2.5-1.5b-instruct_all_k10_n500.json"
SAMPLED_CONTROL = RESULTS / "cond_1.5b_identity_k10_n100.json"
SAMPLED_SYSTEM = RESULTS / "cond_1.5b_all_k10_n100.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "replication_check", REPO / "scripts" / "replication_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_script()
CONTROL, SYSTEM = rc.graded(FULL_CONTROL), rc.graded(FULL_SYSTEM)
SAMPLED = {r["question_id"] for r in json.loads(SAMPLED_CONTROL.read_text())["records"]}
SEEN = [i for i in sorted(CONTROL) if i in SAMPLED]
UNSEEN = [i for i in sorted(CONTROL) if i not in SAMPLED]


def test_the_full_benchmark_result_is_a_null():
    """The headline. If this ever becomes significant the README is wrong."""
    acc_c, acc_s, _, _, p = rc.mcnemar_exact(sorted(CONTROL), CONTROL, SYSTEM)
    assert len(CONTROL) == 446
    assert acc_s - acc_c == pytest.approx(0.0067, abs=5e-4)
    assert p > 0.05, "the full-benchmark comparison is no longer a null"


def test_the_original_sample_still_shows_the_effect_it_showed():
    """The replication failure is only interesting because the first measurement
    was correct. If these 91 questions stopped reproducing, the story would be a
    bug rather than a sampling problem, and the README says the opposite."""
    acc_c, acc_s, _, _, p = rc.mcnemar_exact(SEEN, CONTROL, SYSTEM)
    assert len(SEEN) == 91
    assert acc_s - acc_c > 0.10, "the sampled questions no longer show the effect"
    assert p < 0.05, "the original result is no longer significant on its own sample"


def test_the_gap_is_the_baselines_draw_not_the_treatments():
    """The claim is specific: identity was unlucky on the sampled questions,
    while the conditioned arm scores about the same either way. A README that
    said 'the sample was unrepresentative' without saying which arm moved would
    be hiding the interesting half."""
    base_seen = sum(CONTROL[i]["deterministic"] for i in SEEN) / len(SEEN)
    base_unseen = sum(CONTROL[i]["deterministic"] for i in UNSEEN) / len(UNSEEN)
    sys_seen = sum(SYSTEM[i]["deterministic"] for i in SEEN) / len(SEEN)
    sys_unseen = sum(SYSTEM[i]["deterministic"] for i in UNSEEN) / len(UNSEEN)

    assert base_unseen - base_seen > 0.10, "the control no longer moves between sets"
    assert abs(sys_unseen - sys_seen) < 0.05, "the treatment now moves too"
    assert (base_unseen - base_seen) > 4 * abs(sys_unseen - sys_seen)


def test_the_mechanism_predicts_the_right_signs_at_full_scale():
    """Not significance -- direction. The sort should win the types where recency
    is the answer and lose the ones where search rank already was."""
    helped = [i for i in CONTROL if CONTROL[i]["question_type"] in rc.HELPED]
    hurt = [i for i in CONTROL if CONTROL[i]["question_type"] not in rc.HELPED]
    acc_c_h, acc_s_h, *_ = rc.mcnemar_exact(helped, CONTROL, SYSTEM)
    acc_c_x, acc_s_x, *_ = rc.mcnemar_exact(hurt, CONTROL, SYSTEM)
    assert acc_s_h > acc_c_h, "knowledge-update + temporal-reasoning no longer gain"
    assert acc_s_x < acc_c_x, "the remaining types no longer lose"


def test_the_oracle_router_ceiling_is_small_and_not_significant():
    """The README uses this to argue against building a router. If routing ever
    became worth a lot, that conclusion would have to change."""
    ids = sorted(CONTROL)
    routed = {i: (SYSTEM if CONTROL[i]["question_type"] in rc.HELPED else CONTROL)[i]
              for i in ids}
    acc_c, acc_r, _, _, p = rc.mcnemar_exact(ids, CONTROL, routed)
    assert 0 < acc_r - acc_c < 0.05, "the oracle ceiling moved out of the reported range"
    assert p > 0.05, "the oracle router is now significant; the README argues it is not"


def test_the_two_runs_agree_on_the_questions_they_share():
    """Different sample sizes must not change a question's prompt: retrieval is
    per-question. Greedy decoding should then give the same answer. It mostly
    does -- the residual is serving-level nondeterminism, and the README quotes
    a number for it that should not silently drift."""
    sampled_sys = {r["question_id"]: r
                   for r in json.loads(SAMPLED_SYSTEM.read_text())["records"]}
    shared = set(sampled_sys) & set(SYSTEM)
    identical = sum(1 for i in shared if sampled_sys[i]["pred"] == SYSTEM[i]["pred"])
    assert identical / len(shared) > 0.85, (
        f"only {identical}/{len(shared)} predictions match across runs -- that is "
        "more than serving nondeterminism and suggests the configs differ")
