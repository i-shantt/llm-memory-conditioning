"""Conditioner tests. Cases are built so the right answer is known by hand.

Follows memllm's convention: test names are assertions in prose, and nothing
here needs a model, a network, or the downloaded benchmark.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lastmile._deps  # noqa: E402,F401
from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import MemoryUnit  # noqa: E402

from lastmile.conditioner import (  # noqa: E402
    ComposeConditioner, Conditioning, IdentityConditioner, SupersedeConditioner,
    TemporalConditioner, build, condition, render_context,
)
from lastmile.conditioner.temporal import describe_delta, months_between  # noqa: E402

QDATE = "2023/06/01 (Thu) 09:00"


def unit(uid: int, text: str, day: str, *, evidence: bool = False,
         session: int = 0) -> MemoryUnit:
    return MemoryUnit(unit_id=uid, text=text, session_id=f"s{session}",
                      session_date=day, session_index=session,
                      is_evidence=evidence, roles=("user",))


GYM = [
    unit(0, "I go to the gym at 7:00 pm on weekdays.", "2023/01/10 (Tue) 18:00",
         evidence=True, session=0),
    unit(1, "Here are some tips for staying motivated.", "2023/02/02 (Thu) 10:00",
         session=1),
    unit(2, "I've moved my gym time to 6:00 pm.", "2023/05/03 (Wed) 20:00",
         evidence=True, session=2),
]
GYM_Q = "What time do I usually go to the gym?"


# --- the baseline must not drift -------------------------------------------

def test_identity_reproduces_the_memllm_rendering_byte_for_byte():
    """This is the control arm. If it renders differently from memllm's own
    harness, every measured effect is partly the renderer."""
    led = CostLedger()
    got = render_context(condition(IdentityConditioner(), GYM, GYM_Q, QDATE, led))
    expected = "\n\n".join(f"[{u.session_date}] {u.text}" for u in GYM)
    assert got == expected


def test_identity_annotates_nothing():
    c = IdentityConditioner().analyse(GYM, GYM_Q, QDATE)
    assert c.notes == {} and c.order is None and not c.dropped


# --- invariants that hold for every conditioner ----------------------------

ALL_NAMES = ["identity", "supersede:mark", "supersede:order", "supersede:drop",
             "temporal", "all", "safe"]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_conditioner_ever_calls_a_model(name):
    """The entire economic argument. A conditioner that needed an LLM call
    would cost what write-time consolidation costs."""
    led = CostLedger()
    condition(build(name), GYM, GYM_Q, QDATE, led)
    assert led.read.llm_calls == 0 and led.write.llm_calls == 0
    assert led.read.embed_calls == 0 and led.write.embed_calls == 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_conditioning_is_deterministic(name):
    """Set and dict iteration order inside slot building must not leak into
    the output, or two identical arms would disagree."""
    a = render_context(condition(build(name), GYM, GYM_Q, QDATE, CostLedger()))
    b = render_context(condition(build(name), GYM, GYM_Q, QDATE, CostLedger()))
    assert a == b


@pytest.mark.parametrize("name", [n for n in ALL_NAMES if "drop" not in n])
def test_only_drop_mode_removes_units(name):
    """`mark` and `order` are additive by construction. If one silently
    dropped a unit, its accuracy delta would partly be a recall change."""
    out = condition(build(name), GYM, GYM_Q, QDATE, CostLedger())
    assert {r.unit_id for r in out} == {u.unit_id for u in GYM}


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unit_text_is_never_altered(name):
    out = condition(build(name), GYM, GYM_Q, QDATE, CostLedger())
    by_id = {u.unit_id: u.text for u in GYM}
    assert all(r.text == by_id[r.unit_id] for r in out)


# --- supersede -------------------------------------------------------------

def test_supersede_marks_the_later_gym_time_current_and_the_earlier_outdated():
    out = {r.unit_id: r.prefix
           for r in condition(build("supersede:mark"), GYM, GYM_Q, QDATE, CostLedger())}
    assert "OUTDATED" in out[0], "the 7pm mention is the superseded one"
    assert "LATEST" in out[2], "the 6pm mention is current"
    assert "OUTDATED" not in out[1] and "LATEST" not in out[1], \
        "the motivation-tips turn asserts no gym time and is not in contention"


def test_supersede_says_nothing_when_every_mention_is_same_day():
    """Same-day mentions have no ordering to express, so claiming one is
    outdated would be an invention."""
    same = [unit(0, "I go to the gym at 7:00 pm.", "2023/01/10 (Tue) 18:00"),
            unit(1, "Actually the gym closes at 9:00 pm.", "2023/01/10 (Tue) 19:00")]
    assert SupersedeConditioner("mark").analyse(same, GYM_Q, QDATE).notes == {}


def test_supersede_says_nothing_when_the_mentions_agree():
    """No conflict, nothing to supersede -- otherwise the rule fires on every
    repeated mention of a topic."""
    agree = [unit(0, "I go to the gym at 6:00 pm.", "2023/01/10 (Tue) 18:00"),
             unit(1, "Reminder: gym at 6:00 pm.", "2023/05/03 (Wed) 20:00")]
    assert SupersedeConditioner("mark").analyse(agree, GYM_Q, QDATE).notes == {}


def test_naive_supersede_fires_on_agreement_which_is_why_it_is_at_chance():
    """The variant the CPU gate measured at chance. Kept so the negative
    result stays reproducible rather than becoming a claim in a README."""
    agree = [unit(0, "I go to the gym at 6:00 pm.", "2023/01/10 (Tue) 18:00"),
             unit(1, "Reminder: gym at 6:00 pm.", "2023/05/03 (Wed) 20:00")]
    naive = SupersedeConditioner("mark", require_conflict=False)
    assert naive.analyse(agree, GYM_Q, QDATE).notes != {}


def test_supersede_order_mode_never_claims_currency():
    """The salvage variant: state the ordering, assert nothing about which
    fact holds. It cannot mislead, so it needs no precision to be safe."""
    out = condition(build("supersede:order"), GYM, GYM_Q, QDATE, CostLedger())
    joined = " ".join(r.prefix for r in out)
    assert "OUTDATED" not in joined and "LATEST" not in joined
    assert "mention 1 of 2" in joined and "most recent" in joined


def test_supersede_drop_removes_the_superseded_mention_only():
    out = condition(build("supersede:drop"), GYM, GYM_Q, QDATE, CostLedger())
    assert {r.unit_id for r in out} == {1, 2}, "the 7pm mention is dropped"


def test_supersede_drop_can_delete_evidence_which_is_the_case_against_it():
    """LongMemEval asks past-directed questions, and `drop` answers them by
    deleting the answer. The CPU gate measured evidence survival at 0.800 on
    knowledge-update. This is the argument against write-time deletion."""
    out = condition(build("supersede:drop"), GYM, GYM_Q, QDATE, CostLedger())
    kept = {r.unit_id for r in out}
    assert any(u.is_evidence and u.unit_id not in kept for u in GYM)


def test_unclassifiable_questions_produce_no_supersession():
    """No answer-type pattern means no way to establish conflict, so the rule
    must stay silent rather than guess."""
    c = SupersedeConditioner("mark").analyse(GYM, "Tell me about my gym.", QDATE)
    assert c.notes == {}


# --- temporal --------------------------------------------------------------

def test_months_between_counts_calendar_months_not_thirty_day_blocks():
    assert months_between(date(2023, 1, 31), date(2023, 3, 1)) == 1
    assert months_between(date(2023, 1, 1), date(2023, 3, 1)) == 2
    assert months_between(date(2022, 12, 15), date(2023, 1, 14)) == 0


def test_describe_delta_gives_every_unit_the_benchmark_asks_in():
    """LongMemEval asks "how many days", "weeks" and "months" of the same
    corpus; leaving the conversion to the model is the arithmetic it gets
    wrong."""
    s = describe_delta(date(2023, 1, 10), date(2023, 6, 1))
    assert "142 days" in s and "20 weeks" in s and "4 months" in s
    assert describe_delta(date(2023, 6, 1), date(2023, 6, 1)) == "same day as today"
    assert "1 day" in describe_delta(date(2023, 5, 31), date(2023, 6, 1))


def test_temporal_orders_events_oldest_first_and_numbers_them():
    out = condition(build("temporal"), list(reversed(GYM)), GYM_Q, QDATE, CostLedger())
    assert [r.unit_id for r in out] == [0, 1, 2]
    assert "event 1 of 3" in out[0].prefix and "event 3 of 3" in out[2].prefix


def test_temporal_without_sort_keeps_retrieval_rank():
    """Rank order puts the most relevant unit first, and position is known to
    matter, so re-ordering has to be a measurable choice rather than a default
    nobody checked."""
    ranked = list(reversed(GYM))
    out = condition(TemporalConditioner(sort=False), ranked, GYM_Q, QDATE, CostLedger())
    assert [r.unit_id for r in out] == [u.unit_id for u in ranked]


def test_temporal_keeps_units_whose_date_will_not_parse():
    """Dropping them would change what the model sees for a reason unrelated
    to the thing being tested."""
    broken = [unit(0, "no date here", "not a date"), *GYM]
    out = condition(build("temporal"), broken, GYM_Q, QDATE, CostLedger())
    assert 0 in {r.unit_id for r in out}
    assert out[-1].unit_id == 0, "unparseable dates sort last"


# --- composition -----------------------------------------------------------

def test_compose_accumulates_notes_from_every_conditioner():
    out = condition(build("all"), GYM, GYM_Q, QDATE, CostLedger())
    joined = " ".join(r.prefix for r in out)
    assert "event " in joined, "temporal's note is present"
    assert "OUTDATED" in joined, "supersede's note is present"


def test_compose_gives_the_last_conditioner_the_final_say_on_order():
    out = condition(build("all"), list(reversed(GYM)), GYM_Q, QDATE, CostLedger())
    assert [r.unit_id for r in out] == [0, 1, 2], "temporal sorts, supersede does not"


def test_merge_keeps_an_earlier_order_when_the_later_has_no_opinion():
    a = Conditioning(order=(2, 1, 0))
    b = Conditioning(notes={0: ["x"]})
    assert a.merge(b).order == (2, 1, 0)


def test_compose_requires_at_least_one_conditioner():
    with pytest.raises(ValueError):
        ComposeConditioner()


def test_build_rejects_an_unknown_name_and_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown conditioner"):
        build("nope")


def test_prefix_always_leads_with_the_date():
    """Every arm must carry the date in the same position, or a conditioned
    arm and the baseline differ by more than the annotation."""
    for name in ALL_NAMES:
        for r in condition(build(name), GYM, GYM_Q, QDATE, CostLedger()):
            assert r.prefix.startswith("20"), f"{name}: {r.prefix!r}"
