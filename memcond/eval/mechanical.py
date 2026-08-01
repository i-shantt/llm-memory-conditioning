"""Score a conditioner without a model, using LongMemEval's own evidence labels.

Answer accuracy needs a GPU. Whether a conditioner's *decisions* are sane does
not: LongMemEval labels every turn with `has_answer`, which `MemoryUnit`
exposes as `is_evidence`. So the question "does this rule mark the right unit as
current?" is answerable on a laptop, in seconds, for free.

This exists to be a gate. A conditioner that tells the model the current answer
is stale is not a neutral no-op, and buying GPU time to confirm that would be a
waste.

## Getting the harm metric right

The first version of this file counted "evidence units marked OUTDATED" as harm
and reported 0.681, which looked disqualifying. It was measuring the wrong
thing. On `knowledge-update`, 67 of 78 LongMemEval questions carry **two**
evidence turns on two different dates -- the superseded fact and the one that
replaced it -- because answering needs both. Marking the earlier one OUTDATED is
the correct behaviour, and the metric was penalising it.

What is actually harmful is mislabelling the *newest* evidence unit, since that
is the one holding the current answer. Hence:

    harm      = newest evidence unit marked OUTDATED        (want ~0)
    precision = newest evidence unit marked LATEST          (want high)

Both are computed only over questions where the rule fired at all, so a rule
that stays silent scores neither well nor badly -- it scores `--`, and its
firing rate is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import memcond._deps  # noqa: F401
from memllm.cost import CostLedger, count_tokens  # noqa: E402
from memllm.data.loader import MemoryUnit  # noqa: E402

from memcond.conditioner import (  # noqa: E402
    LATEST, OUTDATED, Conditioner, apply, render_context,
)


@dataclass
class GateTally:
    """Counts accumulated over questions; rates are derived, never stored."""

    n_questions: int = 0
    n_units: int = 0
    n_evidence: int = 0
    n_evidence_dropped: int = 0

    # Questions where the conditioner annotated the newest evidence unit.
    n_newest_ev_annotated: int = 0
    n_newest_ev_latest: int = 0        # correct
    n_newest_ev_outdated: int = 0      # the harm

    # Older evidence units, which SHOULD be marked outdated on knowledge-update.
    n_older_ev_annotated: int = 0
    n_older_ev_outdated: int = 0

    n_questions_fired: int = 0
    n_marked_latest: int = 0
    n_marked_outdated: int = 0

    baseline_tokens: int = 0
    conditioned_tokens: int = 0
    llm_calls: int = 0

    by_type: dict[str, "GateTally"] = field(default_factory=dict)

    def rates(self) -> dict:
        def pct(num: int, den: int) -> float | None:
            return num / den if den else None
        return {
            "n_questions": self.n_questions,
            "n_units": self.n_units,
            "n_evidence": self.n_evidence,
            "fired": pct(self.n_questions_fired, self.n_questions),
            # THE harm metric.
            "newest_evidence_outdated": pct(
                self.n_newest_ev_outdated, self.n_newest_ev_annotated),
            "n_newest_evidence_outdated": self.n_newest_ev_outdated,
            # THE precision metric.
            "newest_evidence_latest": pct(
                self.n_newest_ev_latest, self.n_newest_ev_annotated),
            "n_newest_evidence_annotated": self.n_newest_ev_annotated,
            # Correctly demoting a genuinely superseded fact.
            "older_evidence_outdated": pct(
                self.n_older_ev_outdated, self.n_older_ev_annotated),
            "evidence_survival": pct(
                self.n_evidence - self.n_evidence_dropped, self.n_evidence),
            "units_annotated": pct(
                self.n_marked_latest + self.n_marked_outdated, self.n_units),
            "token_delta_pct": pct(
                self.conditioned_tokens - self.baseline_tokens,
                self.baseline_tokens),
            "tokens_per_query": pct(self.conditioned_tokens, self.n_questions),
            "llm_calls": self.llm_calls,
        }


def _note_kind(notes: list[str]) -> str | None:
    for n in notes:
        if n.startswith(LATEST):
            return LATEST
        if n.startswith(OUTDATED):
            return OUTDATED
    return None


def score_question(cond: Conditioner, units: list[MemoryUnit], query: str,
                   question_date: str, tally: GateTally,
                   ledger: CostLedger) -> None:
    """Accumulate one question's decisions into `tally`."""
    with ledger.timer("read"):
        c = cond.analyse(units, query, question_date)

    tally.n_questions += 1
    tally.n_units += len(units)

    evidence = [u for u in units if u.is_evidence]
    tally.n_evidence += len(evidence)
    tally.n_evidence_dropped += sum(1 for u in evidence if u.unit_id in c.dropped)

    kinds = {uid: _note_kind(n) for uid, n in c.notes.items()}
    fired = [uid for uid, k in kinds.items() if k is not None]
    if fired:
        tally.n_questions_fired += 1
    tally.n_marked_latest += sum(1 for k in kinds.values() if k == LATEST)
    tally.n_marked_outdated += sum(1 for k in kinds.values() if k == OUTDATED)

    if evidence:
        newest = max(evidence, key=lambda u: (u.date_key, u.session_index, u.unit_id))
        k = kinds.get(newest.unit_id)
        if k is not None:
            tally.n_newest_ev_annotated += 1
            tally.n_newest_ev_latest += int(k == LATEST)
            tally.n_newest_ev_outdated += int(k == OUTDATED)
        for u in evidence:
            if u.unit_id == newest.unit_id:
                continue
            ko = kinds.get(u.unit_id)
            if ko is not None:
                tally.n_older_ev_annotated += 1
                tally.n_older_ev_outdated += int(ko == OUTDATED)

    tally.baseline_tokens += count_tokens(
        "\n\n".join(f"[{u.session_date}] {u.text}" for u in units))
    tally.conditioned_tokens += count_tokens(render_context(apply(units, c)))
    tally.llm_calls = ledger.read.llm_calls + ledger.write.llm_calls
