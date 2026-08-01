"""The conditioner interface: how retrieved units become prompt text.

A retriever decides *which* units reach the model. A conditioner decides *how
they are presented*. That second step is normally a one-line f-string nobody
thinks about, and it is where a measurable amount of accuracy is lost.

Every conditioner here is deterministic and calls no model. That is the whole
economic argument: write-time consolidation pays LLM calls proportional to the
corpus whether or not a query ever arrives, while conditioning pays nothing and
touches only the k units a query actually retrieved.

A conditioner returns a `Conditioning` -- a description of what to do -- rather
than finished strings. Two reasons. Composition merges cleanly, because notes
concatenate instead of two conditioners each prepending their own date. And the
CPU gate can score the decisions directly against LongMemEval's `is_evidence`
labels, with no prompt-text parsing and no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import memcond._deps  # noqa: F401  (puts memllm on the path)
from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import MemoryUnit  # noqa: E402


@dataclass(frozen=True)
class Conditioning:
    """What a conditioner decided, before any of it becomes text."""

    notes: dict[int, list[str]] = field(default_factory=dict)
    order: tuple[int, ...] | None = None      # None: keep the incoming order
    dropped: frozenset[int] = frozenset()

    def merge(self, other: "Conditioning") -> "Conditioning":
        notes = {k: list(v) for k, v in self.notes.items()}
        for uid, extra in other.notes.items():
            notes.setdefault(uid, []).extend(extra)
        return Conditioning(
            notes=notes,
            # A later conditioner's ordering wins; None means it had no opinion.
            order=other.order if other.order is not None else self.order,
            dropped=self.dropped | other.dropped,
        )


@dataclass(frozen=True)
class RenderedUnit:
    """One unit as it will appear in the prompt."""

    unit_id: int
    text: str
    prefix: str = ""

    def render(self) -> str:
        return f"[{self.prefix}] {self.text}" if self.prefix else self.text


def apply(units: list[MemoryUnit], c: Conditioning) -> list[RenderedUnit]:
    """Turn a Conditioning into the units the prompt will contain.

    The date always leads the prefix, exactly as memllm's harness renders it,
    with any notes appended after it. That keeps the baseline and every
    conditioned arm on one rendering path, so a comparison cannot accidentally
    measure the join.
    """
    by_id = {u.unit_id: u for u in units}
    order = c.order if c.order is not None else tuple(u.unit_id for u in units)

    out: list[RenderedUnit] = []
    for uid in order:
        if uid in c.dropped or uid not in by_id:
            continue
        u = by_id[uid]
        prefix = " | ".join([u.session_date, *c.notes.get(uid, [])])
        out.append(RenderedUnit(uid, u.text, prefix))
    return out


def render_context(rendered: list[RenderedUnit]) -> str:
    """Join rendered units into the context block a prompt takes."""
    return "\n\n".join(r.render() for r in rendered)


@runtime_checkable
class Conditioner(Protocol):
    """Mirrors memllm's `Retriever` protocol so the two read alike."""

    name: str

    def analyse(self, units: list[MemoryUnit], query: str,
                question_date: str) -> Conditioning:
        ...


def condition(cond: Conditioner, units: list[MemoryUnit], query: str,
              question_date: str, ledger: CostLedger) -> list[RenderedUnit]:
    """Run a conditioner and bill it to the read phase.

    Billing matters more than it looks: the ledger is the evidence for the
    claim. It will show `llm_calls = 0` for every conditioner here, against the
    O(corpus) LLM calls a write-time consolidator spends.
    """
    with ledger.timer("read"):
        c = cond.analyse(units, query, question_date)
    return apply(units, c)


class IdentityConditioner:
    """The baseline: exactly what memllm's e2e harness does today.

    Reproduces `f"[{unit.session_date}] {unit.text}"` in retrieval-hit order,
    byte for byte. This is the control arm, so any drift from memllm's own
    rendering would show up as a spurious effect. `tests/` pins it.
    """

    name = "identity"

    def analyse(self, units: list[MemoryUnit], query: str,
                question_date: str) -> Conditioning:
        return Conditioning()


class ComposeConditioner:
    """Chain conditioners. Notes accumulate; the last opinion on order wins."""

    def __init__(self, *conditioners: Conditioner, name: str | None = None) -> None:
        if not conditioners:
            raise ValueError("compose needs at least one conditioner")
        self.conditioners = conditioners
        self.name = name or "+".join(c.name for c in conditioners)

    def analyse(self, units: list[MemoryUnit], query: str,
                question_date: str) -> Conditioning:
        out = Conditioning()
        for c in self.conditioners:
            out = out.merge(c.analyse(units, query, question_date))
        return out
