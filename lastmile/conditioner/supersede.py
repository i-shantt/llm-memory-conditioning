"""Mark stale facts as stale, at read time, with no model call.

The failure this targets, from memllm's stored predictions (7B, hybrid, k=10):

    Q     What time do I usually go to the gym?
    gold  6:00 pm
    pred  You usually go to the gym at 7:00 pm on Mondays, Wednesdays and Fridays.

    Q     What brand of BBQ sauce am I currently obsessed with?
    gold  Kansas City Masterpiece
    pred  You are currently obsessed with Sweet Baby Ray's BBQ sauce.

Both answers are quoted correctly from the retrieved context. They quote the
*superseded* mention. Retrieval did not fail -- on LongMemEval's
`knowledge-update` slice hybrid retrieval scores `any_hit@10 = 1.000`, and the
7B model still answers 9 of 16 wrong. The `oracle` arm, which differs only by
dropping the non-evidence units, scores 0.688 on the same questions.

The literature calls this the current-state adjudication gap (STALE,
arXiv 2605.06527). Its remedy, CUPMem, runs an LLM adjudicator on every write.
This module tests whether a rule gets a useful share of that for free.

What it can and cannot know: it detects that several retrieved units discuss the
same query term on different dates, and orders them. It does *not* detect
contradiction, which needs semantics. So a slot of three mentions where nothing
changed will still be labelled -- the annotation says "a later mention exists",
which is true, rather than "this was contradicted", which would not be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import lastmile._deps  # noqa: F401
from memllm.data.loader import MemoryUnit  # noqa: E402
from memllm.eval.grade import normalize_tokens  # noqa: E402

from .base import Conditioning

# Note kinds, so the CPU gate can score decisions without parsing prompt text.
LATEST = "LATEST"
OUTDATED = "OUTDATED"

# Question scaffolding carries no topic, so it must not become a slot key.
# Kept small and explicit rather than pulled from a stopword package: the list
# is auditable, and a wrong entry here silently changes clustering.
_STOP = {
    "what", "when", "where", "who", "whom", "which", "how", "why", "did", "do",
    "does", "was", "were", "is", "are", "am", "be", "been", "have", "has", "had",
    "i", "me", "my", "mine", "you", "your", "we", "us", "it", "its", "they",
    "them", "their", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "about", "as", "that", "this", "these",
    "those", "there", "here", "not", "no", "so", "than", "then", "too", "very",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "get", "got", "go", "going", "went", "make", "made", "take", "took", "use",
    "used", "usually", "currently", "now", "still", "last", "first", "recent",
    "recently", "ever", "any", "some", "all", "most", "more", "much", "many",
    "one", "s", "t",
}

_MIN_TERM_LEN = 3

# What kind of value the question is asking for, and how to find one in a unit.
# The first pattern whose trigger matches the question wins, so the order is
# significant: "what time" must beat the bare "what".
_ANSWER_TYPES: list[tuple[re.Pattern, re.Pattern]] = [
    (re.compile(r"\bwhat time\b|\bwhat hour\b", re.I),
     re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:[ap]\.?m\.?)?|\b\d{1,2}\s*[ap]\.?m\.?", re.I)),
    (re.compile(r"\bhow much\b.*\b(cost|pay|paid|price|spend|spent|save)\b"
                r"|\bwhat (?:was |is )?the (?:amount|price|cost|total)\b", re.I),
     re.compile(r"\$\s?[\d,]+(?:\.\d{2})?|\b[\d,]+\s*dollars?\b", re.I)),
    (re.compile(r"\bhow (?:many|much|long|often|far)\b", re.I),
     re.compile(r"\b\d[\d,]*(?:\.\d+)?\b", re.I)),
    (re.compile(r"\bwhat (?:day|date)\b|\bwhen\b", re.I),
     re.compile(r"\b\d{4}/\d{1,2}/\d{1,2}\b|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
                r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
                re.I)),
]


def answer_pattern(question: str) -> re.Pattern | None:
    """The value pattern this question is asking for, or None if unclassified."""
    for trigger, value in _ANSWER_TYPES:
        if trigger.search(question):
            return value
    return None


def query_terms(query: str) -> list[str]:
    """Content words of the query, deduplicated, in deterministic order."""
    seen: dict[str, None] = {}
    for tok in normalize_tokens(query):
        if len(tok) >= _MIN_TERM_LEN and tok not in _STOP:
            seen.setdefault(tok, None)
    return list(seen)


@dataclass(frozen=True)
class Slot:
    """Retrieved units that discuss one query term, oldest first."""

    term: str
    unit_ids: tuple[int, ...]

    @property
    def is_supersedable(self) -> bool:
        return len(self.unit_ids) >= 2


def build_slots(units: list[MemoryUnit], query: str,
                require_conflict: bool = True) -> dict[int, Slot]:
    """Map unit_id -> the Slot it belongs to. Units with no query term are absent.

    A unit's slot key is the query term it contains that is *most shared* across
    the retrieved units. The first version of this used the rarest term instead,
    on the theory that a specific word like "gym" identifies the units in
    contention better than a generic one like "time". It does the opposite: for
    "What time do I usually go to the gym?", "I go to the gym at 7pm" keys on
    "gym" while "I've moved my gym time to 6pm" keys on "time" (rarer), so the
    two mentions that actually conflict land in different slots and neither is
    annotated. Grouping needs the *shared* term, not the distinctive one.

    Over-grouping is the opposite risk, and `require_conflict` bounds it: units
    must also assert a value of the type the question asks for, which excludes
    the topic chatter that a common term would otherwise sweep in.

    Only slots whose members carry at least two distinct dates are returned. A
    slot where every mention is same-day has no ordering to express, and
    labelling one of them "outdated" would be an invention.

    `require_conflict` additionally demands that the units disagree about a
    value of the type the question asks for. Without it this rule is measurably
    at chance -- the CPU gate put `latest_is_evidence` at 0.417 and had it
    labelling 68% of annotated evidence "outdated" -- because recency within a
    topic is not currency of a fact. LongMemEval haystacks are full of later
    chatter about a topic that updates nothing, and the rule cannot tell that
    from a genuine correction unless it looks at the values being asserted.
    """
    terms = query_terms(query)
    if not terms:
        return {}

    unit_tokens = {u.unit_id: set(normalize_tokens(u.text)) for u in units}
    df = {t: sum(1 for u in units if t in unit_tokens[u.unit_id]) for t in terms}
    value_re = answer_pattern(query) if require_conflict else None
    if require_conflict and value_re is None:
        return {}   # cannot establish conflict, so assert nothing

    members: dict[str, list[MemoryUnit]] = {}
    for u in units:
        present = [t for t in terms if t in unit_tokens[u.unit_id]]
        if not present:
            continue
        if value_re is not None and not value_re.search(u.text):
            continue   # says nothing about the asked-for value; not in contention
        # (df, term) makes ties alphabetical, so clustering is reproducible.
        key = max(present, key=lambda t: (df[t], t))
        members.setdefault(key, []).append(u)

    slots: dict[int, Slot] = {}
    for term, group in members.items():
        if len(group) < 2 or len({u.date_key for u in group}) < 2:
            continue
        if value_re is not None:
            values = {frozenset(m.group(0).lower().strip()
                                for m in value_re.finditer(u.text)) for u in group}
            if len(values) < 2:
                continue   # everyone agrees; there is nothing to supersede
        ordered = sorted(group, key=lambda u: (u.date_key, u.session_index, u.unit_id))
        slot = Slot(term, tuple(u.unit_id for u in ordered))
        for u in ordered:
            slots[u.unit_id] = slot
    return slots


class SupersedeConditioner:
    """Annotate (or drop) retrieved units that a later message revisits.

    `mode="mark"` keeps every unit and adds a prefix. `mode="drop"` removes the
    outdated ones, which costs fewer tokens and is expected to be *worse*:
    LongMemEval asks past-directed questions too ("Before I purchased the gravel
    bike, ..."), and deleting the earlier fact destroys the answer. That is the
    concrete case against write-time deletion, which is what Mem0's UPDATE and
    DELETE operations do irreversibly -- so it is measured rather than asserted.
    """

    def __init__(self, mode: str = "mark", require_conflict: bool = True) -> None:
        if mode not in ("mark", "drop", "order"):
            raise ValueError(
                f"mode must be 'mark', 'drop' or 'order', got {mode!r}")
        self.mode = mode
        self.require_conflict = require_conflict
        self.name = f"supersede:{mode}" + ("" if require_conflict else ":naive")

    def analyse(self, units: list[MemoryUnit], query: str,
                question_date: str) -> Conditioning:
        slots = build_slots(units, query, self.require_conflict)
        by_id = {u.unit_id: u for u in units}

        notes: dict[int, list[str]] = {}
        dropped: set[int] = set()
        for uid, slot in slots.items():
            # Everything on the newest DATE is current, not just the last turn
            # of it. Ordering by turn index inside a shared date crowned an
            # assistant's follow-up over the user turn that stated the fact,
            # which the CPU gate saw as the newest evidence unit being marked
            # OUTDATED on 93% of knowledge-update questions. Supersession is a
            # claim about dates; same-day siblings supersede nothing.
            newest_day = max(by_id[i].date_key for i in slot.unit_ids)
            is_current = by_id[uid].date_key == newest_day
            latest_id = slot.unit_ids[-1]

            if self.mode == "order":
                # State the ordering and claim nothing about currency. The
                # gate showed the currency claim is at chance, but the ordering
                # itself is a fact, and this variant cannot mislead.
                i = slot.unit_ids.index(uid) + 1
                where = "most recent" if uid == latest_id else "earlier"
                notes[uid] = [f"mention {i} of {len(slot.unit_ids)} of "
                              f"'{slot.term}' ({where})"]
                continue

            if is_current:
                notes[uid] = [f"{LATEST} mention of '{slot.term}' "
                              f"({len(slot.unit_ids)} in total)"]
            elif self.mode == "mark":
                newer = by_id[latest_id].session_date
                notes[uid] = [f"{OUTDATED} mention of '{slot.term}' "
                              f"-- superseded by {newer}"]
            else:
                dropped.add(uid)
        return Conditioning(notes=notes, dropped=frozenset(dropped))
