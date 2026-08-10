"""Do the date arithmetic for the model, deterministically, before it reads.

The failures this targets are all cases where the evidence was retrieved and the
model still could not compute over it. From memllm's stored `oracle` arm --
where by construction every evidence turn is in the prompt:

    1.5B  "How many months have passed since I last visited a museum with a
           friend?"
          gold 5, predicted "One month" -- one month back is a February visit
          with a parent; the visit with a friend is five months back
    1.5B  "Which event happened first, the purchase of the coffee maker or the
           malfunction of the stand mixer?"
          -- both dates present, ordered wrong
    14B   "How many weeks passed between the time I sold homemade baked goods
           at the Farmers' Market for the last time and the time I
           participated in the Spring Fling Market?"
          gold 3, predicted "The last time you sold homemade baked goods was on
          2023/02/26, and you participated in the Spring Fling Market on
          2023/03/21. There are 4 weeks between these two dates."

Two distinct failures, and the transform addresses both. The first two are
anchoring: ten dated units arrive in retrieval-rank order with nothing to
distinguish them, and the model computes from the wrong one. Sorting them and
labelling each with its distance from today is exactly the missing information.
The third is arithmetic proper -- both correct dates quoted off the page, and
the subtraction over them still wrong, on the largest model memllm ran.

Subtraction is free and exact in Python. Doing it at render time turns "compare
two dates" into "read a number", which is the operation small models are good
at. Costs roughly a dozen tokens per unit, which the ledger reports honestly. It
also stops the model spending output tokens deriving the number in prose: at
memllm's default `max_new_tokens=64` those derivations were frequently cut off
mid-sentence, which is why the arms here run at 256.

The size gradient is the reason to expect this to matter most on small models.
memllm's `temporal-reasoning` lift over its best control is +0.040 at 1.5B and
+0.080 at 3B, against +0.480 at 14B, from identical retrieval -- the small
models hold the dates and cannot use them.
"""

from __future__ import annotations

from datetime import date

import memcond._deps  # noqa: F401
from memllm.data.loader import MemoryUnit, parse_date  # noqa: E402

from .base import Conditioning


def _as_date(raw: str) -> date | None:
    """LongMemEval dates are '2023/04/10 (Mon) 17:50'. Returns None if unparseable."""
    try:
        y, m, d = parse_date(raw)
        return date(y, m, d)
    except Exception:
        return None


def months_between(earlier: date, later: date) -> int:
    """Whole calendar months, the way a person counts them.

    (later - earlier).days / 30 is not the same question: LongMemEval's gold
    answers are calendar counts, so "Jan 31 to Mar 1" is 1 month, not 1.03.
    """
    months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    if later.day < earlier.day:
        months -= 1
    return months


def describe_delta(then: date, now: date) -> str:
    """Every unit the benchmark asks in, so no unit conversion is left to do.

    LongMemEval asks "how many days", "how many weeks" and "how many months"
    about the same corpus, and a model that has to convert between them is doing
    arithmetic it frequently gets wrong. Giving all three costs a few tokens and
    removes the conversion entirely.
    """
    days = (now - then).days
    if days == 0:
        return "same day as today"
    if days < 0:
        return f"{-days} days AFTER today"
    parts = [f"{days} day{'s' if days != 1 else ''}"]
    if days >= 7:
        weeks = days // 7
        parts.append(f"{weeks} week{'s' if weeks != 1 else ''}")
    months = months_between(then, now)
    if months >= 1:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    return " / ".join(parts) + " before today"


class TemporalConditioner:
    """Order retrieved units in time and label each with its distance from today.

    `sort=True` (the default) presents units oldest-first rather than in
    retrieval-rank order. For "which happened first" that converts a comparison
    into reading the top line. It is also the one thing here that could plausibly
    hurt -- rank order puts the most relevant unit first, and position is known
    to matter -- so it is a flag, and both settings get measured.

    Units whose date will not parse keep their raw date string and are sorted
    last. Silently dropping them would change what the model sees for reasons
    unrelated to the thing being tested.
    """

    def __init__(self, sort: bool = True) -> None:
        self.sort = sort
        self.name = "temporal" if sort else "temporal:norank"

    def analyse(self, units: list[MemoryUnit], query: str,
                question_date: str) -> Conditioning:
        today = _as_date(question_date)
        dated = [(u, _as_date(u.session_date)) for u in units]

        if self.sort:
            # None sorts last; session_index/unit_id keep ties stable.
            dated.sort(key=lambda p: (p[1] is None,
                                      p[1] or date.min,
                                      p[0].session_index, p[0].unit_id))

        n = len(dated)
        notes: dict[int, list[str]] = {}
        for i, (u, when) in enumerate(dated, start=1):
            bits = [f"event {i} of {n}"]
            if when is not None and today is not None:
                bits.append(describe_delta(when, today))
            notes[u.unit_id] = bits

        order = tuple(u.unit_id for u, _ in dated) if self.sort else None
        return Conditioning(notes=notes, order=order)
