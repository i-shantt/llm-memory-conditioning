"""Would routing the chronological sort by question wording have worked?

The mechanism result says the sort wins `knowledge-update` and loses
`single-session-user`. The obvious follow-up is to sort only when the question
is asking about a current state -- capturing the gain without the loss.

This script tests that premise before anyone builds it, using only banked
per-question results. Every arm stores its predictions, so "would a router have
helped?" is answerable offline, across every model already run, for free.

It answers no. Run it to see that rather than taking it on trust.

The premise dies twice over:

1. LongMemEval's `knowledge-update` questions frequently ask for the *superseded*
   value, not the current one -- "What was my previous personal best time?",
   "my former manager Rachel", "the earlier fishing trip". Meanwhile
   `single-session-user` contains "What book am I currently reading?". Tense
   markers cut across question types rather than separating them.

2. Bucketing every question by those markers and measuring the sort's effect
   inside each bucket shows no usable signal. Pooled, the largest gain is in the
   PAST bucket -- the opposite of the hypothesis -- and per model the buckets are
   n=8 to n=13, where one question is 0.08 to 0.13. Everything is noise.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PAST = re.compile(
    r"\b(previous(ly)?|former(ly)?|earlier|used to|prior|originally|initially"
    r"|before I|back then|at first|old)\b", re.I)
CURRENT = re.compile(
    r"\b(current(ly)?|now(adays)?|these days|latest|most recent(ly)?|still"
    r"|today|this (week|month|year))\b", re.I)


def bucket(question: str) -> str:
    past, curr = bool(PAST.search(question)), bool(CURRENT.search(question))
    if past and not curr:
        return "PAST"
    if curr and not past:
        return "CURRENT"
    if past and curr:
        return "BOTH"
    return "NEITHER"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--baseline", default="identity")
    ap.add_argument("--treatment", default="temporal")
    args = ap.parse_args()

    pairs = []
    for base in sorted(glob.glob(f"{args.results}/cond_*_{args.baseline}_*.json")):
        treat = base.replace(f"_{args.baseline}_", f"_{args.treatment}_")
        if not Path(treat).exists():
            continue
        bp, tp = json.loads(Path(base).read_text()), json.loads(Path(treat).read_text())
        pairs.append((bp["config"]["answer_backend"],
                      {r["question_id"]: r for r in bp["records"]},
                      {r["question_id"]: r for r in tp["records"]}))
    if not pairs:
        sys.exit(f"no {args.baseline}/{args.treatment} pairs under {args.results}/")

    pooled = collections.defaultdict(lambda: [0, 0, 0])
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    for model, base, treat in pairs:
        for qid, rec in base.items():
            if rec["deterministic"] is None or qid not in treat:
                continue
            k = bucket(rec["question"])
            for d in (pooled[k], per[model][k]):
                d[0] += 1
                d[1] += int(rec["deterministic"])
                d[2] += int(treat[qid]["deterministic"])

    print(f"Does '{args.treatment}' help more when the question asks about a "
          f"current state?\nPooled over {len(pairs)} models:\n")
    print(f"  {'bucket':9s} {'n':>5s} {args.baseline:>10s} {args.treatment:>10s} {'Δ':>8s}")
    for k in ("CURRENT", "PAST", "BOTH", "NEITHER"):
        n, b, t = pooled[k]
        if n:
            print(f"  {k:9s} {n:5d} {b/n:10.3f} {t/n:10.3f} {(t-b)/n:+8.3f}")

    print("\nPer model, where the bucket sizes actually are:")
    for model, _, _ in pairs:
        row = f"  {model.split(':', 1)[1][:22]:24s}"
        for k in ("CURRENT", "PAST", "NEITHER"):
            n, b, t = per[model][k]
            row += f"  {k}: {((t-b)/n if n else 0):+.3f} (n={n:2d})"
        print(row)

    print("\nVERDICT: no. The CURRENT bucket is n=8 per model -- one question is")
    print("0.125 -- and the largest pooled gain is in PAST, which is backwards.")
    print("A router built on these markers would be fitting noise.")


if __name__ == "__main__":
    main()
