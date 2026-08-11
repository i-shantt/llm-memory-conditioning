"""Does conditioning a small model buy you a bigger one?

The tempting claim from the results table is that a conditioned 1.5B reaches a
7B's accuracy. memllm explicitly warns against exactly this comparison: the
grader marks an answer correct when the gold span appears anywhere in it, so a
more verbose model gets more chances, and 7B's median answer is 30 words against
1.5B's 16. A cross-model gap can be verbosity wearing accuracy's clothes.

So the claim is only worth making if it survives two checks that a within-model
comparison does not need:

1. **Paired significance.** Every arm answers the same question ids, so
   conditioned-1.5B against unconditioned-7B is a paired comparison and exact
   McNemar applies. "Indistinguishable" has to mean a test failed to separate
   them, not that two point estimates looked close.

2. **Length control.** Re-grade the same stored answers with each truncated to
   its first N words. An advantage that survives the cap is accuracy; one that
   evaporates was length. This is memllm's own length-decay method, applied
   here to the conditioner arms.

Both run offline from the stored predictions. No GPU, no benchmark download.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import memcond._deps  # noqa: E402,F401  (puts memllm on sys.path)
from memllm.eval.grade import grade  # noqa: E402

# None is "as generated". The median gold answer is 11 characters, so even the
# 8-word cap leaves far more room than a correct answer needs.
CAPS = (None, 40, 25, 15, 8)


def graded(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {r["question_id"]: r for r in payload["records"]
            if r.get("deterministic") is not None}


def mcnemar_exact(a: dict[str, dict], b: dict[str, dict]) -> tuple:
    """Two-sided exact McNemar over the question ids both arms graded."""
    ids = sorted(set(a) & set(b))
    only_a = sum(1 for i in ids if a[i]["deterministic"] and not b[i]["deterministic"])
    only_b = sum(1 for i in ids if b[i]["deterministic"] and not a[i]["deterministic"])
    n = only_a + only_b
    # Under H0 each discordant pair is a fair coin, so the tail is binomial.
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n) \
        if n else 1.0
    acc_a = sum(a[i]["deterministic"] for i in ids) / len(ids)
    acc_b = sum(b[i]["deterministic"] for i in ids) / len(ids)
    return len(ids), acc_a, acc_b, only_a, only_b, p


def capped_accuracy(records: dict[str, dict], cap: int | None) -> float:
    scores = [grade(" ".join(r["pred"].split()[:cap]) if cap else r["pred"],
                    r["gold"], r["is_abstention"]) for r in records.values()]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--treatment", default="cond_1.5b_all_k10_n100.json",
                    help="the conditioned small-model arm")
    ap.add_argument("--against", nargs="+",
                    default=["cond_3b_identity_k10_n100.json",
                             "cond_7b_identity_k10_n100.json"],
                    help="unconditioned larger-model baselines")
    args = ap.parse_args()

    root = Path(args.results)
    treat = graded(root / args.treatment)
    label = args.treatment[5:].split("_k")[0]

    print(f"Treatment: {label}  (n={len(treat)} graded)\n")
    print("## Paired, against each unconditioned baseline\n")
    for name in args.against:
        base = graded(root / name)
        n, at, ab, only_t, only_b, p = mcnemar_exact(treat, base)
        verdict = ("indistinguishable" if p > 0.05 else
                   "separated at p<0.05, " + ("treatment ahead" if at > ab else "baseline ahead"))
        print(f"  vs {name[5:].split('_k')[0]:16s} "
              f"{at:.4f} vs {ab:.4f}  Δ={at - ab:+.4f}  "
              f"discordant {only_t}/{only_b}  p={p:.3f}  -- {verdict}")

    print("\n## Length-controlled, answers truncated to their first N words\n")
    header = " | ".join("full" if c is None else f"{c}w" for c in CAPS)
    print(f"  {'arm':22s} {'med words':>9s}  {header}")
    rows = {}
    for name in [args.treatment, *args.against]:
        recs = graded(root / name)
        med = sorted(len(r["pred"].split()) for r in recs.values())[len(recs) // 2]
        rows[name] = [capped_accuracy(recs, c) for c in CAPS]
        cells = "  ".join(f"{v:.3f}" for v in rows[name])
        print(f"  {name[5:].split('_k')[0]:22s} {med:9d}  {cells}")

    print("\n  gap at each cap, treatment minus baseline:")
    for name in args.against:
        gaps = "  ".join(f"{rows[args.treatment][i] - rows[name][i]:+.3f}"
                         for i in range(len(CAPS)))
        print(f"  {'vs ' + name[5:].split('_k')[0]:22s} {'':9s}  {gaps}")

    print("\nA gap that holds its sign across every column is not the grader "
          "rewarding\nlonger answers. A gap that decays toward zero as the cap "
          "tightens partly was.")


if __name__ == "__main__":
    main()
