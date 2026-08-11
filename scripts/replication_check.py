"""Why the 100-question result did not survive the full benchmark.

This repo's first headline was +13.2 points on Qwen2.5-1.5B, p = 0.002, measured
on a stratified 100-question sample. Re-running the same two arms over all 500
LongMemEval questions put it at +0.7, p = 0.79.

Nothing was wrong with the original measurement, and this script is here to show
that rather than assert it. It splits the full run three ways:

  * the whole benchmark
  * the 91 gradable questions the original sample drew
  * the 355 it never saw

The middle row reproduces the original result. The bottom row is slightly
negative. The sampled questions were ones where the *baseline* did unusually
badly -- identity scores 0.286 there against 0.409 elsewhere -- so most of the
original gap was the control's bad draw, not the treatment's good one.

That is a failure mode a paired test cannot detect. McNemar holds the question
fixed and varies only the conditioner, which is exactly right, and says nothing
about whether the questions themselves represent the population.

Also prices the best case for the idea: an oracle router that applies the
transform only to the question types it helps, using LongMemEval's own type
labels. It cheats, and it still only reaches +2.2 points.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The two question types the sort mechanism predicts it should help: a fact that
# was later revised, and a question about when something happened.
HELPED = ("knowledge-update", "temporal-reasoning")


def graded(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {r["question_id"]: r for r in payload["records"]
            if r.get("deterministic") is not None}


def mcnemar_exact(ids, control, system) -> tuple:
    fixed = sum(1 for i in ids if system[i]["deterministic"] and not control[i]["deterministic"])
    broke = sum(1 for i in ids if control[i]["deterministic"] and not system[i]["deterministic"])
    n = fixed + broke
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(fixed, broke) + 1)) / 2**n) \
        if n else 1.0
    acc_c = sum(control[i]["deterministic"] for i in ids) / len(ids)
    acc_s = sum(system[i]["deterministic"] for i in ids) / len(ids)
    return acc_c, acc_s, fixed, broke, p


def row(label, ids, control, system) -> None:
    acc_c, acc_s, fixed, broke, p = mcnemar_exact(ids, control, system)
    print(f"  {label:34s} n={len(ids):4d}  {acc_c:.4f} -> {acc_s:.4f}  "
          f"Δ={acc_s - acc_c:+.4f}  fixed {fixed:3d} broke {broke:3d}  p={p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--full-control", default="cond_qwen2.5-1.5b-instruct_identity_k10_n500.json")
    ap.add_argument("--full-system", default="cond_qwen2.5-1.5b-instruct_all_k10_n500.json")
    ap.add_argument("--sampled", default="cond_1.5b_identity_k10_n100.json",
                    help="the small run, read only for which question ids it drew")
    args = ap.parse_args()

    root = Path(args.results)
    control, system = graded(root / args.full_control), graded(root / args.full_system)
    sampled = {r["question_id"]
               for r in json.loads((root / args.sampled).read_text())["records"]}

    all_ids = sorted(control)
    seen = [i for i in all_ids if i in sampled]
    unseen = [i for i in all_ids if i not in sampled]

    print("## The same comparison, on three question sets\n")
    row("the whole benchmark", all_ids, control, system)
    row("the original sample", seen, control, system)
    row("the questions it never saw", unseen, control, system)

    base_seen = sum(control[i]["deterministic"] for i in seen) / len(seen)
    base_unseen = sum(control[i]["deterministic"] for i in unseen) / len(unseen)
    sys_seen = sum(system[i]["deterministic"] for i in seen) / len(seen)
    sys_unseen = sum(system[i]["deterministic"] for i in unseen) / len(unseen)
    print(f"\n  The baseline moves {base_unseen - base_seen:+.4f} between the two sets "
          f"({base_seen:.4f} -> {base_unseen:.4f});")
    print(f"  the treatment moves {sys_unseen - sys_seen:+.4f} "
          f"({sys_seen:.4f} -> {sys_unseen:.4f}).")
    print("  The sample was unrepresentative of the control, not of the treatment.")

    print("\n## By question type, on the full benchmark\n")
    by = defaultdict(lambda: [0, 0, 0])
    for i in all_ids:
        t = control[i]["question_type"]
        by[t][0] += 1
        by[t][1] += control[i]["deterministic"]
        by[t][2] += system[i]["deterministic"]
    print(f"  {'question type':28s} {'n':>4s} {'identity':>9s} {'all':>7s} {'Δ':>8s}")
    for t, (n, c, s) in sorted(by.items(), key=lambda kv: -(kv[1][2] - kv[1][1]) / kv[1][0]):
        mark = "  <- predicted to gain" if t in HELPED else ""
        print(f"  {t:28s} {n:4d} {c/n:9.3f} {s/n:7.3f} {(s-c)/n:+8.3f}{mark}")

    print()
    row("types predicted to gain", [i for i in all_ids
                                    if control[i]["question_type"] in HELPED], control, system)
    row("types predicted to lose", [i for i in all_ids
                                    if control[i]["question_type"] not in HELPED], control, system)

    print("\n## The ceiling: an oracle router that reads the answer key\n")
    routed = {i: (system if control[i]["question_type"] in HELPED else control)[i]
              for i in all_ids}
    base = sum(control[i]["deterministic"] for i in all_ids) / len(all_ids)
    every = sum(system[i]["deterministic"] for i in all_ids) / len(all_ids)
    _, acc_r, fixed, broke, p_r = mcnemar_exact(all_ids, control, routed)
    print(f"  identity everywhere        {base:.4f}")
    print(f"  conditioned everywhere     {every:.4f}   ({every - base:+.4f})")
    print(f"  oracle type-routed         {acc_r:.4f}   ({acc_r - base:+.4f})  "
          f"fixed {fixed} broke {broke}  p={p_r:.4f}")
    print("\n  Not a system -- it uses LongMemEval's own type label to decide how to")
    print("  format the prompt. It is the most this idea could be worth, and it is")
    print("  about two points, and it is not significant.")


if __name__ == "__main__":
    main()
