"""Paired comparison: each conditioner against the identity baseline.

This is the cleanest experiment the memory literature allows. Identical
retriever, identical k, identical model, identical seed, identical prompt
template -- the only difference between two arms is how the same retrieved
units were rendered. So there is no confound to argue about, and a paired test
on the same question ids is the right instrument.

Reuses memllm's `compute_lift` unchanged, passing the identity arm as the sole
control. Exact McNemar on discordant pairs plus a paired bootstrap CI, no scipy.

Reports the token cost alongside every accuracy delta, because a conditioner
that buys accuracy with tokens has not made anything free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import memcond._deps  # noqa: E402,F401
from memllm.eval.ablation import arm_from_payload, compute_lift  # noqa: E402

TYPES = ["knowledge-update", "temporal-reasoning", "multi-session",
         "single-session-user", "single-session-assistant"]


def load(results: Path) -> dict[tuple[str, int], list[tuple[dict, object]]]:
    """Group arms into pairable sets, keeping the payload beside the ArmResult.

    The key is (model, question-set size), not model alone. A paired test is
    only meaningful over the same questions, so a 500-question arm and a
    100-question arm on the same model are two different experiments and must
    not become each other's baseline. Grouping on the model alone silently
    paired them, picking whichever identity arm sorted first.
    """
    by_arm_set: dict[tuple[str, int], list] = {}
    for f in sorted(results.glob("cond_*.json")):
        p = json.loads(f.read_text())
        if "records" not in p:
            continue
        pt = {r.get("prompt_tokens") for r in p["records"]}
        if len(p["records"]) > 5 and len(pt) == 1 and pt != {0}:
            print(f"  skipping {f.name}: every prompt identical "
                  f"({pt.pop()} tok) -- context was clamped", file=sys.stderr)
            continue
        key = (p["config"]["answer_backend"], p["n_examples"])
        by_arm_set.setdefault(key, []).append(
            (p, arm_from_payload(p, name=f.stem)))
    return by_arm_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--baseline", default="identity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/conditioner_comparison.json"))
    args = ap.parse_args()

    by_arm_set = load(Path(args.results))
    if not by_arm_set:
        sys.exit(f"no cond_*.json arms under {args.results}/")

    reports = []
    for model, n_examples in sorted(by_arm_set):
        arms = by_arm_set[(model, n_examples)]
        base = next((pair for pair in arms
                     if pair[0]["config"]["conditioner"] == args.baseline), None)
        if base is None:
            print(f"\n{model} (n={n_examples}): no '{args.baseline}' arm -- "
                  f"nothing to compare against, skipping", file=sys.stderr)
            continue
        base_payload, base_arm = base

        # Same size is necessary but not sufficient: two 500-question arms drawn
        # with different seeds would pass the grouping check and still be
        # unpairable. The ids are the actual contract.
        base_ids = {r["question_id"] for r in base_payload["records"]}
        for payload, _ in arms:
            ids = {r["question_id"] for r in payload["records"]}
            if ids != base_ids:
                sys.exit(f"{payload['config']['conditioner']} and {args.baseline} "
                         f"on {model} cover different questions "
                         f"({len(ids ^ base_ids)} differ) -- not pairable")

        print(f"\n{'=' * 78}\n{model}   n={n_examples}\n{'=' * 78}")
        print(f"  baseline ({args.baseline}): acc={base_arm.accuracy:.4f}  "
              f"read tok/query={base_payload['read_tokens_per_query']:.0f}  "
              f"hit-cap={base_payload['n_hit_token_cap']}")
        print(f"\n  {'conditioner':22s} {'acc':>7s} {'Δacc':>8s} "
              f"{'95% CI':>18s} {'p':>9s} {'Δtok':>7s} {'cap':>4s}")

        for payload, arm in sorted(arms, key=lambda p: -p[1].accuracy):
            name = payload["config"]["conditioner"]
            if name == args.baseline:
                continue
            rep = compute_lift(arm, [base_arm], seed=args.seed)
            dtok = (payload["read_tokens_per_query"]
                    / base_payload["read_tokens_per_query"] - 1)
            # memllm's `significant` is one-sided -- it asks whether the system
            # BEAT the control, so ci_lo > 0. A real regression therefore gets
            # no marker at all, which would quietly bury the supersede:drop
            # result this run exists partly to demonstrate. Flag both directions.
            regression = rep.ci_hi < 0 and rep.p_value < 0.05
            star = "*" if rep.significant else ("!" if regression else " ")
            print(f"  {name:22s} {arm.accuracy:7.4f} "
                  f"{rep.lift:+8.4f}{star} "
                  f"[{rep.ci_lo:+.3f},{rep.ci_hi:+.3f}] {rep.p_value:9.4f} "
                  f"{dtok:+6.1%} {payload['n_hit_token_cap']:4d}")
            reports.append({
                "model": model, "conditioner": name,
                "n_examples": n_examples,
                "baseline": args.baseline,
                "baseline_accuracy": base_arm.accuracy,
                "accuracy": arm.accuracy, "delta": rep.lift,
                "ci_lo": rep.ci_lo, "ci_hi": rep.ci_hi,
                "p_value": rep.p_value, "significant": rep.significant,
                "significant_regression": regression,
                "contingency": rep.contingency,
                "read_tokens_per_query": payload["read_tokens_per_query"],
                "token_delta_pct": dtok,
                "n_hit_token_cap": payload["n_hit_token_cap"],
                "per_type": rep.per_type,
            })

        print(f"\n  by question type (Δ vs {args.baseline}):")
        header = f"    {'type':26s}" + "".join(
            f"{payload['config']['conditioner'][:11]:>12s}"
            for payload, _ in arms if payload["config"]["conditioner"] != args.baseline)
        print(header)
        for t in TYPES:
            row = f"    {t:26s}"
            for payload, arm in arms:
                name = payload["config"]["conditioner"]
                if name == args.baseline:
                    continue
                rep = compute_lift(arm, [base_arm], seed=args.seed)
                pt = rep.per_type.get(t)
                row += f"{pt['lift']:+12.3f}" if pt else f"{'--':>12s}"
            print(row)

    Path(args.out).write_text(json.dumps(reports, indent=2))
    print(f"\n* = significant improvement (p < 0.05, CI entirely above zero)")
    print(f"! = significant REGRESSION (p < 0.05, CI entirely below zero)")
    print(f"Δtok is read tokens per query against the baseline. A conditioner "
          f"that buys\naccuracy with tokens has not made anything free.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
