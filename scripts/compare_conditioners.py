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


def load(results: Path) -> dict[str, list[tuple[dict, object]]]:
    """Group arms by model, keeping the payload alongside the ArmResult."""
    by_model: dict[str, list] = {}
    for f in sorted(results.glob("cond_*.json")):
        p = json.loads(f.read_text())
        if "records" not in p:
            continue
        pt = {r.get("prompt_tokens") for r in p["records"]}
        if len(p["records"]) > 5 and len(pt) == 1 and pt != {0}:
            print(f"  skipping {f.name}: every prompt identical "
                  f"({pt.pop()} tok) -- context was clamped", file=sys.stderr)
            continue
        by_model.setdefault(p["config"]["answer_backend"], []).append(
            (p, arm_from_payload(p, name=f.stem)))
    return by_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--baseline", default="identity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/conditioner_comparison.json"))
    args = ap.parse_args()

    by_model = load(Path(args.results))
    if not by_model:
        sys.exit(f"no cond_*.json arms under {args.results}/")

    reports = []
    for model in sorted(by_model):
        arms = by_model[model]
        base = next((pair for pair in arms
                     if pair[0]["config"]["conditioner"] == args.baseline), None)
        if base is None:
            print(f"\n{model}: no '{args.baseline}' arm -- nothing to compare "
                  f"against, skipping", file=sys.stderr)
            continue
        base_payload, base_arm = base

        print(f"\n{'=' * 78}\n{model}\n{'=' * 78}")
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
