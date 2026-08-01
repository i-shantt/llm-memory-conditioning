"""CPU-only gate. Run this before spending any GPU on a conditioner.

Retrieves with a real retriever, conditions the result, and scores the
conditioner's decisions against LongMemEval's `has_answer` labels. No model, no
API, no quota.

    python scripts/run_mechanical_gate.py --limit 100 --conditioners supersede:mark temporal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import memcond._deps  # noqa: E402,F401
from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import load_examples, stratified_subset  # noqa: E402

from memcond.conditioner import build  # noqa: E402
from memcond.eval.mechanical import GateTally, score_question  # noqa: E402


def build_retriever(name: str, embed_model: str, device: str):
    if name == "bm25":
        from memllm.retrieval.bm25 import BM25Retriever
        return BM25Retriever()
    if name == "hybrid":
        from memllm.retrieval.hybrid import HybridRetriever
        return HybridRetriever(model_name=embed_model, device=device)
    if name == "oracle":
        from memllm.retrieval import baselines
        return baselines.OracleRetriever()
    raise ValueError(f"unknown retriever: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="default: <memllm>/data/raw/longmemeval_s")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "hybrid", "oracle"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--granularity", default="turn")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conditioners", nargs="+",
                    default=["supersede:mark", "supersede:drop", "temporal", "all"])
    ap.add_argument("--out", default="results/mechanical_gate.json")
    args = ap.parse_args()

    data = args.data or str(memcond._deps.memllm_root() / "data/raw/longmemeval_s")
    examples = load_examples(data)
    if args.limit and args.limit < len(examples):
        examples = stratified_subset(examples, args.limit, seed=0)
    print(f"{len(examples)} questions from {data}", flush=True)

    tallies = {n: GateTally() for n in args.conditioners}
    conds = {n: build(n) for n in args.conditioners}
    ledger = CostLedger()

    for i, ex in enumerate(examples, start=1):
        units = ex.units(args.granularity)
        r = build_retriever(args.retriever, args.embed_model, args.device)
        r.index(units, ledger, f"{ex.question_id}|{args.granularity}")
        hits = r.search(ex.question, args.k, ledger, ex.question_date)
        chosen_ids = [h[0] for h in hits]
        by_id = {u.unit_id: u for u in units}
        sel = [by_id[u] for u in chosen_ids if u in by_id]

        for name, cond in conds.items():
            for t in (tallies[name], tallies[name].by_type.setdefault(
                    ex.question_type, GateTally())):
                score_question(cond, sel, ex.question, ex.question_date, t, ledger)
        if i % 25 == 0:
            print(f"  {i}/{len(examples)}", flush=True)

    report = {
        "config": vars(args) | {"data": data},
        "conditioners": {n: t.rates() for n, t in tallies.items()},
        "by_question_type": {
            n: {qt: t2.rates() for qt, t2 in t.by_type.items()}
            for n, t in tallies.items()
        },
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    def f(x, spec=".3f"):
        return "--" if x is None else format(x, spec)

    def table(title: str, get) -> None:
        print(f"\n{title}")
        print(f"{'conditioner':22s} {'fired':>6s} {'HARM':>6s} {'PREC':>6s} "
              f"{'n':>4s} {'older':>6s} {'surv':>6s} {'tok Δ':>7s} {'llm':>4s}")
        for n in tallies:
            r = get(n)
            if r is None:
                continue
            print(f"{n:22s} {f(r['fired']):>6s} "
                  f"{f(r['newest_evidence_outdated']):>6s} "
                  f"{f(r['newest_evidence_latest']):>6s} "
                  f"{r['n_newest_evidence_annotated']:>4d} "
                  f"{f(r['older_evidence_outdated']):>6s} "
                  f"{f(r['evidence_survival']):>6s} "
                  f"{f(r['token_delta_pct'], '+.1%'):>7s} {r['llm_calls']:>4d}")

    table("ALL QUESTION TYPES", lambda n: tallies[n].rates())
    ku = "knowledge-update"
    table(f"{ku.upper()} ONLY (the slice supersession is for)",
          lambda n: tallies[n].by_type[ku].rates() if ku in tallies[n].by_type else None)

    print("\n  HARM  = newest evidence unit marked OUTDATED. Want ~0: this is "
          "the rule\n          telling the model that the current answer is "
          "stale.")
    print("  PREC  = newest evidence unit marked LATEST. Want high.")
    print("  n     = questions where the rule annotated the newest evidence "
          "unit at all;\n          HARM and PREC are rates over n, so a small "
          "n means a thin result.")
    print("  older = older evidence units marked OUTDATED. On knowledge-update "
          "these\n          genuinely ARE superseded, so high is correct here.")
    print("  surv  = fraction of evidence units not dropped. Must be 1.000 for "
          "any\n          'mark' mode.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
