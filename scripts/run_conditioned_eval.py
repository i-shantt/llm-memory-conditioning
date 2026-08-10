"""End-to-end eval with a read-time conditioner between retrieval and prompt.

Mirrors memllm's `scripts/run_e2e_eval.py` loop exactly and emits the identical
payload schema, so `memllm/eval/ablation.py` and `scripts/run_ablation.py`
consume these arms with no changes. The single difference is one line: the
retrieved units go through a conditioner before the context string is built.

The `identity` conditioner reproduces memllm's rendering byte for byte (pinned
by `tests/test_conditioner.py`), so an `identity` arm here and a `hybrid` arm
there are the same experiment. That is what makes the comparison clean: same
retriever, same k, same model, same seed, same prompt -- only the rendering of
the same retrieved units differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import memcond._deps  # noqa: E402,F401
from memcond.conditioner import build as build_conditioner  # noqa: E402
from memcond.conditioner import condition, render_context  # noqa: E402

MEMLLM = memcond._deps.memllm_root()
sys.path.insert(0, str(MEMLLM / "scripts"))

from memllm.cost import CostLedger  # noqa: E402
from memllm.data.loader import load_examples, stratified_subset  # noqa: E402
from memllm.eval.grade import grade, is_extractive, token_f1  # noqa: E402
from memllm.generate.backends import build_backend  # noqa: E402
from memllm.retrieval.embed_cache import EmbeddingCache  # noqa: E402
from run_e2e_eval import ANSWER_PROMPT, CLOSED_BOOK_PROMPT  # noqa: E402
from run_retrieval_eval import build_retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(MEMLLM / "data/raw/longmemeval_s"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--conditioner", default="identity")
    ap.add_argument("--granularity", default="turn")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recency-weight", type=float, default=0.0)
    ap.add_argument("--answer-backend", default="ollama:qwen2.5:7b-instruct")
    # 256, not memllm's default of 64. At 64, 7B answers were cut off
    # mid-subtraction on temporal questions ("...have been taking classes fo"),
    # which understates exactly the slice a temporal conditioner targets.
    # memllm has since re-run its own arms at 256, so that truncation is no
    # longer visible in its stored predictions -- but the cap still has to be
    # set here, and both arms of every pair are run at the same one.
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=str(REPO / "results"))
    args = ap.parse_args()

    examples = stratified_subset(load_examples(args.data), args.limit, seed=args.seed)
    cache = EmbeddingCache(enabled=True)
    retriever = build_retriever(args.retriever, args, cache=cache)
    if hasattr(retriever, "warmup"):
        retriever.warmup()
    cond = build_conditioner(args.conditioner)

    answer_kw = {"max_new_tokens": args.max_new_tokens}
    if args.answer_backend.startswith("ollama:"):
        answer_kw["num_ctx"] = args.num_ctx
        answer_kw["timeout"] = args.gen_timeout
    answerer = build_backend(args.answer_backend, **answer_kw)
    print(f"answer backend: {answerer.name}")
    print(f"{len(examples)} examples, retriever={args.retriever}, "
          f"conditioner={cond.name}, k={args.k}, max_new={args.max_new_tokens}",
          flush=True)

    ledger = CostLedger()
    records: list[dict] = []
    n_det_correct = n_det_graded = n_not_gradable = 0
    f1_sum = 0.0
    t0 = time.perf_counter()

    for i, ex in enumerate(examples, 1):
        units = ex.units(args.granularity)
        by_id = {u.unit_id: u for u in units}
        retriever.index(units, ledger, f"{ex.question_id}|{args.granularity}")
        hits = retriever.search(ex.question, args.k, ledger, ex.question_date)

        chosen = [by_id[uid] for uid, _ in hits if uid in by_id]
        context = render_context(
            condition(cond, chosen, ex.question, ex.question_date, ledger)
        )
        if not context:
            prompt = CLOSED_BOOK_PROMPT.format(
                date=ex.question_date, question=ex.question)
        else:
            prompt = ANSWER_PROMPT.format(
                context=context, date=ex.question_date, question=ex.question)

        gen = answerer.generate(prompt)
        ledger.add_llm("read", gen.prompt_tokens, gen.completion_tokens)

        gold = str(ex.answer)
        det = grade(gen.text, gold, ex.is_abstention, ex.question)
        if det is None:
            n_not_gradable += 1
        else:
            n_det_graded += 1
            n_det_correct += int(det)
        f1_sum += token_f1(gen.text, gold)

        records.append({
            "question_id": ex.question_id, "question": ex.question,
            "question_type": ex.question_type, "gold": gold, "pred": gen.text,
            "is_abstention": ex.is_abstention,
            "extractive": is_extractive(gold),
            "deterministic": det, "judge": None, "raw_verdict": "",
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "hit_token_cap": gen.completion_tokens >= args.max_new_tokens,
        })
        if i % 10 == 0 or i == len(examples):
            acc = n_det_correct / max(n_det_graded, 1)
            print(f"  {i}/{len(examples)}  acc={acc:.3f}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    n = len(examples)
    by_type: dict[str, dict] = {}
    for r in records:
        if r["deterministic"] is None:
            continue
        b = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["deterministic"])

    tag = args.tag or f"cond_{args.conditioner.replace(':', '-')}_k{args.k}_n{n}"
    payload = {
        # `retriever` must stay the plain retriever name: ablation.py reads it to
        # decide which arms are controls.
        # `tag` must be the resolved value, not None: ablation.py names arms
        # from config["tag"].
        "config": vars(args) | {"answer_backend_name": answerer.name,
                                "judge_backend_name": None,
                                "conditioner": cond.name,
                                "tag": tag},
        "n_examples": n,
        "grader": "deterministic (memllm.eval.grade); audit in "
                  "memllm/results/grader_audit.json",
        "accuracy": n_det_correct / max(n_det_graded, 1),
        "n_graded": n_det_graded,
        "n_not_gradable": n_not_gradable,
        "not_gradable_note": "abstractive gold answers with no checkable surface "
                             "form; excluded rather than scored as wrong",
        "token_f1_mean": f1_sum / n,
        "accuracy_by_question_type": {
            t: {**v, "accuracy": v["correct"] / v["n"]}
            for t, v in sorted(by_type.items())
        },
        "cost_total": ledger.to_dict(),
        "read_tokens_per_query": (
            ledger.read.llm_prompt_tokens + ledger.read.llm_completion_tokens) / n,
        "prompt_tokens_max": max((r["prompt_tokens"] for r in records), default=0),
        "n_hit_token_cap": sum(r["hit_token_cap"] for r in records),
        "records": records,
    }

    out = Path(args.out_dir) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\naccuracy {payload['accuracy']:.4f} over {n_det_graded} graded"
          f"  |  read tok/query {payload['read_tokens_per_query']:.0f}"
          f"  |  hit cap {payload['n_hit_token_cap']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
