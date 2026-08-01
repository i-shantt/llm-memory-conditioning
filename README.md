# lastmile — retrieval already worked. The answer is still wrong.

Memory research optimises **recall**: did the system find the evidence? On
LongMemEval, good retrievers now essentially have. Hybrid retrieval scores
`any_hit@10 = 1.000` on the `knowledge-update` slice — the evidence is in the
prompt every single time.

A 7B model still gets 9 of those 16 questions wrong.

Joining per-question retrieval hits against per-question correctness across
[memllm](../memllm)'s stored runs (Qwen2.5-7B, hybrid, k=10, n=100):

| question type | evidence retrieved, answer **wrong** | accuracy given evidence present |
|---|---|---|
| knowledge-update | 9 / 16 | 0.438 |
| temporal-reasoning | 13 / 21 | 0.381 |
| multi-session | 15 / 22 | 0.318 |
| single-session-assistant | 4 / 10 | 0.600 |
| single-session-user | 3 / 12 | 0.750 |

**44 of 91 graded questions had the answer sitting in the prompt and were
answered wrong anyway.** Recall is not the binding constraint. What the model
does with the retrieved text is.

This repo is a **read-time context conditioner**: a deterministic transform
applied to the k retrieved units after retrieval, before prompt assembly. No LLM
calls, no model, no training, no write-time work. It is a layer, not a
competitor — it sits on top of any retriever or memory system.

## What is and is not new here

The phenomenon is **not** a discovery. STALE (arXiv 2605.06527) names it the
*current-state adjudication gap*; the Always-On Agents survey (2606.30306)
repeats it. Both are worth reading first.

What is missing from that literature is the **cost of the fix**. STALE's own
remedy, CUPMem, takes a backbone from 8.7% to 68.0% using an LLM adjudicator on
**every write** — O(corpus) LLM calls — and reports no cost analysis and no
control arms. TISER, TimeRefine and TG-LLM all answer with test-time reasoning
or fine-tuning. Nobody has published the deterministic, zero-LLM, read-time
version, and nobody prices theirs.

That gap is the point. Write-time consolidation pays LLM calls proportional to
the corpus whether or not a query ever arrives. Conditioning pays nothing, ever,
and touches only the k units a query actually retrieved.

## The conditioners

| name | what it does | targets |
|---|---|---|
| `identity` | the baseline — memllm's current rendering, byte for byte | control |
| `supersede:mark` | groups retrieved units that assert conflicting values for the same query term, labels the newest date `LATEST` and earlier ones `OUTDATED` | knowledge-update |
| `supersede:order` | states the ordering within a topic, claims nothing about currency | safe variant |
| `supersede:drop` | deletes the superseded units | the case *against* deletion |
| `temporal` | sorts chronologically, labels each unit with days/weeks/months before the question date | temporal-reasoning |
| `safe` / `all` | compositions | — |

`temporal` exists because the models demonstrably cannot do the arithmetic. From
memllm's `oracle` arm, where every evidence turn is in the prompt by
construction:

```
1.5B  "How many months since I last visited a museum?"   gold 5, said "One month"
7B    "How many weeks had I been taking sculpting classes...?"
      "You started on 2023/01/12 and got tools on 2023/03/04. This means you
       have been taking classes fo"          <- cut off mid-subtraction
```

Subtraction is free and exact in Python. Doing it at render time turns "compare
two dates" into "read a number".

## The CPU gate

`scripts/run_mechanical_gate.py` scores a conditioner's *decisions* against
LongMemEval's own `has_answer` labels — no model, no GPU, no quota. It exists so
a broken rule is caught before anyone buys GPU time.

All 500 LongMemEval questions, BM25 at k=10:

```
knowledge-update (n=78) — the slice supersession is for
conditioner             fired   HARM   PREC    n  older   surv   tok Δ  llm
supersede:mark:naive    0.974  0.079  0.921   63  1.000  1.000   +7.0%    0
supersede:mark          0.538  0.036  0.964   28  1.000  1.000   +2.5%    0
supersede:drop          0.538  0.000  1.000   27     --  0.781  -23.7%    0
temporal                0.000     --     --    0     --  1.000   +7.6%    0

all question types (n=500)
supersede:mark:naive    0.746  0.312  0.688  240  0.956  1.000   +4.4%    0
supersede:mark          0.278  0.209  0.791   67  0.982  1.000   +1.1%    0
supersede:drop          0.278  0.000  1.000   53  0.000  0.883  -11.1%    0
temporal                0.000     --     --    0     --  1.000   +5.9%    0
```

- **HARM** — the newest evidence unit marked `OUTDATED`, i.e. the rule telling
  the model the current answer is stale. Want ~0.
- **PREC** — the newest evidence unit marked `LATEST`. Want high.
- **surv** — evidence units not dropped. Must be 1.000 for any `mark` mode.

The rule is accurate exactly where the phenomenon it models exists, and noisy
elsewhere: on knowledge-update it marks the right unit current 92–96% of the
time, while across all question types precision drops to 0.69–0.79 because it
fires where there is no supersession to find.

`supersede:drop` deletes **22% of all evidence** on knowledge-update. That is the
concrete argument against write-time deletion, which is what Mem0's UPDATE and
DELETE operations do irreversibly — measured rather than asserted.

### The gate earned its keep twice

Both of these looked like the idea failing, and both were bugs:

**Slot fragmentation.** Units were keyed on the *rarest* query term they
contained, on the theory that a specific word identifies the units in contention
better than a generic one. It does the opposite. For "What time do I usually go
to the gym?", `I go to the gym at 7pm` keys on "gym" while `I've moved my gym
time to 6pm` keys on "time" — so the two mentions that actually conflict landed
in different slots and neither was annotated. Grouping needs the *shared* term.

**Same-date siblings.** Ordering inside a slot broke ties by turn index, so an
assistant's follow-up outranked the user turn that stated the fact. The gate
reported the newest evidence unit marked `OUTDATED` on **93%** of
knowledge-update questions. Supersession is a claim about dates; same-day
siblings supersede nothing. Fixing it moved HARM from 0.933 to 0.079.

A third correction was to the gate itself. Its first harm metric counted *any*
evidence unit marked `OUTDATED` and reported 0.681, which looked disqualifying.
It was wrong: 67 of 78 knowledge-update questions carry **two** evidence turns on
two dates — the superseded fact and its replacement — because answering needs
both. Marking the earlier one `OUTDATED` is correct, and the metric was
penalising it.

## Status

The gate passes. **Answer accuracy has not been measured yet** — that needs GPU,
and no accuracy claim appears anywhere in this repo until it has been run with
memllm's control arms and paired significance tests.

The experiment is unusually clean when it happens: identical retriever, k, model
and seed, with the *only* difference being how the same retrieved units are
rendered. A paired McNemar on that has no confound to argue about.

## Layout

```
lastmile/conditioner/   base.py, supersede.py, temporal.py — the transforms
lastmile/eval/          mechanical.py — the CPU gate's metrics
scripts/                run_mechanical_gate.py
tests/                  48 tests, no model, no network, no benchmark download
```

## Running it

Needs a [memllm](../memllm) checkout beside this one, or `MEMLLM_PATH` set. It
supplies the cost ledger, the audited grader and the lift statistics — reusing
them is deliberate, because a number from this repo is only worth something if
it is measured the way memllm's were.

```bash
python -m pytest tests/ -q
python scripts/run_mechanical_gate.py --limit 100 --retriever bm25
```

## Honest limits

- **No accuracy result exists yet.** Everything above measures decisions against
  evidence labels, not answers. A conditioner can be mechanically correct and
  still not help a model.
- **Away from knowledge-update the rule is noisy.** Across all 500 questions
  HARM is 0.209–0.312 — it fires where there is no supersession to find. Whether
  that costs accuracy is exactly what the GPU run is for, and it is the reason
  `supersede:order`, which claims no currency at all, exists as a fallback.
- **`fired` is not `n`.** `supersede:mark` fires on 53.8% of knowledge-update
  questions but annotates the *newest evidence unit* on 28 of 78, which is what
  HARM and PREC are rates over. The rest of the time it annotated other units.
- **The gate uses BM25**, not the hybrid retriever the accuracy arms will use.
  BM25 also scores `any_hit@10 = 1.000` on knowledge-update, so the evidence is
  equally present, but the distractors differ.
- **`is_evidence` is LongMemEval's own labelling** and inherits whatever errors
  it contains.
- **Conditioning cannot fix a retrieval miss.** It only changes the presentation
  of what was already retrieved, so its ceiling is the 44-of-91 slice above.
