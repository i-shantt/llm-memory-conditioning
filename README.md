# llm-memory-conditioning

**Retrieval already worked. The answer is still wrong.**

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

**Measured: +0.132 accuracy on Qwen2.5-1.5B (p = 0.002), for 6.8% more read
tokens and zero LLM calls** — enough that a 1.5B model matches a 7B one with no
conditioning. Nothing is significant at 7B, which is the point: the transforms
do work the small model cannot do for itself. [Results below.](#results)

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
a broken rule is caught before anyone buys GPU time. It caught two real bugs.
It also certified a conditioner that turned out to do nothing at all, which is
its limit — see [the annotation nobody read](#the-annotation-nobody-read).

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

## Results

Qwen2.5, BM25 at k=10, n=100, `max_new_tokens=256`, deterministic grading.
Each conditioner is paired against the `identity` arm on the same 91 graded
question ids — same retriever, same k, same model, same seed, same prompt
template. Exact McNemar on discordant pairs, paired bootstrap CI.

**1.5B** (baseline `identity` = 0.2747)

| conditioner | acc | Δ | 95% CI | p | Δ read tokens | LLM calls |
|---|---|---|---|---|---|---|
| `all` | 0.4066 | **+0.1319** | [+0.055, +0.209] | **0.0018** | +6.8% | 0 |
| `temporal` | 0.3846 | **+0.1099** | [+0.044, +0.187] | **0.0063** | +5.7% | 0 |
| `supersede:mark` | 0.2637 | −0.0110 | [−0.044, +0.022] | 1.000 | +1.0% | 0 |

**7B** (baseline `identity` = 0.4176)

| conditioner | acc | Δ | 95% CI | p | Δ read tokens | LLM calls |
|---|---|---|---|---|---|---|
| `all` | 0.4615 | +0.0440 | [−0.033, +0.121] | 0.424 | +6.9% | 0 |
| `temporal` | 0.4286 | +0.0110 | [−0.066, +0.088] | 1.000 | +5.8% | 0 |
| `supersede:drop` | 0.4066 | −0.0110 | [−0.066, +0.044] | 1.000 | −8.3% | 0 |
| `supersede:mark` | 0.4066 | −0.0110 | [−0.044, +0.022] | 1.000 | +1.0% | 0 |

### Conditioning substitutes for model capacity

Everything significant happens at 1.5B. Nothing at 7B is distinguishable from
noise. That is the result, not a disappointment: the transforms do work the
model cannot do for itself, and a 7B model can already do it.

The comparison that makes the point:

| | accuracy | token-F1 | median answer |
|---|---|---|---|
| 1.5B `identity` | 0.2747 | 0.1755 | 14 words |
| 1.5B `all` | **0.4066** | **0.1821** | 16 words |
| 7B `identity` | 0.4176 | 0.1759 | 30 words |

**A 1.5B model with free conditioning matches a 7B model without it** — 0.4066
against 0.4176, a one-question difference at 4.7× fewer parameters.

That comparison is exactly the kind memllm warns about, because containment
rewards length and 7B answers are twice as long. So check it against token-F1,
which penalises length instead: scaling 1.5B→7B buys **+0.1429 accuracy but only
+0.0005 token-F1** — the scaling gain is almost entirely verbosity. Conditioning
at 1.5B buys +0.1319 accuracy *and* **+0.0066 token-F1**, thirteen times the
scaling gain on the metric that cannot be inflated by talking more. Median answer
length barely moves (14 → 16 words). The answers are not longer. They are right
more often.

### The annotation nobody read

`supersede:mark` did **nothing**: −0.0110 at both 1.5B and 7B.

This is the part worth dwelling on, because it is where the CPU gate failed. The
gate certified this conditioner: on knowledge-update it marked the newest
evidence unit `LATEST` **96.4%** of the time and mislabelled it 3.6%. The
annotations were right. The model ignored them.

The gate measured whether the labels were *correct*. It could not measure whether
they were *used*, and those turned out to be different questions. Any
mechanical-correctness harness has this blind spot, and no amount of raising the
precision bar would have caught it.

Worse for the original hypothesis: the knowledge-update gain that supersession
was designed for was delivered by `temporal` instead — **+0.250 at 1.5B**,
against `supersede:mark`'s +0.000 on the same slice.

| Δ by question type, 1.5B | `temporal` | `supersede:mark` | `all` |
|---|---|---|---|
| knowledge-update | **+0.250** | +0.000 | **+0.250** |
| temporal-reasoning | +0.120 | +0.000 | +0.200 |
| multi-session | +0.077 | +0.038 | +0.077 |
| single-session-assistant | +0.100 | −0.100 | +0.100 |
| single-session-user | +0.000 | −0.071 | +0.000 |

`temporal` makes no claim about currency at all. It sorts chronologically and
labels each unit with its age. That is apparently enough: **the model does not
need to be told which fact is current, it needs the chronology made legible
enough to work it out.** Being told directly, in a label with 96% precision,
changed nothing.

### A correction found by reading the flips

The first version of these numbers had `temporal` at +0.1209. Checking *which*
twelve questions it fixed turned one up that was not a conditioning effect at
all: both arms refused an abstention question, but `identity` wrote "the
excerpts provided **do** not contain information" and the grader's refusal
patterns only matched "**does** not contain". One arm's refusal was detected and
the other's was not, which is a fake +1.

Fixed in memllm and re-graded through `scripts/regrade.py --pattern 'cond_*.json'`,
which is why the numbers above are slightly smaller than a first reading would
have given. Aggregate accuracy would never have shown this. Reading the individual
flips did.

### Deletion

`supersede:drop` came out at −0.0110, not significant, for −8.3% read tokens. The
CPU gate showed it deleting 22% of all evidence on knowledge-update, so a sharp
loss was the prediction; it merely failed to help. The honest reading is weaker
than "deletion is harmful": deletion bought nothing, while costing an
irreversible loss of information that a past-directed question would need.

### Cost

Every number above was produced with **zero LLM calls** and 1–7% more read
tokens. STALE's CUPMem reaches its numbers with an LLM adjudicator on every
write, which is O(corpus) calls paid whether or not a query arrives. This is the
comparison the literature does not report.

## Layout

```
memcond/conditioner/   base.py, supersede.py, temporal.py — the transforms
memcond/eval/          mechanical.py — the CPU gate's metrics
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

- **Mechanical correctness does not imply usefulness, and the gate cannot tell
  the difference.** `supersede:mark` passed at 96.4% precision and moved accuracy
  by 0.000. Treat every gate number as necessary, never sufficient.
- **One benchmark, one model family, two sizes, n=100.** Whether the 1.5B result
  holds for other small models, or at 3B, is untested. The 7B nulls have CIs
  wide enough (±0.08) to hide a real effect of the size seen at 1.5B.
- **The cross-model claim is the fragile one.** "1.5B conditioned matches 7B" is
  a comparison memllm explicitly warns against, because containment rewards
  length. It survives a token-F1 check, but token-F1 differences here are small
  in absolute terms (0.1821 vs 0.1759) and the arms were not designed for a
  cross-model test.
- **Away from knowledge-update the rule is noisy.** Across all 500 questions
  HARM is 0.209–0.312 — it fires where there is no supersession to find. Whether
  that costs accuracy is exactly what the GPU run is for, and it is the reason
  `supersede:order`, which claims no currency at all, exists as a fallback.
- **`fired` is not `n`.** `supersede:mark` fires on 53.8% of knowledge-update
  questions but annotates the *newest evidence unit* on 28 of 78, which is what
  HARM and PREC are rates over. The rest of the time it annotated other units.
- **Everything here uses BM25**, not a hybrid retriever. That started as a
  workaround -- sentence-transformers crashed on the Kaggle GPU with "no kernel
  image is available for execution on the device" -- but it is the better
  choice: BM25 also scores `any_hit@10 = 1.000` on knowledge-update, and the
  gate and the accuracy arms now describe the same retrieval rather than two
  different ones. Whether these conditioners behave the same over a dense
  retriever's distractors is untested.
- **`is_evidence` is LongMemEval's own labelling** and inherits whatever errors
  it contains.
- **Conditioning cannot fix a retrieval miss.** It only changes the presentation
  of what was already retrieved, so its ceiling is the 44-of-91 slice above.
