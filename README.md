# llm-memory-conditioning

[![tests](https://github.com/i-shantt/llm-memory-conditioning/actions/workflows/tests.yml/badge.svg)](https://github.com/i-shantt/llm-memory-conditioning/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Research code and measurements for one question:

> When a search system has already put the right text into a language model's
> prompt, how much accuracy is still lost in *how that text is laid out* — and
> how much of it can be recovered with plain code, without another model call?

The code is a small Python library (`memcond`) that rewrites retrieved
conversation history before it reaches the model: it sorts the pieces into date
order, labels each one with how long ago it happened, and does the date
arithmetic in advance. The measurements are 21 full evaluation runs across five
open models, with every prediction saved in `results/`.

This is a research repo, not a package. Nothing here is on PyPI, and it depends
on a sibling checkout of [memllm](https://github.com/i-shantt/memllm) for its
grader and statistics.

## Summary of findings

**The problem is real.** On the benchmark used here, retrieval finds the right
text and the model still answers wrong: on **30 of 91** test questions *every*
piece of evidence the benchmark labels as necessary was in the prompt and the
model got it wrong anyway.

**The fix works on one model family, and the effect shrinks with size.**

| model | accuracy change from the full transform |
|---|---|
| Qwen2.5 1.5B | **+13.2 points**, p = 0.002 |
| Qwen2.5 3B | +8.8 points, p = 0.057 |
| Qwen2.5 7B | +4.4 points, p = 0.424 |
| Gemma2 2B | −1.1 points, p = 1.000 |
| Llama3.2 3B | −5.5 points, p = 0.180 |

Only the 1.5B result is statistically significant. **So the accurate headline is
narrow: this is a Qwen2.5 result, not a small-model result.**

**The mechanism is identified, which is what explains both the gain and its
absence.** The transform does two things — it relabels each piece with its age, and it re-sorts
the pieces into date order instead of leaving them in search-ranking order.
Running those separately shows the *sort* is what moves the numbers. It wins the
questions where "what is most recent?" is the answer, and loses the questions
where the search engine had already ranked the answer first. On Qwen that trade
comes out positive; on Gemma2 and Llama3.2 the two halves cancel.

**The original idea did not work.** This project was built around explicitly
tagging superseded facts `OUTDATED`. Those tags are correct 96.4% of the time
and moved accuracy by **−1.1 points** (p = 1.000) — one question in 91, in the
wrong direction. The models ignored them.

Everything below is the evidence, including the measurements that contradicted
what I expected and the bugs found along the way. Jump to [Results](#results) ·
[Why it does not transfer](#why-it-does-not-transfer) · [Limits](#limits) ·
[Running it](#running-it).

---

## Background

Skip this if you already work on LLM memory. Everything after it assumes the
terms defined here.

### The problem being solved in the first place

A chat assistant has no memory between sessions. The standard fix is
**retrieval**: store past conversations, and when a new question arrives, search
that store and paste the most relevant pieces into the prompt. Almost all
research attention goes to the search step — *did we find the right piece?*

This repo is about the step immediately after: the retrieved pieces have to be
turned into prompt text, and that is normally a one-line f-string that nobody
measures.

### The benchmark

**LongMemEval** is the standard test for this. Each question comes with roughly
50 past chat sessions — about 490 conversation turns — and asks one thing whose
answer is somewhere inside them. There are 500 questions in six types:

| type | count | what it asks for |
|---|---|---|
| `multi-session` | 133 | evidence combined from more than one session |
| `temporal-reasoning` | 133 | dates compared, or a duration computed |
| `knowledge-update` | 78 | a fact the user later revised — sometimes the current value, sometimes the previous one |
| `single-session-user` | 70 | something the user said, in one session |
| `single-session-assistant` | 56 | something the assistant said, in one session |
| `single-session-preference` | 30 | a stated preference, applied to a new request |

30 of the 500 are **abstention** questions, where the correct response is to say
the information is not present.

### Vocabulary used throughout

| term | meaning in this repo |
|---|---|
| **unit** | one conversation turn — the thing retrieval returns and the thing a transform operates on |
| **retriever** | the search step. `BM25` is keyword search; `hybrid` is BM25 plus embedding similarity |
| **k** | how many units the retriever returns. Always 10 here |
| **evidence** | a unit that LongMemEval itself labels `has_answer` — it contains what is needed to answer |
| **conditioner** | the transform this repo adds, applied to the k retrieved units after retrieval and before the prompt is assembled |
| **`identity`** | the control conditioner. Renders units exactly as memllm already does, byte for byte. Every other run is compared against it |
| **arm** | one complete run: one model × one conditioner × 100 questions, saved to `results/` with every prediction kept |
| **gold** | the benchmark's reference answer |
| **read tokens** | tokens spent answering one question, prompt plus completion. The cost measure used throughout |

### How the accuracy runs are set up

Every arm is a stratified 100-question sample of LongMemEval (seed 0). **91 of
the 100 are gradable** — the other 9 have free-form reference answers with no
checkable surface form, and are excluded rather than counted wrong. Grading is
deterministic string matching from memllm, whose published audit over 3,166
cases reports a false-accept rate of 0.0.

A conditioner is always compared against the *same model's own* `identity` arm
on the *same 91 question ids*: same retriever, same k, same model, same seed,
same prompt template. The only difference between the two runs is how the
retrieved units were written into the prompt. Significance is exact McNemar on
the questions where the two arms disagree, plus a paired bootstrap confidence
interval.

### Relationship to memllm

[memllm](https://github.com/i-shantt/memllm) is a companion repo that measured
what LLM memory *costs*. This repo imports its cost ledger, its audited grader
and its lift statistics, and compares against its stored baseline runs. That
reuse is deliberate: a number here is only worth something if it was produced
the same way memllm's were.

---

## The problem: retrieval succeeds and the answer is still wrong

Taking memllm's stored runs (Qwen2.5-7B, hybrid retrieval, k=10, 100 questions)
and joining per-question retrieval hits against per-question correctness — that
is, asking "when the evidence *was* retrieved, did the model get it right?"

There are two ways to ask that, and only one of them is fair. LongMemEval labels
every turn that is needed to answer a question, and some questions need several.
`any_hit@10` is satisfied when the retriever finds **one** of them, so a question
needing six evidence turns can score `any_hit = 1.0` while five sixths of the
answer is still missing. Getting that question wrong is not the model's fault.
`recall@10 = 1.0` is the honest condition: *every* labelled turn is in the
prompt, so nothing the model needs is absent.

| question type | wrong on `any_hit` | wrong on **full recall** | accuracy on full recall |
|---|---|---|---|
| knowledge-update | 8 / 16 | **6 / 13** | 0.538 |
| temporal-reasoning | 13 / 21 | **10 / 16** | 0.375 |
| multi-session | 15 / 22 | **9 / 16** | 0.438 |
| single-session-assistant | 3 / 10 | **3 / 10** | 0.700 |
| single-session-user | 2 / 12 | **2 / 12** | 0.833 |
| **all** | **41 / 81** | **30 / 67** | **0.552** |

The gap between the two columns is real and it is exactly the objection above:
11 of the 41 apparent failures had only part of their evidence retrieved, and
most of those are `multi-session`, the type that needs the most turns. Those are
retrieval failures wearing a reasoning failure's clothes.

**What survives the correction is still the point.** On 30 of 91 questions the
prompt contained every turn the benchmark says is needed and the model answered
wrong anyway — 45% of the questions it was fully equipped to answer. The two
single-session types, which need one turn and therefore cannot be short of
evidence, are unaffected by the correction at all.

**Search is not the binding constraint on this slice.** What the model does with
the text it was given is.

A concrete case, from those stored predictions:

```
Q     What brand of BBQ sauce am I currently obsessed with?
gold  Kansas City Masterpiece
pred  You are currently obsessed with Sweet Baby Ray's BBQ sauce.
```

Both the old preference and the newer one were in the prompt, and the model
quoted the outdated one — accurately, word for word. Each excerpt does carry its
date, but they arrive in search-relevance order, with nothing saying which is
most recent or how far back any of them is. Working that out is left to the
model, and here it did not.

In the vocabulary above, what this repo adds is a **read-time conditioner**: a
deterministic transform over the k retrieved units, run after retrieval and
before prompt assembly. No model call, no training, no work at write time. It is
a layer on top of a retriever, not a replacement for one.

```mermaid
flowchart LR
    A[Conversation<br/>archive] --> B[Search<br/>returns k=10 turns]
    B --> C[Conditioner<br/>sort by date · label age<br/>pre-compute arithmetic]
    C --> D[Model]
    D --> E[Answer]
    style C fill:#2d6a4f,color:#fff
```

## How this compares to published work

The phenomenon is **not** a discovery of mine. STALE (arXiv 2605.06527) names it
the *current-state adjudication gap*; the Always-On Agents survey (2606.30306)
repeats it. Both are worth reading first.

What is missing from that literature is the **cost of the fix**. STALE's own
remedy, CUPMem, takes a backbone from 8.7% to 68.0% by running an LLM
adjudicator on **every write** — a number of LLM calls proportional to the size
of the corpus — and reports no cost analysis and no control arms. TISER,
TimeRefine and TG-LLM all answer with test-time reasoning or fine-tuning. As far
as I can find, nobody has published the deterministic, zero-LLM, read-time
version, and nobody prices their own fix.

That gap is the reason this repo exists. Write-time consolidation pays LLM calls
proportional to the corpus whether or not anyone ever asks a question.
Conditioning pays no LLM calls at all, and touches only the k units that one
query actually retrieved. The price it does pay is 1–7% more read tokens, which
is reported in [Cost](#cost).

## The conditioners

| name | what it does | aimed at |
|---|---|---|
| `identity` | the control — memllm's existing rendering, byte for byte | — |
| `supersede:mark` | finds retrieved units that assert conflicting values for the same query term, then labels the one on the newest date `LATEST` and the earlier ones `OUTDATED` | knowledge-update |
| `supersede:order` | states the ordering within a topic but claims nothing about which value is current | a safer variant of `mark` |
| `supersede:drop` | deletes the superseded units outright | built to make the case *against* deletion |
| `supersede:mark:naive` | `mark` without the requirement that the units actually disagree; measured to show why that requirement is needed | — |
| `temporal` | sorts units into date order and labels each with how many days / weeks / months before the question it happened | temporal-reasoning |
| `temporal:norank` | the same labels with the sorting turned off — the control that separates the two halves | — |
| `all` | `supersede:mark` + `temporal` | — |
| `safe` | `supersede:order` + `temporal` | — |

### What that actually looks like

Three retrieved turns, for the question *"What time do I usually go to the
gym?"* asked on 2023/06/01. This is the real output of `memcond`, and the exact
text the model sees.

`identity` — memllm's existing rendering, and the control every arm is measured
against:

```
[2023/01/10 (Tue) 18:00] I go to the gym at 7:00 pm on weekdays.

[2023/02/02 (Thu) 10:00] Here are some tips for staying motivated.

[2023/05/03 (Wed) 20:00] I've moved my gym time to 6:00 pm.
```

`all` — reordered by date, each turn labelled with its position and its distance
from today, and the superseded gym time marked:

```
[2023/01/10 (Tue) 18:00 | OUTDATED mention of 'gym' -- superseded by 2023/05/03 (Wed) 20:00 | event 1 of 3 | 142 days / 20 weeks / 4 months before today] I go to the gym at 7:00 pm on weekdays.

[2023/02/02 (Thu) 10:00 | event 2 of 3 | 119 days / 17 weeks / 3 months before today] Here are some tips for staying motivated.

[2023/05/03 (Wed) 20:00 | LATEST mention of 'gym' (2 in total) | event 3 of 3 | 29 days / 4 weeks before today] I've moved my gym time to 6:00 pm.
```

(Those three lines are long; the block scrolls sideways rather than being
rewrapped, so what is shown is byte-for-byte what `memcond` emits.)

Same three turns, same words, nothing added from outside the archive. The date
always leads the prefix, exactly as `identity` writes it, so the two renderings
differ only by what was appended and by the ordering.

Every measurement in this repo is the difference between those two prompts.

### Why `temporal` exists

Models mishandle dated context in two distinct ways. Both examples below are
from memllm's `oracle` arm, where every evidence turn is in the prompt by
construction, so nothing is being blamed on retrieval:

```
1.5B  "How many months have passed since I last visited a museum with a friend?"
      gold 5, said "One month" -- one month back is the February visit, which
      was with a parent; the visit with a friend is the October one, five back

14B   "How many weeks passed between the time I sold homemade baked goods at
       the Farmers' Market for the last time and the time I participated in
       the Spring Fling Market?"                                  gold 3 weeks
      "The last time you sold homemade baked goods was on 2023/02/26, and you
       participated in the Spring Fling Market on 2023/03/21. There are 4 weeks
       between these two dates."
```

The first is **anchoring**: ten dated turns arrive in an order that says nothing
about time — evidence first and then filler in the `oracle` arm, search rank in
the real ones — and the model computes from whichever one it settled on. This is
the common case.

The second is **arithmetic**: both correct dates are quoted straight off the
page and the subtraction over them is still wrong. That is 14B, the largest
model memllm ran.

Sorting into date order addresses the first. Pre-computed distances address the
second — subtraction is free and exact in Python, and doing it at render time
turns "compare two dates" into "read a number".

## Checking the rules before spending any GPU

`scripts/run_mechanical_gate.py` scores a conditioner's *decisions* against
LongMemEval's own `has_answer` labels. No model, no GPU, no quota — it runs on a
laptop in seconds. The point is to catch a broken rule before buying GPU time.

Reading the table below:

| column | meaning |
|---|---|
| `fired` | fraction of questions where the rule annotated anything at all |
| `HARM` | the **newest** evidence unit was marked `OUTDATED` — the rule telling the model that the current answer is stale. Want ≈ 0 |
| `PREC` | the newest evidence unit was marked `LATEST` — the rule getting it right. Want high |
| `n` | questions where the rule annotated the newest evidence unit at all. `HARM` and `PREC` are rates over this, so a small `n` means a thin result |
| `older` | older evidence units marked `OUTDATED`. On knowledge-update these genuinely are superseded, so high is correct |
| `surv` | fraction of evidence units not deleted. Must be 1.000 for any `mark` mode |
| `tok Δ` | change in prompt tokens against `identity` |
| `llm` | LLM calls made. Zero by construction, and reported to prove it |

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

`temporal` scores `--` because it never marks anything `LATEST` or `OUTDATED` —
it only sorts and labels ages, so there is no currency claim to be right or
wrong about.

The supersession rule is accurate exactly where the thing it models exists, and
noisy everywhere else: on knowledge-update it marks the right unit current
92–96% of the time, while across all question types precision falls to 0.69–0.79
because it fires where there is no supersession to find.

`supersede:drop` deletes **22% of all evidence** on knowledge-update (`surv`
0.781). That is the concrete argument against write-time deletion — which is
what Mem0's UPDATE and DELETE operations do irreversibly — measured rather than
asserted.

The gate has a real limit, and it is the most important thing on this page: it
certified a conditioner that turned out to do nothing at all. See
[the annotation nobody read](#the-annotation-nobody-read).

### Two bugs the gate caught

Both looked like the idea failing. Both were bugs in my code.

**Slot fragmentation.** Units were grouped by the *rarest* query term they
contained, on the theory that a specific word identifies the units in contention
better than a generic one. It does the opposite. For "What time do I usually go
to the gym?", `I go to the gym at 7pm` groups under "gym" while `I've moved my
gym time to 6pm` groups under "time" — so the two mentions that actually
conflict landed in different groups and neither was annotated. Grouping needs
the *shared* term.

**Same-date siblings.** Ordering inside a group broke ties by turn index, so an
assistant's follow-up outranked the user turn that stated the fact. The gate
reported the newest evidence unit marked `OUTDATED` on **93%** of
knowledge-update questions. Supersession is a claim about dates, and same-day
siblings supersede nothing. Fixing it moved HARM from 0.933 to 0.079.

A third correction was to the gate itself. Its first harm metric counted *any*
evidence unit marked `OUTDATED` and reported 0.681, which looked disqualifying.
The metric was wrong: 67 of the 78 knowledge-update questions carry **two**
evidence turns on two different dates — the superseded fact and its replacement
— because answering needs both. Marking the earlier one `OUTDATED` is correct
behaviour, and the metric was penalising it.

## Results

Qwen2.5, Gemma2 and Llama3.2 run locally through Ollama. BM25 at k=10, 100
questions, `max_new_tokens=256`, deterministic grading, paired against each
model's own `identity` arm as described in [Background](#how-the-accuracy-runs-are-set-up).

| model | `identity` baseline | `all` Δ | `temporal` Δ |
|---|---|---|---|
| Qwen2.5 1.5B | 0.2747 | **+0.1319** (p=0.002) | **+0.1099** (p=0.006) |
| Qwen2.5 3B | 0.3077 | +0.0879 (p=0.057) | +0.0549 (p=0.302) |
| Qwen2.5 7B | 0.4176 | +0.0440 (p=0.424) | +0.0110 (p=1.000) |
| Gemma2 2B | 0.3297 | −0.0110 (p=1.000) | +0.0110 (p=1.000) |
| Llama3.2 3B | 0.4286 | −0.0549 (p=0.180) | −0.0659 (p=0.109) |

**The bottom two rows are the important ones.** They are the two non-Qwen
families, and they are where the headline stops being true. The next section is
about them.

`temporal:norank` — the same labels with the date sorting switched off — was run
on those two models as well: Gemma2 −0.0110, Llama3.2 +0.0110. Both null, and
[why that matters](#what-survives-the-sort-is-the-active-ingredient) is the most
useful result in this repo.

## Why it does not transfer

Within Qwen2.5 the effect is large at 1.5B and decays cleanly with model size —
12, then 8, then 4 net questions fixed out of 91. That pattern looked like a
capacity story: the transform does ordering and arithmetic that a small model
cannot do for itself, so the smaller the model, the more it gains.

**On Gemma2 2B and Llama3.2 3B, the effect is gone.** Gemma2 is a flat null.
Llama3.2 trends *negative* on both arms, and `temporal` alone at −0.0659
(p = 0.109) is closer to a real regression than to nothing.

So the claim has to be narrower than the Qwen gradient suggests: **this is a
Qwen2.5 result, not a small-model result.** A dose-response curve inside one
model family is not evidence about models in general. Running two other families
is what turned that from an assumption into a finding.

### What survives: the sort is the active ingredient

The per-type breakdown says more than the totals do, because the conditioner
does two things at once: it labels each unit with its age, and it re-sorts the
units by date instead of by search rank. `temporal:norank` runs the labelling
with the sorting switched off, which separates the two.

| Δ by question type | n | Gemma `temporal` | Gemma `norank` | Llama `temporal` | Llama `norank` |
|---|---|---|---|---|---|
| knowledge-update | 16 | **+0.188** | **−0.125** | **+0.062** | +0.000 |
| single-session-user | 14 | **−0.143** | **+0.071** | **−0.214** | +0.000 |
| temporal-reasoning | 25 | −0.080 | −0.040 | −0.040 | +0.040 |
| multi-session | 26 | +0.000 | +0.038 | −0.077 | +0.000 |
| single-session-assistant | 10 | +0.200 | +0.000 | −0.100 | +0.000 |

Look at the top two rows. Turning the sort off flips the sign of both, in both
models, in opposite directions:

- **With the sort:** `knowledge-update` gains, `single-session-user` loses.
- **Without it:** `knowledge-update` loses, `single-session-user` recovers.

That is one mechanism, not two. **The date sort trades away the questions where
search rank was already the answer, in order to win the questions where recency
is the answer.** `single-session-user` is a single turn that BM25 puts first —
re-ordering buries it. `knowledge-update` is a fact that was later revised —
ordering by date is exactly what surfaces the revision.

On Qwen2.5 that trade comes out favourable and the net is positive. On Gemma2
and Llama3.2 the two effects cancel, which is why both models sit at a null
overall no matter how the conditioner is configured.

My original hypothesis — labelling good, sorting bad — was **wrong, and
usefully so**. The sort is not a defect to be removed. It is the active
ingredient, doing exactly what it was designed to do, while also doing damage
elsewhere that nobody was measuring.

**These per-type numbers deserve caution.** The slices are n=14 and n=16, so one
question is worth ±0.07 and no individual cell is significant. What carries
weight is that four independent sign flips all landed where the mechanism
predicts — not the size of any one of them.

### The obvious next step, and why I did not build it

If the sort helps recency questions and hurts rank questions, then sorting
*conditionally* — only when the question is asking about a current state —
should capture the gain without the loss. Cheap, deterministic, and the natural
next move.

It does not survive contact with the data. `scripts/test_sort_router.py` tests
the premise against every arm already run, with no GPU, because every arm stores
its predictions and "would a router have helped?" is answerable from those.

The premise fails twice.

**The question types do not split on tense.** `knowledge-update` frequently asks
for the *superseded* value — "What was my **previous** personal best time?", "my
**former** manager Rachel", "the **earlier** fishing trip" — while
`single-session-user` contains "What book am I **currently** reading?". The tense
markers cut across the question types instead of separating them.

**And the buckets carry no signal.** Grouping all 500 questions by tense marker
and measuring the sort's effect within each:

| bucket | n (pooled over 5 models) | `identity` | `temporal` | Δ |
|---|---|---|---|---|
| CURRENT | 40 | 0.500 | 0.550 | +0.050 |
| PAST | 65 | 0.523 | 0.600 | **+0.077** |
| NEITHER | 345 | 0.307 | 0.319 | +0.012 |

The largest gain is in PAST, which is backwards from the hypothesis. Per model
the CURRENT bucket is only **n=8** — one question is worth 0.125 — and four of
the five models show exactly 0.000 there. A router built on this would be
fitting noise.

So it was not built. The script is committed so that the next person can see the
negative result instead of re-deriving it on a GPU.

### Does conditioning a small model buy a bigger one?

Conditioned 1.5B scores 0.4066. The unconditioned 7B baseline scores 0.4176.
The tempting sentence is "conditioning makes a 1.5B match a model 4.7× its
size", and memllm explicitly warns against writing it: the grader marks an
answer correct when the gold span appears anywhere in it, so a more verbose
model gets more chances, and 7B's median answer is 27 words against 1.5B's 16.

`scripts/cross_model_check.py` tests it two ways, offline, from the stored
predictions:

**Paired, not eyeballed.** Every arm answers the same 91 question ids, so exact
McNemar applies across models the same way it does within one.

| comparison | accuracy | Δ | discordant | p |
|---|---|---|---|---|
| conditioned 1.5B vs **unconditioned 7B** | 0.4066 vs 0.4176 | −0.0110 | 9 / 10 | **1.000** |
| conditioned 1.5B vs unconditioned 3B | 0.4066 vs 0.3077 | +0.0989 | 16 / 7 | 0.093 |

Against 7B the discordant pairs are 9 against 10 — as close to a coin flip as
91 questions can produce. The test cannot separate them.

**Length-controlled.** Re-grading the same stored answers, each truncated to its
first N words. An advantage that survives the cap is accuracy; one that
evaporates was verbosity.

| arm | median words | full | 40w | 25w | 15w | 8w |
|---|---|---|---|---|---|---|
| conditioned 1.5B | 16 | 0.407 | 0.407 | 0.363 | 0.330 | 0.209 |
| unconditioned 3B | 18 | 0.308 | 0.297 | 0.275 | 0.275 | 0.187 |
| unconditioned 7B | 27 | 0.418 | 0.407 | 0.374 | 0.352 | 0.231 |
| **gap to 7B** | | −0.011 | +0.000 | −0.011 | −0.022 | −0.022 |
| **gap to 3B** | | +0.099 | +0.110 | +0.088 | +0.055 | +0.022 |

**Against 7B the gap never exceeds two questions at any cap.** It does not open
up as truncation removes 7B's verbosity advantage, which is what would happen if
the tie were an artefact of answer length. So the defensible claim is: *a
conditioned 1.5B is not distinguishable from an unconditioned 7B on this
benchmark* — not that it beats one.

**Against 3B the lead is real but mostly length.** It holds its sign at every
cap, so it is not purely an artefact, but it decays from +0.099 to +0.022 as
answers are truncated. Roughly three quarters of that lead was verbosity. Quote
the capped number, not the headline one.

### The arms reproduce exactly

Re-running the two `identity` baselines in a fresh Kaggle session, on different
allocated hardware, produced **100/100 byte-identical predictions** on both
models. Generation is greedy (`temperature: 0.0`), and this confirms it holds in
practice and not only in principle. Every paired comparison here therefore
compares two runs that differ only in the thing under test.

### The annotation nobody read

`supersede:mark` — the feature this project was built around — did **nothing**:
−0.0110 at both 1.5B and 7B.

This is where the CPU gate failed, which makes it the most instructive result
here. The gate certified this conditioner: on knowledge-update it marked the
newest evidence unit `LATEST` **96.4%** of the time and mislabelled it 3.6%. The
annotations were right. The models ignored them.

The gate measured whether the labels were *correct*. It could not measure
whether they were *used*, and those turned out to be different questions. Any
mechanical-correctness harness has this blind spot, and no amount of raising the
precision bar would have caught it.

Worse for the original hypothesis, the knowledge-update gain that supersession
was designed to deliver was delivered by `temporal` instead — **+0.250 at
1.5B**, against `supersede:mark`'s +0.000 on the same slice:

| Δ by question type, 1.5B | `temporal` | `supersede:mark` | `all` |
|---|---|---|---|
| knowledge-update | **+0.250** | +0.000 | **+0.250** |
| temporal-reasoning | +0.120 | +0.000 | +0.200 |
| multi-session | +0.077 | +0.038 | +0.077 |
| single-session-assistant | +0.100 | −0.100 | +0.100 |
| single-session-user | +0.000 | −0.071 | +0.000 |

`temporal` makes no claim about currency at all. It sorts by date and labels
each unit with its age. That is apparently enough: **the model does not need to
be told which fact is current, it needs the chronology made legible enough to
work it out.** Being told directly, in a label that is right 96% of the time,
changed nothing.

### A correction found by reading individual flips

The first version of these numbers had `temporal` at +0.1209. Checking *which*
twelve questions it fixed turned up one that was not a conditioning effect at
all: both arms refused an abstention question, but `identity` wrote "the
excerpts provided **do** not contain information" and the grader's refusal
patterns only matched "**does** not contain". One arm's refusal was detected and
the other's was not, which is a fake +1.

Fixed in memllm and re-graded with `scripts/regrade.py --pattern 'cond_*.json'`,
which is why the numbers above are slightly smaller than a first reading would
have given. Aggregate accuracy would never have shown this. Reading the
individual flips did.

### Deletion

`supersede:drop` came out at −0.0110, not significant, for 8.3% fewer read
tokens. The CPU gate showed it deleting 22% of all evidence on knowledge-update,
so a sharp loss was the prediction; it merely failed to help. The accurate
reading is weaker than "deletion is harmful": deletion bought nothing, while
costing an irreversible loss of information that a past-directed question would
need.

### Cost

Every number above was produced with **zero LLM calls** and **1–7% more read
tokens** — 1.0% for `supersede:mark`, 5.3–5.8% for `temporal`, 6.2–6.9% for
`all`. `supersede:drop` is the only one that saves tokens, at −8.3%, and it is
the one that does not work.

STALE's CUPMem reaches its numbers with an LLM adjudicator on every write, a
number of calls proportional to the corpus, paid whether or not a query ever
arrives. That comparison is the one the literature does not report.

## Layout

```
memcond/conditioner/   base.py, supersede.py, temporal.py — the transforms
memcond/eval/          mechanical.py — the CPU gate's metrics
scripts/               run_mechanical_gate.py  — the CPU gate
                       run_conditioned_eval.py — one GPU arm
                       compare_conditioners.py — paired stats against identity
                       cross_model_check.py    — is a conditioned 1.5B a 7B?
                       test_sort_router.py     — the router negative result
results/               every arm as JSON, per-question predictions included
kaggle/                the notebook cells the GPU arms were run from
tests/                 73 tests, no model and no network; the five that drive
                       the eval script end to end skip without the benchmark
.github/workflows/     CI: the suite, plus a check that the committed tables
                       still regenerate byte-identically
```

## Running it

Needs a [memllm](https://github.com/i-shantt/memllm) checkout beside this one,
or `MEMLLM_PATH` set. It supplies the cost ledger, the audited grader and the
lift statistics.

```bash
git clone https://github.com/i-shantt/memllm.git
git clone https://github.com/i-shantt/llm-memory-conditioning.git
cd llm-memory-conditioning

pip install -r requirements.txt   # four small packages: no torch, no GPU
python -m pytest tests/ -q        # 68 pass, 5 skip without the benchmark
```

**The analysis needs nothing beyond that.** Every table above except the
accuracy runs themselves comes back from the stored arms in seconds:

```bash
python scripts/compare_conditioners.py   # the Results table and the per-type splits
python scripts/cross_model_check.py      # the 1.5B-vs-7B tables, both checks
python scripts/test_sort_router.py       # the router negative result
```

`results/conditioner_comparison.json` regenerates byte-identically, and CI
re-checks that on every push.

**The CPU gate needs the benchmark**, which is a 265 MB download and is not
vendored here:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('xiaowu0162/longmemeval', 'longmemeval_s', \
repo_type='dataset', local_dir='../memllm/data/raw')"

python scripts/run_mechanical_gate.py --limit 100 --retriever bm25
```

**Re-running the accuracy arms needs a GPU** on top of the benchmark, plus
memllm's own requirements and a local Ollama. `kaggle/` has the notebook cells
they were produced from.

## Limits

- **Mechanical correctness does not imply usefulness, and the gate cannot tell
  the difference.** `supersede:mark` passed at 96.4% precision and moved
  accuracy by −0.011. Treat every gate number as necessary, never sufficient.
- **The headline result is specific to one model family.** Within Qwen2.5 the
  gradient is clean; on Gemma2 2B and Llama3.2 3B it disappears. Only Qwen 1.5B
  clears significance outright — 3B is suggestive with the two tests disagreeing
  at the margin, 7B is null, and both non-Qwen models are null-to-negative.
  Nothing here supports a claim about small models in general.
- **100 questions gives a confidence interval of roughly ±0.08 per arm**, so no
  single model settles anything. The Qwen gradient, and the consistent
  `knowledge-update` sign across all five models, are the load-bearing evidence;
  individual arms are not.
- **The sort / no-sort split is measured on two models and five question-type
  slices of n=10–26.** Four sign flips landing where the mechanism predicts is
  suggestive; none of the individual cells is significant, and the whole pattern
  rests on roughly a dozen questions changing hands.
- **The `temporal:norank` comparison was designed after seeing which question
  type broke.** That is post-hoc, and post-hoc hypotheses find patterns in
  noise. It was pre-registered only in the weak sense that the flag already
  existed in the code before the transfer run.
- **The cross-model comparison is the most fragile thing here**, and
  [it is checked two ways](#does-conditioning-a-small-model-buy-a-bigger-one)
  rather than asserted. It says conditioned 1.5B is *indistinguishable* from the
  7B baseline, which is a weaker statement than beating it — and these arms were
  not designed for a cross-model test in the first place.
- **Away from knowledge-update the supersession rule is noisy.** Across all 500
  questions HARM is 0.209–0.312 — it fires where there is no supersession to
  find. That is why `supersede:order`, which claims no currency at all, exists
  as a fallback.
- **`fired` is not `n`.** `supersede:mark` fires on 53.8% of knowledge-update
  questions but annotates the *newest evidence unit* on 28 of 78, which is what
  HARM and PREC are rates over. The rest of the time it annotated other units.
- **Everything here uses BM25**, not a hybrid retriever. That started as a
  workaround — sentence-transformers crashed on the Kaggle GPU with "no kernel
  image is available for execution on the device" — but it is the better choice
  anyway: BM25 also scores `any_hit@10 = 1.000` on knowledge-update, so the
  evidence is equally present, and the gate and the accuracy arms now describe
  the same retrieval rather than two different ones. Whether these conditioners
  behave the same over a dense retriever's distractors is untested.
- **`is_evidence` is LongMemEval's own labelling** and inherits whatever errors
  it contains.
- **Conditioning cannot fix a retrieval miss.** It only changes the presentation
  of what was already retrieved, so its ceiling is the 30-of-91 full-recall
  slice above — and on the 11 partial-recall questions, no amount of
  reformatting can supply a turn the retriever never returned.

## License

[MIT](LICENSE). LongMemEval is a separate dataset under its own terms and is
not redistributed here.
