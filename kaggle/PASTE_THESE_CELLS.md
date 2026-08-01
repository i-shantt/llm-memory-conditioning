# Kaggle cells — conditioner arms

Pushed and polled by memllm's `kaggle_mcp` server, which builds a notebook from
the `## Cell ...` headings below. Cells are always emitted in document order.

The experiment: identical retriever, k, model, seed and prompt template, with
the *only* difference being how the same retrieved units are rendered. Every
conditioned arm is paired against the `identity` arm on the same question ids.

Budget ~2–3 h of GPU. Cell 4 is resumable — an arm whose results file already
exists is skipped, so a re-push after a timeout continues rather than restarts.

## Cell 0 — run configuration

```python
# Ordered smallest-first on purpose. If the session dies partway, a complete
# 1.5B sweep is worth far more than two half-finished ones: the whole claim is
# a within-model paired comparison, so an orphaned arm proves nothing.
SIZES = ["1.5b", "7b"]
N = "100"           # stratified subset size
CTX = "8192"        # ollama context window
MAXNEW = "256"      # NOT memllm's 64 -- see Cell 4a
```

## Cell 1 — Ollama and the model ladder (~12 min)

```python
import subprocess, time, urllib.request, os, shutil

os.environ["OLLAMA_MODELS"] = "/kaggle/working/.ollama/models"
os.environ["OLLAMA_SCHED_SPREAD"] = "1"
os.makedirs(os.environ["OLLAMA_MODELS"], exist_ok=True)

print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)

# The Ollama installer unpacks a zstd tarball and the Kaggle image ships without
# zstd, so the install dies ~10s in. Installing it first is the whole fix.
subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
               shell=True, check=False)
assert shutil.which("zstd"), "zstd unavailable; the ollama installer will fail"

subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
subprocess.Popen(["ollama", "serve"], env=os.environ,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for i in range(90):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print(f"ollama up after {i}s"); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("ollama did not start -- is Internet enabled in the sidebar?")

LADDER = [f"qwen2.5:{s}-instruct" for s in SIZES]
for m in LADDER:
    print(f"pulling {m} ...", flush=True)
    subprocess.run(["ollama", "pull", m], check=True)

subprocess.run(["ollama", "run", LADDER[-1], "hi"],
               capture_output=True, text=True, timeout=600)
print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)
```

**Read `ollama ps`.** Anything other than `100% GPU` for the largest rung means
it did not fit and every timing below is wrong.

## Cell 2 — both repos, data, and pre-flight (~4 min)

```python
import subprocess, os, json, sys

for repo in ("memllm", "llm-memory-conditioning"):
    subprocess.run(["git", "clone", "--depth", "1",
                    f"https://github.com/i-shantt/{repo}.git",
                    f"/kaggle/working/{repo}"], check=True)
os.chdir("/kaggle/working/llm-memory-conditioning")
os.environ["MEMLLM_PATH"] = "/kaggle/working/memllm"
subprocess.run(["pip", "install", "-q", "rank_bm25", "sentence-transformers",
                "tiktoken"], check=True)

# Fail in four minutes rather than after three hours. Each of these has cost a
# run before: a missing flag, a stale clone, an absent split.
help_text = subprocess.run(["python", "scripts/run_conditioned_eval.py", "--help"],
                           capture_output=True, text=True).stdout
for flag in ["--conditioner", "--max-new-tokens", "--num-ctx", "--out-dir"]:
    assert flag in help_text, f"{flag} missing -- the memcond clone is stale"

from huggingface_hub import hf_hub_download
for split in ("longmemeval_s", "longmemeval_oracle"):
    hf_hub_download("xiaowu0162/longmemeval", split, repo_type="dataset",
                    local_dir="/kaggle/working/memllm/data/raw")

# memcond's own suite, then memllm's -- the grader is shared, and every
# accuracy number below depends on its false-accept rate being zero.
for cwd, cmd in ((".", ["python", "-m", "pytest", "tests/", "-q"]),
                 ("/kaggle/working/memllm", ["python", "tests/test_harness.py"]),
                 ("/kaggle/working/memllm", ["python", "scripts/audit_graders.py"])):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    print(r.stdout[-2000:], r.stderr[-1200:])
    assert r.returncode == 0, f"{cmd} failed -- fix before spending GPU"

audit = json.load(open("/kaggle/working/memllm/results/grader_audit.json")
                  )["graders"]["deterministic"]
assert audit["false_accept_rate"] == 0.0, "grader accepts known-wrong answers"
print("pre-flight ok | grader false-accept:", audit["false_accept_rate"])
```

## Cell 3 — the CPU gate, no GPU contention (~6 min)

```python
# Re-runs the mechanical gate on the machine that will produce the accuracy
# numbers, so the two are known to come from the same code. Costs no GPU.
import subprocess
r = subprocess.run(["python", "scripts/run_mechanical_gate.py",
                    "--limit", N, "--retriever", "bm25"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1000:])
assert r.returncode == 0
```

## Cell 4a — the `arm()` helper (instant)

```python
import subprocess, os, time, json

# 256, not memllm's 64. Reading stored predictions showed 7B answers cut off
# mid-subtraction on temporal questions ("...have been taking classes fo"),
# which understates precisely the slice the temporal conditioner targets. The
# baseline arm is re-run at the same cap, so the comparison stays paired.
def arm(size, conditioner, k="10", retriever="hybrid"):
    tag = f"cond_{size}_{conditioner.replace(':', '-')}_k{k}_n{N}"
    out = f"/kaggle/working/llm-memory-conditioning/results/{tag}.json"
    if os.path.exists(out):
        print(f"skip {tag} (already done)"); return
    t0 = time.time()
    r = subprocess.run(
        ["python", "scripts/run_conditioned_eval.py",
         "--data", "/kaggle/working/memllm/data/raw/longmemeval_s",
         "--limit", N, "--retriever", retriever, "--conditioner", conditioner,
         "--k", k, "--answer-backend", f"ollama:qwen2.5:{size}-instruct",
         "--num-ctx", CTX, "--max-new-tokens", MAXNEW, "--tag", tag],
        capture_output=True, text=True)
    print(r.stdout[-1200:], r.stderr[-800:])
    print(f"--- {tag}: {time.time()-t0:.0f}s\n", flush=True)
    assert r.returncode == 0, f"{tag} failed"
```

## Cell 4 — conditioner arms (~2–2.5 h, resumable)

```python
# identity FIRST for each size. It is the baseline every other arm is paired
# against, so a size with only conditioned arms and no baseline is unusable.
ARMS = ["identity", "supersede:mark", "temporal", "all"]

for size in SIZES:
    for c in ARMS:
        arm(size, c)
```

## Cell 4b — the deletion arm, 7B only (~12 min)

```python
# Predicted to LOSE, and run anyway. The CPU gate measured supersede:drop
# deleting 22% of all evidence on knowledge-update, because LongMemEval asks
# past-directed questions ("Before I purchased the gravel bike, ...") whose
# answer lives in the superseded turn. This is the measured argument against
# write-time deletion, which is what Mem0's UPDATE/DELETE do irreversibly.
if "7b" in SIZES:
    arm("7b", "supersede:drop")
```

## Cell 4c — paired comparison (~1 min)

```python
import subprocess
r = subprocess.run(["python", "scripts/compare_conditioners.py"],
                   capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-2000:])
```

## Cell 5 — summary, truncation audit, and download

```python
import json, glob, shutil, os

print(f"{'arm':44s} {'acc':>7s} {'graded':>7s} {'tok/q':>7s} {'cap':>4s} {'maxtok':>7s}")
for f in sorted(glob.glob("/kaggle/working/llm-memory-conditioning/results/cond_*.json")):
    p = json.load(open(f))
    pt = {r["prompt_tokens"] for r in p["records"]}
    flag = "  <-- CLAMPED" if len(pt) == 1 and len(p["records"]) > 5 else ""
    print(f"{os.path.basename(f)[:-5]:44s} {p['accuracy']:7.4f} "
          f"{p['n_graded']:7d} {p['read_tokens_per_query']:7.0f} "
          f"{p['n_hit_token_cap']:4d} {p['prompt_tokens_max']:7d}{flag}")

# The cap audit that justified raising max_new_tokens. If these are still
# nonzero, 256 was not enough either and the temporal numbers stay suspect.
total_cap = sum(json.load(open(f))["n_hit_token_cap"]
                for f in glob.glob("/kaggle/working/llm-memory-conditioning/results/cond_*.json"))
print(f"\nanswers hitting the {MAXNEW}-token cap across all arms: {total_cap}")

for f in glob.glob("/kaggle/working/llm-memory-conditioning/results/*.json"):
    shutil.copy(f, "/kaggle/working/")
print("\nresults copied to /kaggle/working for download")
```
