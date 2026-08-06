---
name: physical-ai-deft
description: Run NVIDIA's Physical AI Data Factory (Cosmos AnomalyGen) as a bounded enrichment loop on any ClearML Enterprise server — register real defect data as a HyperDataset, read the rare-class gap from metadata, generate against it, gate on nn_score, publish each round as an immutable version, and train a model per round with lineage.
---

# Run the Physical AI Data Factory on a ClearML server

You are setting this lab up on a ClearML Enterprise server that is **not** the
one it was built on. Everything you need is in this folder; nothing is specific
to the original host except the values in "What you must be told" below.

Read this file top to bottom before running anything. Most of it is failures
that already happened once — each one cost a GPU hour and looked like a
different problem than it was.

---

## What this actually does

```
register the real data          177 frames -> HyperDataset v1-real
  |
  v
read the gap                    label counts, server-side, ~1s, no pixels move
  |
  v                             for each of N rounds:
generate                        NVIDIA AnomalyGen -- masks placed from the board's CAD,
  |                             defect inpainted where the mask says
  v
score                           nn_score against the REAL examples of that class
  |
  v
publish                         survivors -> the next immutable version, parented on the last
  |
  v
train                           an inspector on that version, registered against it
```

The output is N+1 dataset versions and N+1 registered models (N rounds plus a
real-only baseline), each model carrying the version id it trained on.

---

## What you must be told (ask if you weren't)

| Thing | Why | If missing |
|---|---|---|
| ClearML credentials for the target server | Everything | **Blocking.** Ask. |
| A queue backed by a **whole** GPU, ≥22 GB | Generation peaks ~16.5 GiB. A fraction will OOM. | **Blocking.** Ask which queue. |
| The GPU driver version on that queue | NVIDIA's images are CUDA 12.8/13. Driver must be **≥ 535**; 580 is what this was proven on. | Run the probe (below) before anything else. |
| `NGC_API_KEY` + `HF_TOKEN` **in the task pod's environment** | The container and the weights are gated. | See "Credentials" — do not pass them as task parameters. |
| A shared writable path mounted on every task pod | NVIDIA's checkpoints are ~22 GB; without it every stage re-downloads them. | Works without it, just slowly. Set `DEFT_CACHE`. |
| Whether the server has **HyperDatasets** | `datasets.*` / `frames.*` are Enterprise-only. | **Blocking** — the whole gap-reading premise depends on them. |

Assume PVCs, secrets and queues already exist on the target host. Your job is
the pipeline, not the cluster.

### Pod requirements NVIDIA's `docker run` implies

Their reference invocation is
`docker run --gpus all --ipc=host --shm-size=16g -v $WS:$WS --user ...`.
On Kubernetes those become, **once per lab, on the agent's pod template**:

| `docker run` | Pod template |
|---|---|
| `--gpus all` | the queue profile's whole-GPU request |
| `--ipc=host --shm-size=16g` | a memory-backed `emptyDir` at `/dev/shm` (kubelet default is 64 MB and does **not** fail loudly — it breaks dataloader workers in ways that look like model bugs) |
| `-v $WS:$WS` | the shared cache PVC, mounted at the same absolute path |
| uid 10000 | `securityContext.fsGroup: 10000` — AnomalyGen's image runs as uid 10000 and writes into mounted dirs |

`probe_parity.py` checks all four **from inside a real task pod** and names
which one is missing. Run it first. It is cheap and it will save you an hour.

---

## Run it

```bash
python probe_parity.py            # verify the pod matches what NVIDIA's container expects
python launch.py register         # 177 frames -> HyperDataset v1-real
python launch.py rounds --rounds 3
```

`launch.py` does not do the work — it creates a task pointing at this repo at a
commit and enqueues it. The GPU work happens on the agent.

Verify after each step rather than at the end:

| After | Check | Expect |
|---|---|---|
| `probe_parity.py` | its printed table | GPU present, `/dev/shm` ≥ 16 G, cache path writable, both credentials present |
| `register` | the HyperDataset in the UI | **177 frames, 100% annotated**; label counts `missing 62 / excess_solder 16 / bridge 8 / clean 5 / mask 86` |
| generation (inside `rounds`) | the console | `Total: 24 samples (0 text2roi, 24 cad2roi, 0 free)` — see trap 4 if you see `free` |
| scoring | the console | three `nn_score` lines, n=8 each, medians ~0.65 |
| publish | the version list | a NEW version each round; counts **grew** |
| train | the models list | one model per round + `inspector-baseline` |

---

## Credentials

`NGC_API_KEY` and `HF_TOKEN` go into the task pod **from a Kubernetes Secret**,
never as task parameters or `docker_args` values you type. Task parameters are
readable by anyone with project access, and `docker_args` lands on the task
record in plaintext — which is a credential leak that survives the run.

If the target host has no such Secret, that is a cluster task for whoever owns
it. Do not work around it by putting the key on the task.

---

## The traps

Each of these happened. Each cost real time. In rough order of how much.

### 1. The agent builds a venv and shadows NVIDIA's torch

NVIDIA's image already carries torch 2.10+cu128 and every AnomalyGen
dependency. Left to itself, clearml-agent creates a fresh virtualenv and
installs a *different* torch into it. Symptoms range from CUDA version
mismatches to missing custom ops.

```
-e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
```

**But** that also means `packages=` is never installed, so `clearml` itself is
not present and the task dies on `import clearml`. Pip-install it in the docker
setup script instead. `launch.py` does both; if you write your own launcher,
you need both or neither works.

### 2. `from __future__ import annotations` breaks entry points

clearml-agent patches the top of the executed script to inject its own
bootstrap. If the file starts with a `__future__` import, the patched file is a
`SyntaxError` — Python requires `__future__` imports to come first.

Remove it from **entry-point** files only. Modules that are merely imported are
fine.

### 3. Checkpoints are downloaded flat, and read nested

NVIDIA's downloader writes checkpoints flat into the target directory. Their
`run_sdg.sh` reads them from `<dir>/checkpoints/model/`. Nothing warns you; the
stage fails later looking like a missing-weights bug.

`ag_common.link_checkpoints()` creates the symlinks. **Call it in every stage
that touches checkpoints**, not just the first — this was rediscovered three
times because a later stage ran in a fresh pod with the same PVC.

### 4. `spatial_dependency` decides where the defect lands

Automatic Mask Placement routes on the defect spec:

- `cad` — region from the board's own CAD mask + `semantic_segmentation_labels.json`
- `text` — you describe the location; Qwen-VL + SAM2 segment it
- `free` — anywhere

NVIDIA's PCB dataset is `cad`, and the console should say `24 cad2roi`. If you
see `free`, the CAD mask or the segmentation labels did not reach the container
and the defects are landing in arbitrary places — the generated data is
worthless and nothing will tell you so.

### 5. Publishing the same run twice silently doubles the counts

A version inherits its parent's frames. Publish the same output directory twice
and those images are in the dataset twice. Nothing errors — the counts the loop
reasons over quietly inflate, which is the one thing a metadata-driven loop
cannot tolerate. Observed live: v2 held 24 synthetic frames, v3 inherited them
and added the same 24 again.

The guard is **run-scoped**, on the run id embedded in the upload path.
Do not "fix" it to match on basename: generated filenames are deterministic
(`<anomaly_type>_<NNNNN>.png`), so every round produces the same names for
different images. A basename guard blocks every legitimate round after the
first and stalls the loop silently.

### 6. Three separate reasons rounds fail to accumulate

All three produce the same symptom — the training set never grows, accuracy is
flat, and it reads as "synthetic data changed nothing":

1. **Shared output dir** — every round wrote to the same path and overwrote the
   last. Fixed by a per-run output dir.
2. **Non-accumulating collect** — the trainer read only the latest run dir
   instead of all of them.
3. **Colliding run ids** — `round1/2/3` collides the moment the loop runs a
   *second* time, so this invocation's round 2 overwrites the previous
   invocation's. Run ids are scoped to the invocation (`<task-id-prefix>-roundN`).

If accuracy is suspiciously flat, check the *training set size per round* before
you believe anything about the data.

### 7. The holdout must be real-only, fixed, and it must have a control

Two bugs, both of which produce a meaningless number that looks fine:

- A holdout that **grows each round** and **contains synthetic frames** is
  measuring the generator against itself.
- Without a **round-0 baseline** trained on real images only, there is nothing
  to compare against and "the synthetic data helped" is untestable.

`train_inspector.py` takes a fixed real-only holdout and `run_rounds.py` runs
round 0 as the control. Keep both.

### 8. Xet stalls

`-e HF_HUB_DISABLE_XET=1`. Hugging Face's Xet transport stalls on some
networks; the download simply never progresses.

### 9. Phase 1 (few-shot fine-tune) needs more than 24 GB

Measured at batch size 1 and 2: the failing allocation is **identical**, so it
is model-resident memory, not activations. No batch-size knob reaches it.

On a 24 GB card, skip it and use NVIDIA's published adapter
`nvidia/Cosmos-AnomalyGen-PCB-2B` (ungated) — the same 2.9M parameters phase 1
would produce. `anomalygen_generate.py` runs `mode=inference_only` for exactly
this. On a bigger card (L40S / A100 / H100), `anomalygen_finetune.py` runs
phases 0+1 properly.

### 10. Version names collide

`next_version_name()` derives the next name from what already exists. Do not
hardcode `v2` — a re-run then fails on a name that is already taken, mid-loop,
after the GPU work is done.

---

## What to tell the user at the end

Report the measurement as measured. On NVIDIA's 86 sample images the honest
finding was: **baseline 0.964, rounds 0.929 flat across 48/60/72 synthetic
frames** — which is one image on a 28-image holdout, and the rare class the
whole lab is about (`bridge`, 8 real examples, ~2 in the holdout) cannot be
resolved in either direction.

That is not a failure of the pipeline and it is not evidence that synthetic data
hurts. It is an instrument without resolution. Say so. If the user has their own
dataset with hundreds of real examples per class, the same code answers the
question properly — and that is the actual recommendation.

---

## Files

| File | What it is |
|---|---|
| `launch.py` | Entry point. Creates + enqueues a task per stage, with the three settings that matter. |
| `probe_parity.py` | Run first. Reports what the task pod actually gave the container. |
| `register_real.py` | NVIDIA's 86 images -> HyperDataset `v1-real`. |
| `hyperdataset.py` | The six `datasets.*` / `frames.*` calls, including `stats()` — the ~1s gap read. |
| `anomalygen_generate.py` | Phases 2–3. Mask placement + generation, per-run output dir. |
| `anomalygen_evaluate.py` | Phase 4. `run_eval.sh`, parses `per_sample.csv`. |
| `anomalygen_finetune.py` | Phases 0–1. Needs >24 GB. |
| `publish_synthetic.py` | Survivors -> next version, with run-scoped dedup and per-frame label earning. |
| `train_inspector.py` | Frozen DINOv2 + logistic regression, fixed real-only holdout. |
| `run_rounds.py` | The bounded loop, with the round-0 control. |
| `ag_common.py` | `link_checkpoints()`, `run()`, `require_hf_token()`. |

Every one of them has a docstring explaining *why* it is shaped the way it is.
Read the docstring before changing the file — most of the odd-looking decisions
are a bug that already bit.
