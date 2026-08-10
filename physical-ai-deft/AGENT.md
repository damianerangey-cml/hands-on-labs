# You own this inspection dataset

You are running inside a Klique lab, in a workspace with this repository cloned
and ClearML credentials already in the environment. There is a GPU behind a
queue. You do not hold it; you submit work to it.

Your job is to close the gap between the defect examples this dataset has and
the ones it needs — and to be honest about how far you got.

Nothing here is a script to run in order. These are operations. You decide what
to call, with what, and when to stop.

---

## Your state is the dataset, not a file

```python
import deft
deft.gap()        # {'version', 'counts', 'gap', 'at_target'}
deft.history()    # every published version and registered model so far
```

`gap()` is server-side label counts: about a second, downloads nothing. Ask it
as often as you like — before deciding, after publishing, and first thing if
you have lost your place.

**Keep no local state file.** If you crash mid-round, restart and ask what is
true now. A previous you may have published a version you have no memory of;
`history()` is how you find out. This is the one thing that makes the loop
survive its own driver, so do not undermine it by keeping notes on disk and
trusting them over the server.

---

## You are probably not on the server this was built on

This repository was written against one particular ClearML deployment. Its queue
names, its dataset name and its GPU sizes are **local conventions, not facts**.
Assuming them somewhere else fails in the worst available way: enqueueing to a
queue nobody serves is accepted silently and then sits in `queued` forever,
because "no worker has picked this up yet" and "this queue does not exist" look
identical from outside.

So before the first launch:

```python
deft.queues()          # what is actually here, and whether anything serves it
deft.pick_queue("gpu") # resolves, or raises with the question to ask
```

`pick_queue` deliberately does **not** guess from names. A queue called
`gpu-shared` might be eight fractional slices of one card — exactly the wrong
place to send a fine-tune — and the server does not expose enough to tell them
apart. When it cannot resolve, it raises with the list of real queues and the
question. **Ask it.** Ten seconds of a person's time beats a demo that hangs.

Three roles to fill, and you may need one answer per role:

| Role | What it needs | Record the answer as |
|---|---|---|
| `gpu` | a whole GPU, 24 GB is enough — generation, scoring, training | `DEFT_QUEUE_GPU` |
| `cpu` | no GPU — registration and coordination | `DEFT_QUEUE_CPU` |
| `gpu48` | **48 GB or more** — only the phase-1 fine-tune | `DEFT_QUEUE_GPU48` |

Set the env var once you have the answer so you stop asking. If there is no
48 GB queue at all, say so and skip the fine-tune — the rest of the pipeline
runs on 24 GB and this lab has been demonstrating exactly that.

### Everything else that is a local convention

Each of these has a default that is right *here* and may be wrong where you are.
All are environment variables, so you can fix one without editing code.

| Variable | Default | Breaks how, if wrong |
|---|---|---|
| `DEFT_REPO` | this checkout's `origin`, SSH rewritten to HTTPS | **Read this one.** The agent clones this URL — if it points at the upstream while you are editing a fork, your task runs *the original code* and your edits are silently ignored. You then debug something that is not running. |
| `DEFT_BRANCH` | this checkout's current branch | Same class of problem: a task running a branch you are not on. |
| `DEFT_IMAGE` | `nvcr.io/nvidia/paidf-anomalygen:1.0.1` | A different NVIDIA release, or your own rebuild from their Apache-2.0 source. |
| `DEFT_AG_REPO_ROOT` | `/workspace/paidf-anomalygen` | Where NVIDIA's code sits *inside* the image. A rebuild can move it, and every stage then dies on a relative script path with a bare "No such file or directory". |
| `DEFT_CACHE` | `/cache` | The shared path between stages. Without one, each stage re-downloads ~22 GB — slow, not wrong. |
| `DEFT_PROJECT` | `Physical AI Inspection` | Where tasks and models land. |
| `DEFT_HYPERDATASET` | `PCB Inspection` | Which dataset the gap is read from. |
| `DEFT_DATASET` | `pcb-uc1` | NVIDIA's use-case id. **`ensure_dataset` calls `prepare_dataset_uc1.py`** — they ship `uc2` and `uc3` too, and this lab has only ever been run against `uc1`. Another use case needs that call generalised, not just this variable changed. |

Two things are **not** parameterised and you should know why:

- **The defect types are read from `defect_spec.jsonl`**, not configured. They
  used to be a hardcoded PCB list, which silently scored the wrong classes on
  any other dataset — the eval returns NaN for types that are absent and never
  looks at the ones present, so the batch comes back "unscored" for a reason
  nothing in the output explains.
- **The adapter and its step (`Cosmos-AnomalyGen-PCB-2B`, iteration 14000) are
  pinned** in `anomalygen_generate.py`. They are NVIDIA's published PCB
  weights. A different use case needs different weights, and that is a real
  change rather than a setting.

## Launch stages, do not call them

`gap()` and `history()` are yours to call directly — they are plain API reads
and work from anywhere.

**Everything else must be launched as a task**, with `launch.py`:

```bash
python launch.py generate --gap bridge:52,excess_solder:44 --run-id myrun-r1
python launch.py evaluate --run-dir /cache/results/pcb-uc1/runs/myrun-r1
python launch.py improve  --run-dir /cache/results/pcb-uc1/runs/myrun-r1 --search-rounds 1
python launch.py publish  --run-dir /cache/results/pcb-uc1/runs/myrun-r1/searched --run-id myrun-r1
python launch.py train    --name inspector-round1
```

The reason is not style. **`/cache` is the task pod's volume.** Wherever you are
running — a Claude Code session, a workbench, a laptop — you almost certainly do
not have it. Call a stage in-process from there and it fails with

```
no SDG_result.csv under /cache/results/... -- run generation first
```

which reads like generation never happened. It did; you just cannot see its
output from where you are standing. The in-process functions in `deft.py` exist
for code already running *inside* a task, which is how `run_rounds.py` uses
them.

One consequence worth knowing: **publish runs on the GPU queue even though it
needs no GPU.** The CPU queue's pods deliberately drop this recipe's overrides,
so they have no cache mount. A stage that touches `/cache` has to go where the
volume is.

---

## The operations

| Call | What it does | What you decide |
|---|---|---|
| `deft.gap(target=60)` | Reads the shortfall | The target |
| `deft.generate(gap=…, n=24, run_id=…)` | Masks from the board's CAD, then generation | How many, and of what — pass `per_defect_counts` to split it yourself |
| `deft.score(run_dir)` | NVIDIA's `nn_score` against the real examples | The threshold, or let it default to the batch median |
| `deft.improve(run_dir, search_rounds=1)` | Re-rolls generation parameters, keeps the best attempt per sample, regenerates what is still short | Whether it is worth the GPU |
| `deft.publish(run_dir, run_id)` | Commits survivors as the next immutable version | Nothing — the gate decides what earns a label |
| `deft.train(name, use_synthetic=…)` | Trains an inspector, registers it against the version | When, and whether this is the control |

Each returns a dict. Read it — `generate` tells you what was actually delivered
per class, `score` gives you the distribution, `improve` reports what the search
bought and what it did not.

You can also import the stage modules directly and call them with arguments
this file does not mention. `deft.py` is a convenience, not a fence.

---

## Rules you do not get to break

These are not style preferences. Each one is a way the loop can start lying to
itself, and every one has happened.

1. **`run_id` must be unique per invocation.** Reuse one and `publish()` treats
   the round as already published and adds nothing — no error, no frames, and a
   round that looks like it worked. Scope it to your session.

2. **A frame earns its label from the gate.** Never publish a generated frame
   under its defect class because you expect it to be good. What the gate
   rejects lands as `pending-review` and counts toward nothing. You are reading
   these counts next round; if you inflate them you are deceiving yourself.

3. **The real data is registered once.** If `v1-real` exists, reuse it. A second
   copy doubles every count you reason over.

4. **Bound your own spending.** Each search round regenerates every sample.
   `--rounds 3 --search-rounds 2` is nine batches of generation, not three.
   Decide a ceiling before you start and hold to it.

5. **Report what you measured.** If the holdout cannot resolve the difference
   between two rounds, say that. On NVIDIA's 86 sample images it cannot — 28
   held-out images means accuracy moves in steps of 3.6%, and `bridge` has about
   two examples in there. A flat line is not evidence of anything.

---

## Judgement worth exercising

The interesting decisions are not "how many" — that is arithmetic once you have
the gap. They are:

- **A class scoring badly.** `excess_solder` has come in around 0.52–0.66 while
  `bridge` sits near 0.66. Is that the generator, the defect spec describing the
  wrong thing, or a class that will not synthesise on this board? `improve()`
  with a search is the cheap way to find out; giving up on the class and
  spending the GPU on `bridge` is a legitimate answer too.
- **Whether a search is worth it.** It costs a full regeneration per round. On
  one batch a lower guidance helped; on another it made no difference. You will
  not know in advance — but you can look at what the last one bought.
- **When to stop.** At target, out of budget, or when a round adds nothing. If
  the gap did not move, running the same generation again will not move it.

`draws_for_round()` in `anomalygen_improve.py` is where per-sample generation
parameters are chosen — NVIDIA documents that argument as "Claude-selected
hyperparameters per sample", so they built it expecting you. It currently holds
a two-point grid. Replacing it is fair game.

---

## When something fails

- `no per_sample.csv` → you called `improve()` before `score()`.
- `A dataset version with the provided name already exists` → the real data is
  already registered. That means "done", not "pick another name".
- `0 free` missing from the mask-placement line → the CAD did not reach the
  container and the defects landed in arbitrary places. **Stop.** The images are
  worthless and nothing else will tell you.
- `asked N got fewer` → usually legitimate. Automatic Mask Placement can only
  put a defect where the CAD says that fault can occur. Check the AMP log for
  `ok < requested` before treating it as a bug.

`SKILL.md` has the full list, including the ones that cost a GPU hour.

---

## What you should have at the end

A published dataset version per round, a registered model per round plus a
real-only baseline, and a short account of what you decided and why — including
the decisions that did not pay off. The platform already recorded what ran; what
it cannot record is your reasoning, and that is the part a person reading this
tomorrow will want.
