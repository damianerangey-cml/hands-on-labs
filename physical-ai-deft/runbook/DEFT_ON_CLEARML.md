# Runbook: running NVIDIA's DEFT AOI loop on ClearML

**Read this before you start the loop. It changes exactly one thing about how you
execute NVIDIA's `tao-run-deft-aoi` skill, and nothing about the loop's logic.**

You are running on a ClearML platform. You have **no docker daemon and no GPU**.
That is deliberate: you submit work, you do not run it. Every GPU stage runs in
the same container image NVIDIA specifies, on a GPU worker, through a queue --
with its console, metrics, artifacts and lineage recorded.

---

## The one substitution

Wherever NVIDIA's skill or its stage references tell you to run a container:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
    -v $WS:$WS -w /workspace/paidf-anomalygen $AG_IMAGE \
    bash -lc "${ANOMALYGEN_SCRIPTS}/run_sdg.sh --checkpoint_dir ... --step ..."
```

do this instead:

```python
from run_stage import run_stage

run_stage(
    image=AG_IMAGE,
    command="${ANOMALYGEN_SCRIPTS}/run_sdg.sh --checkpoint_dir ... --step ...",
    queue="1XGPU",
    stage="anomalygen",
    iteration="iter1",
    workdir="/workspace/paidf-anomalygen",
)
```

`run_stage()` blocks until the stage finishes and raises on failure with the last
40 console lines attached, so your control flow is unchanged from the docker
version. Common stages have named helpers in `deft_stages.py` -- use those where
they exist, `stage()` for anything else.

**Take the command verbatim from NVIDIA's reference.** Everything inside the
container is unchanged: same image, same script, same flags, same workspace
paths. Do not rewrite a stage's CLI to "fit" ClearML. If a reference gives you a
command this runbook does not cover, pass it through `deft_stages.stage()`.

### Which queue

Queue = how much GPU the stage gets. Ask for what the stage needs:

| Stage | Queue | Why |
|---|---|---|
| `anomalygen` (SDG diffusion) | `1XGPU` | Cosmos-Predict2 2B + text encoder; wants a whole card |
| `train` | `0.5XGPU` | ChangeNet fine-tune fits a half slice |
| `evaluate`, `inference`, `data_mining`, `routing` | `0.25XGPU` | short, small |
| `anomalygen_prep` (AMP routing) | `0.25XGPU` | ~10s, no diffusion |

If a stage OOMs, move it up a size and say so in the loop log -- do not silently
shrink the workload.

---

## Pre-Flight: what changes

NVIDIA's pre-flight checks a docker host. You do not have one. Replace those
specific checks; keep every other check exactly as written.

| NVIDIA's check | On ClearML |
|---|---|
| `docker image inspect <image>` | skip -- the cluster pulls the image; a bad image surfaces as a task that fails to start, and the task console says so |
| GPU host runtime / driver check | skip -- the GPU worker owns its driver |
| workspace + specs + CSVs exist | **keep** -- run it, the workspace is mounted here too |
| checkpoints staged | **keep** |
| credentials resolve (`NGC_KEY`, `HF_TOKEN`) | **changed** -- do not read values. They are injected into task pods from the lab's Kubernetes Secret. Confirm presence with `probe_parity.py`, not `[ -n "$NGC_KEY" ]` in your own shell |

Add one check of your own before the gate: submit `probe_parity.py` if it has not
been run on this lab. It answers "can a stage see the workspace, a GPU, enough
/dev/shm and the credentials?" in about a minute, and every DEFT failure mode
downstream of that is a real failure rather than a plumbing surprise.

**The user gate is unchanged.** Print the Pre-Flight Summary, stop, wait for
explicit approval. Add two lines to the summary: which queues the stages will use
and the loop controller task URL.

---

## State, logging and the loop's own files

**Keep NVIDIA's state machine exactly as it is.** `results/deft_state.json` and
`results/loop_log.jsonl` stay the source of truth: still written by
`scripts/log_stage.py`, still read from disk before every stage, `seq` still
computed from the live tail. Do not route logging "through ClearML instead".

ClearML mirrors that record; it does not replace it:

1. **Once, at loop start** -- create the controller task:

   ```python
   import clearml_deft_hooks as hooks
   hooks.loop_task(workspace="/workspace/deft", kpi_target="FAR < 0.5% at recall=100%")
   ```

   It writes its id into the shared workspace so every stage pod finds it.

2. **Per stage** -- nothing to do. `stage_entry.py` reports metrics and mirrors the
   headline KPI onto the controller automatically.

3. **Per iteration** -- register what the iteration produced:

   ```python
   hooks.register_synthetic(sdg_dir, iteration="iter1", parents=[real_dataset_id])
   hooks.register_training_set(csv_dir, iteration="iter1",
                               parents=[real_dataset_id, synthetic_dataset_id])
   hooks.register_checkpoint(train_task, ckpt_path, iteration="iter1", far=3.11)
   ```

   Parent the training set on **both** the real and the synthetic dataset. That
   parentage is what later answers "was any of this synthetic, and which images?"

4. **At loop end**, after `scripts/align_token_usage.py` has backfilled tokens:

   ```bash
   python clearml_deft_hooks.py /workspace/deft
   ```

   This replays `loop_log.jsonl` onto the controller -- including per-stage token
   counts, so agent tokens plot beside GPU seconds. Run it after the alignment
   step, never before: the token numbers do not exist until then.

---

## When a stage fails

`run_stage()` raises with the last console lines. Do what NVIDIA's skill already
tells you: **halt, surface the disk evidence verbatim, do not auto-retry.** The
hard-stop gates are unchanged (train/val leakage, empty mining pool, silent drop,
AMP allocation mismatch).

Two failure shapes are new here, and both are infrastructure rather than model
problems -- report them plainly instead of working around them:

- **Task sits queued.** No worker on that queue, or no GPU node yet. It is not a
  reason to shrink the job or move to a different queue.
- **Task fails immediately with a pull error.** The image is not reachable from
  the cluster -- an entitlement or pull-secret problem. Do not substitute a
  different image.

---

## Why it is built this way

NVIDIA's loop is autonomous: one approval, then dozens of GPU jobs that change
your model and your data on your behalf. Its native memory of all that is three
files in a directory, on one machine, for one user.

Running the same stages through a platform means the next morning you can answer
which checkpoint won, what data trained it, how much of that data was synthetic,
what it cost, and whether it can be reproduced -- without the machine that ran it
still existing.

The agent having no GPU and no docker is part of the answer, not a limitation: it
submits jobs under the same queues, quotas and RBAC as any human user.
