# Physical AI Data Factory + DEFT — example code

*NVIDIA's autonomous model-improvement loop, run on ClearML.*

NVIDIA's [`tao-run-deft-aoi`](https://github.com/NVIDIA/skills/tree/main/skills/tao-run-deft-aoi)
skill improves a PCB inspection model on its own: evaluate, root-cause the
failures, **generate the defects you don't have** with Cosmos AnomalyGen (the
Physical AI Data Factory synthetic-data stage), mine real images that look like
them, retrain, and gate on a KPI — for as many iterations as you allow, after a
single approval.

It executes every stage as `docker run` on the agent's own host, and remembers
what happened in three files in one directory: `deft_state.json`,
`loop_log.jsonl`, `DEFT_Loop_Report.html`.

This folder changes **one** thing: each stage becomes a **ClearML task** instead.
Same image, same command, same workspace paths — but on a queue, with console,
metrics, artifacts and lineage recorded, and with the agent holding no GPU and no
docker daemon of its own.

## The files

| File | What it does |
|---|---|
| `run_stage.py` | The substitution. `Task.create(docker=<image>)` + enqueue + poll. ~1 call per stage. |
| `stage_entry.py` | Runs *inside* NVIDIA's container: execs the stage command, streams output, hands it to the hooks. |
| `deft_stages.py` | Named helpers per stage (`train`, `evaluate`, `anomalygen_prep`, `anomalygen_sdg`) plus a passthrough for stages whose CLI belongs to another skill. |
| `clearml_deft_hooks.py` | Metrics onto the stage task, KPI onto the loop controller, Datasets + Model with lineage, and the loop-end mirror that produces the agent-tokens-vs-GPU-seconds chart. |
| `probe_parity.py` | Run first. Reports what a task pod actually gave the container: GPU, workspace mount, `/dev/shm`, uid, credentials. |
| `runbook/DEFT_ON_CLEARML.md` | What the agent reads. The contract: how to substitute, which queue per stage, what changes in pre-flight, what to do when a stage fails. |

## The substitution, in one view

```python
# NVIDIA's reference says:
#   docker run --rm --gpus all --ipc=host --shm-size=16g -v $WS:$WS \
#       -w /workspace/paidf-anomalygen $AG_IMAGE \
#       bash -lc "${ANOMALYGEN_SCRIPTS}/run_sdg.sh --checkpoint_dir ... --step ..."

from deft_stages import anomalygen_sdg

anomalygen_sdg(iteration="iter1", checkpoint_dir=CKPT, step=14000,
               run_dir=RUN_DIR, queue="1XGPU")
```

The four `docker run` flags are not lost — they move into the lab's agent pod
template, once per lab rather than once per stage:

| `docker run` | Pod template |
|---|---|
| `-v $WS:$WS` | the workspace PVC, mounted at the same absolute path |
| `--gpus all` | the GPU the queue's profile requests |
| `--ipc=host --shm-size=16g` | an emptyDir-memory `/dev/shm` |
| `--user` + `chmod 777` | `securityContext` / `fsGroup` (AnomalyGen runs as uid 10000) |

`probe_parity.py` verifies all four from inside a real task pod, and tells you
which one is missing if a stage misbehaves.

## What NVIDIA's loop keeps

Everything. `deft_state.json` and `loop_log.jsonl` are still written by
`scripts/log_stage.py`, still read from disk before every stage, `seq` still
computed from the live tail. The hooks **mirror** that record into ClearML — they
never replace it. The loop's own HTML report is uploaded as an artifact, so the
two records can't drift.

## Credentials

`NGC_KEY` and `HF_TOKEN` are **never** passed as task parameters — task
parameters are readable by anyone with project access. They are injected into
task pods from the lab namespace's Kubernetes Secret. `probe_parity.py` checks
they are present without printing them.

---

These scripts are **pre-seeded** by the HOL orchestrator as ClearML tasks that
reference this folder by commit — but there's nothing hidden: it's ordinary
Python you can read, run, and modify.
