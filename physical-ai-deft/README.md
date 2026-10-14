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

## Run it on your own environment

Nothing here is specific to one server. You need:

| What | Notes |
|---|---|
| A Klique (ClearML Enterprise) environment with **HyperDatasets** | The loop reads its state from dataset label counts. |
| The platform's **Claude Code** app | It wires the platform's MCP server into the session. The agent is started from `PROMPT.md`. |
| A **whole GPU of 24 GB or more** behind a queue | Generation, scoring, training. A fractional GPU will run out of memory. |
| Optionally, a GPU **larger than 24 GB** | Only the few-shot fine-tune. Without one it is skipped and NVIDIA's published adapter is used. |
| Optionally, **shared storage** on task pods | Lets stages hand files to each other. Without it, run the loop as one task (`launch.py rounds`). |
| `NGC_KEY`, `NGC_API_KEY` and `HF_TOKEN` in the task environment, and pull access to `nvcr.io` | From the Configuration Vault or a Kubernetes secret; see Credentials. `probe_parity.py` checks them. |

The agent asks for anything it cannot discover (which queue serves which role,
where the shared path is) and records the answers. No queue name, host or image
layout is built in.

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
               run_dir=RUN_DIR, queue=deft.pick_queue("gpu"))
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

`NGC_KEY` (set `NGC_API_KEY` to the same value; NVIDIA's tools read both) and
`HF_TOKEN` are **never** passed as task parameters: task parameters are
readable by anyone with project access. They should reach the task from
wherever your platform keeps secrets. On Klique that is usually the user's
Configuration Vault:

```
environment {
  NGC_KEY: "nvapi-..."
  NGC_API_KEY: "nvapi-..."
  HF_TOKEN: "hf_..."
}
```

or a Kubernetes secret projected into task pods. The workers also need pull
access to `nvcr.io` (an image pull secret). `probe_parity.py` checks they are present without printing them.

---

A hosted hands-on lab may arrive with these scripts **pre-seeded** as tasks
that reference this folder by commit. On your own environment the agent creates
those seed tasks itself the first time (`launch.py <stage> --dry-run`). There is
nothing hidden and nothing environment-specific: it is ordinary Python against
the ClearML SDK, and it runs on any server you point it at. `AGENT.md` is the operations contract, and it
assumes nothing about your queues, your hardware, or your naming.
