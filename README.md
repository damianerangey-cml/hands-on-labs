# ClearML Hands-On-Labs — Example Code

The Python behind the [ClearML Hands-On Labs](https://www.clearml-hol.com). Each
subfolder is one lab; every script is plain Python on the **ClearML SDK**.

The HOL orchestrator pre-seeds these as ClearML tasks that **reference this repo
by commit** — so each task's *Execution* tab shows the real source (repository +
commit + file path) instead of an inline diff, and any run is reproducible
straight from git.

## Labs

| Folder | Lab | Scripts |
|---|---|---|
| [`adapter-lifecycle/`](adapter-lifecycle/) | Enterprise LLMOps: One Base, Many Adapters | `prepare_datasets.py`, `finetune_adapter.py` |
| [`meta-scheduler/`](meta-scheduler/) | ClearML Meta Scheduler (Slurm + Kubernetes) | `finetune_slurm.py` |

## How it's wired

The orchestrator seeds each lab task with `repository=<this repo>`,
`branch=main`, `entry_point=<lab>/<script>.py`. On enqueue, the ClearML agent
clones this repo at that commit, installs the lab's `requirements.txt`, and runs
the script — and the server tracks the run (hyper-parameters, scalars,
artifacts, datasets, output models) from that point on.

That's the whole point: the labs aren't magic, they're **ordinary versioned
Python that ClearML executes and tracks**. Read it, run it, change it.

## Run one yourself

```bash
pip install -r adapter-lifecycle/requirements.txt
# point at your ClearML server (clearml.conf or CLEARML_* env vars), then:
python adapter-lifecycle/prepare_datasets.py
```
