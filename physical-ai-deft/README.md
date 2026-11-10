# Physical AI Data Factory: an agent-driven enrichment loop

*NVIDIA's Cosmos AnomalyGen, run as tasks on Klique (ClearML Enterprise), with
an agent deciding each round.*

An inspection dataset is short of its rare defect classes. This folder lets an
agent close that gap: it reads the shortfall from the dataset's label counts,
generates the missing defects with NVIDIA's AnomalyGen, keeps only the frames
that score well against the real examples, publishes them as the next dataset
version, and retrains an inspection model to see whether it helped.

Every step is a task on a queue. The agent holds no GPU. It clones a template
task, writes one line of arguments into the clone's comment, enqueues it and
reads the result back, so any tool that can do those things can drive the loop.

## Run it

Start with [`PROMPT.md`](PROMPT.md): one prompt, two queue names.
[`AGENT.md`](AGENT.md) is what the agent reads.

You need:

| What | Notes |
|---|---|
| A Klique (ClearML Enterprise) environment with **HyperDatasets** | The loop reads its state from dataset label counts. |
| The platform's **Claude Code** app, or any agent with the platform's MCP tools | It needs to create a task, edit its comment, enqueue it and read it back. |
| A queue with a **whole GPU of 24 GB or more** | Generation, scoring and training. A fractional GPU runs out of memory. |
| A queue with **no GPU** | Status reads and registration. |
| `HF_TOKEN` in the task environment, with the Cosmos model licences accepted | From your Configuration Vault or a platform secret. Never on the task. |
| Pull access to `nvcr.io` on the workers | For `nvcr.io/nvidia/paidf-anomalygen:1.0.1`. |

No shared storage is needed. A round generates, scores and publishes inside one
task, and training reads its frames from the published dataset version. A
volume that outlives a task is only a speed-up for the checkpoint download.

## The files

| File | What it is |
|---|---|
| `run.py` | The one entry point. Reads `stage=...` from the task's comment and runs that stage. `stage=template` creates the task the others are cloned from. |
| `register_real.py` | NVIDIA's 86 real images into HyperDataset version `v1-real`. |
| `anomalygen_generate.py` | Mask placement from the board's CAD, then generation, allocated by shortfall. |
| `anomalygen_evaluate.py` | NVIDIA's `nn_score` for every frame against the real examples. |
| `anomalygen_improve.py` | NVIDIA's parameter search, keep-best and regeneration. |
| `publish_synthetic.py` | The survivors become the next immutable version. A frame earns its label only if the gate accepted it. |
| `train_inspector.py` | Frozen DINOv2 features and a linear classifier, on a fixed real-only holdout. |
| `train_customer.py` | A Hugging Face ViT fine-tuned end to end, standing in for your own model. |
| `anomalygen_finetune.py` | NVIDIA's few-shot fine-tune of the generator. Needs more than 24 GB. |
| `hyperdataset.py` | The HyperDataset calls, over the stock `clearml` package. |
| `ag_common.py` | Setup every AnomalyGen stage shares. |
| `TRAPS.md` | Failures that already happened, and what they really were. |

## Credentials

`HF_TOKEN` (and `NGC_KEY` / `NGC_API_KEY` if your workers need them) reach a
task from wherever your platform keeps secrets. On Klique that is usually your
Configuration Vault:

```
environment {
  HF_TOKEN: "hf_..."
}
```

They are never passed as task arguments, names or comments: those are readable
by anyone with access to the project. The preflight stage reports whether they
arrived, without printing them.
