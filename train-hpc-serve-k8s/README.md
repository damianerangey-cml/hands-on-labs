# Train on Slurm, serve on Kubernetes

General-purpose example code for the Klique Builders series guide of the same name. One Python
task fine-tunes an open-weight chat model with LoRA and registers it; the same task runs on an HPC
compute resource (Slurm, PBS or LSF) or a Kubernetes compute resource; vLLM on Kubernetes serves the
registered model; one pipeline trains on both and tags the better model.

| File | What it does |
|---|---|
| `train.py` | LoRA fine-tune, loss curves to Klique, held-out `eval_loss`, merged model registered |
| `pipeline.py` | Runs `train.py` on two compute resources at once and tags the lower-`eval_loss` model `champion` |
| `call_endpoint.py` | Calls the vLLM endpoint with an access token (OpenAI-compatible) |

## Where to adapt it

| Change | Where |
|---|---|
| Base model | `--base_model`: any causal LM on the Hugging Face Hub; pick a compute resource with enough GPU memory |
| Data | `--dataset` (a Hub name or a local `.jsonl`) and the three `*_field` arguments; or replace `load_rows()` |
| Storage for models | `task.output_uri`: platform storage by default; your own bucket for production |
| Where it runs | `--queue` for one run, `--hpc` / `--k8s` / `--cpu` for the pipeline: your compute resource names |
| Serving | vLLM Model Deployment app: a GPU compute resource on Kubernetes, the model ID from the registry, an endpoint name |
| Hub access | Gated models need a Hugging Face token in the Klique Configuration Vault (`HF_TOKEN`) |
| Container on Kubernetes | A Kubernetes compute resource runs the task in a container. Set the task's image (Execution › Container, or `--k8s_image` in the pipeline) to one with Python 3.9 or later, such as `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` |
| Package versions | `requirements.txt`: a torch build every cluster's GPU driver supports, the serving image's transformers and tokenizers versions, and a `datasets` release that accepts the same `huggingface_hub` |
