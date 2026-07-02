#!/usr/bin/env python3
"""Capstone automation for the adapter-lifecycle lab: the whole flow as ONE
ClearML pipeline.

After doing the steps by hand (Acts 1-7), run this to reproduce the entire
lifecycle as a single, parameterized, triggerable pipeline:

    Prepare Datasets ──► SupportBot Fine-tune ──┐
                     └─► SalesBot Fine-tune  ───┴─► deploy

The two fine-tunes run IN PARALLEL on fractional GPUs. It reuses the lab's two
already-seeded, pre-configured fine-tune tasks (each cloned as-is), so there's
no duplicate code.
This is also exactly what a CI job or ArgoCD hook would invoke to retrain + roll
the endpoint on new data.

    python pipeline_adapter.py        # creates + starts the pipeline on the cluster

Requires ClearML credentials configured (CLEARML_API_* or ~/clearml.conf), the
seeded tasks present ("Prepare Datasets", "SupportBot Fine-tune", "SalesBot Fine-tune"), and the lab's
queues (1XCPU, 0.5XGPU). Needs live testing on the cluster.
"""
from clearml import PipelineController

PROJECT = "Fine-Tuned Chatbots"
CPU_QUEUE = "1XCPU"
GPU_QUEUE = "0.5XGPU"


def deploy_step(project_tag_a="SupportBot", project_tag_b="SalesBot"):
    """Resolve both project models by tag and (re)deploy the multi-model vLLM
    endpoint via the apps API. Self-contained (pipeline function step)."""
    from clearml import Model
    from clearml.backend_api import Session

    def latest(tag):
        # Search across ALL projects: pipeline steps register their models in the
        # pipeline's own project, not Examples, so a project-scoped query misses them.
        ms = Model.query_models(tags=[tag], max_results=1)
        if not ms:
            raise RuntimeError("no model tagged %r" % tag)
        return ms[0].id

    # Full model-deployment launch_params (the apps API requires ALL of these;
    # mirror seed_configs/multimodel-serve.json or you get a 400 missing-param).
    def instance(tag):
        return {
            "cli": None, "model": latest(tag), "served_model_name": tag,
            "max_model_len": "4096", "max_num_seqs": "1",
            "gpu_memory_utilization": "0.9", "tensor_parallel_size": None,
            "dtype": "auto", "kv_cache_dtype": "auto", "tokenizer_mode": "auto",
            "quantization": None, "max_concurrent_requests": "10",
            "enforce_eager": False, "enable_prefix_caching": False,
            "trust_remote_code": False, "model_overrides": None,
        }

    cfg = {
        "task_name": "project-models-serve", "project": "Fine-Tuned Chatbots",
        "queue_name": "0.5XGPU", "route_name": "inference-internal",
        "config_per_instance": [instance(project_tag_a), instance(project_tag_b)],
        "enable_lora": False, "use_v1_engine": True, "trust_remote_code": False,
        "hf_token": None, "max_cuda_mem": None, "load_format": "auto",
        "enable_automatic_cpu_offloading": True, "enable_disk_model_swapping": False,
        "environment_vars_list": [], "max_idle_time_hour": "None",
        "last_activity_report_interval_seconds": 300,
    }
    res = Session().send_request(
        service="apps", action="launch_instance", method="post",
        json={"app": "model-deployment", "launch_params": cfg, "task_name": cfg["task_name"]})
    ok = getattr(res, "ok", False)
    print("deploy:", getattr(res, "status_code", "?"), getattr(res, "text", "")[:300])
    if not ok:  # never let a failed launch pass as a green step
        raise RuntimeError("apps.launch_instance failed (see status above)")
    return cfg["task_name"]


def main() -> None:
    pipe = PipelineController(name="Adapter Lifecycle Pipeline", project=PROJECT, version="1.0.0")
    pipe.set_default_execution_queue(CPU_QUEUE)

    pipe.add_step(
        name="prepare_data", base_task_project=PROJECT, base_task_name="Prepare Datasets",
        execution_queue=CPU_QUEUE)
    # Each fine-tune step clones its own PRE-CONFIGURED seeded task — no
    # parameter_override needed (SupportBot Fine-tune = LoRA/supportbot-data,
    # SalesBot Fine-tune = QLoRA/salesbot-data). They share one parent and don't
    # depend on each other, so the platform runs them in parallel.
    pipe.add_step(
        name="finetune_a", base_task_project=PROJECT, base_task_name="SupportBot Fine-tune",
        parents=["prepare_data"], execution_queue=GPU_QUEUE)
    pipe.add_step(
        name="finetune_b", base_task_project=PROJECT, base_task_name="SalesBot Fine-tune",
        parents=["prepare_data"], execution_queue=GPU_QUEUE)
    pipe.add_function_step(
        name="deploy", function=deploy_step, parents=["finetune_a", "finetune_b"],
        execution_queue=CPU_QUEUE)

    # The controller runs where you launch it (your shell / CI runner); each step
    # is enqueued to its execution_queue and runs on the cluster. We use
    # start_locally so the lab works even though the agents can't clone this
    # private repo -- the seeded step tasks carry their own code as diffs. If the
    # pipeline code lives in an agent-reachable (e.g. public) repo, switch to a
    # fully-remote controller with:  pipe.start(queue=CPU_QUEUE)
    pipe.start_locally(run_pipeline_steps_locally=False)
    print("pipeline started.")


if __name__ == "__main__":
    main()
