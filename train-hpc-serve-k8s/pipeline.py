"""Train the same task on two compute resources at once and keep the better model.

    python pipeline.py --hpc "<HPC compute resource>" --k8s "<Kubernetes compute resource>" \
                       --cpu "<CPU compute resource>"

Needs a draft of train.py in Klique first: run `python train.py --queue ""` once, or clone
an existing run, and pass its task ID with --base_task_id.
"""
import argparse

from clearml import PipelineController


def pick_champion(hpc_task_id, k8s_task_id):
    """Compare the two runs' eval_loss and tag the better model 'champion'."""
    from clearml import Task
    runs = {tid: Task.get_task(task_id=tid) for tid in (hpc_task_id, k8s_task_id)}
    scores = {tid: t.get_reported_single_value("eval_loss") for tid, t in runs.items()}
    best = min(scores, key=lambda tid: scores[tid])
    model = runs[best].models["output"][-1]
    model.tags = list(set((model.tags or []) + ["champion"]))
    print("eval_loss per run:", scores, "| champion model:", model.id)
    return model.id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_task_id", required=True, help="a draft or completed train.py task")
    p.add_argument("--hpc", required=True)   # ADAPT: your HPC compute resource
    p.add_argument("--k8s", required=True)   # ADAPT: your Kubernetes compute resource
    p.add_argument("--cpu", required=True)   # ADAPT: a CPU compute resource for the controller
    # ADAPT: the container for the Kubernetes step. torch 2.7 needs Python 3.9 or later, and a
    # cluster's default image may be older. This one also ships torch, so the step starts sooner.
    p.add_argument("--k8s_image", default="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime")
    p.add_argument("--project", default="Builders - Train on Slurm, Serve on Kubernetes")   # ADAPT
    a = p.parse_args()

    pipe = PipelineController(name="Train on two schedulers", project=a.project, version="1.0.0")
    pipe.set_default_execution_queue(a.cpu)
    pipe.add_step(name="train_hpc", base_task_id=a.base_task_id, execution_queue=a.hpc)
    pipe.add_step(name="train_k8s", base_task_id=a.base_task_id, execution_queue=a.k8s,
                  task_overrides={"container.image": a.k8s_image})
    pipe.add_function_step(
        name="pick_champion", function=pick_champion,
        function_kwargs={"hpc_task_id": "${train_hpc.id}", "k8s_task_id": "${train_k8s.id}"},
        function_return=["champion_model_id"], parents=["train_hpc", "train_k8s"])
    pipe.start(queue=a.cpu)   # the controller runs on the CPU compute resource; CI can call this file


if __name__ == "__main__":
    main()
