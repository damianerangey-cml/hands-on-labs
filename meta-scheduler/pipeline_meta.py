#!/usr/bin/env python3
"""The meta-scheduler lab's capstone: ONE pipeline whose steps run on TWO
schedulers at the same time.

    check_planes (CPU pod, Kubernetes)
        ├─► train_on_slurm       (sbatch --gres=shard:2, Slurm EC2 / T4)
        └─► train_on_kubernetes  (pod, CFGI half-slice, Karpenter A10G)
                     └────┬────┘
                     crown_champion (CPU pod: compares voice_score,
                                     tags the winning model "champion")

The two training steps are clones of the SAME seeded task — `Fine-tune
(Slurm)` — differing only in the queue each is enqueued to. The DAG therefore
*is* the meta-scheduler demonstration: identical work, dispatched concurrently
to an HPC cluster and a Kubernetes cluster, judged by a third step that never
asks which scheduler produced which model.

How it runs in the lab: this script is itself a seeded draft task
(`Cross-Scheduler Pipeline`). Clone + enqueue it to `1XCPU`; the pod that picks
it up becomes the pipeline controller, and the steps fan out to their queues.
This is also exactly what a CI job would invoke on push — GitOps for a
two-scheduler estate.
"""
from clearml import PipelineController

PROJECT = "Examples"
BASE_TASK = "Fine-tune (Slurm)"

CPU_QUEUE = "1XCPU"
SLURM_QUEUE = "train-0.5xgpu"       # consumed by clearml-agent-slurm on the EC2
K8S_QUEUE = "train-k8s-0.5xgpu"     # consumed by the K8s glue agent (CFGI 0.500)


def check_planes():
    """Sanity-gate: both training queues exist before we spend GPU time.

    Runs as a function step. IMPORTANT: function steps are serialized
    STANDALONE — module globals do NOT travel with them — so the queue names
    are inlined here, not read from the module constants.
    """
    from clearml.backend_api.session.client import APIClient

    slurm_q, k8s_q = "train-0.5xgpu", "train-k8s-0.5xgpu"
    client = APIClient()
    names = {q.name for q in client.queues.get_all()}
    missing = [q for q in (slurm_q, k8s_q) if q not in names]
    if missing:
        raise RuntimeError("training queues missing: %s" % missing)
    print("both planes reachable: %s (Slurm), %s (Kubernetes)" % (slurm_q, k8s_q))
    return True


def crown_champion(slurm_task_id, k8s_task_id):
    """Compare the two runs' voice_score and tag the winner's model "champion".

    The judge reads scalars off both tasks and never asks which scheduler ran
    which — the registry and the metrics are scheduler-neutral. Standalone
    function step: everything imported inside, tagging via the raw models.edit
    endpoint (deterministic across SDK versions).
    """
    from clearml import Task

    def score(tid):
        t = Task.get_task(task_id=tid)
        m = t.get_last_scalar_metrics() or {}
        for metric, variants in m.items():
            for variant, vals in (variants or {}).items():
                if "voice_score" in (str(metric), str(variant)):
                    try:
                        return float(vals.get("last", 0.0)), t
                    except (TypeError, ValueError):
                        return 0.0, t
        return 0.0, t

    s_score, s_task = score(slurm_task_id)
    k_score, k_task = score(k8s_task_id)
    print("voice_score — Slurm: %.3f  Kubernetes: %.3f" % (s_score, k_score))

    winner = s_task if s_score >= k_score else k_task
    label = "slurm" if winner is s_task else "kubernetes"
    models = (winner.get_models() or {}).get("output") or []
    if not models:
        raise RuntimeError("winning task %s registered no model" % winner.id)
    champion = models[-1]
    # Tag on the MODEL, not the task: the serving act picks models by tag/UUID.
    from clearml.backend_api import Session
    res = Session().send_request(
        service="models", action="edit", method="post",
        json={"model": champion.id,
              "tags": ["champion", "trained-on-%s" % label]})
    if res.status_code != 200:
        raise RuntimeError("models.edit failed: HTTP %s %s"
                           % (res.status_code, res.text[:300]))
    print("champion: %s (%s, trained on %s)" % (champion.id, champion.name, label))
    return champion.id


def main() -> None:
    pipe = PipelineController(
        name="Cross-Scheduler Pipeline",
        project=PROJECT,
        version="1.0.0",
        add_pipeline_tags=True,
    )
    pipe.set_default_execution_queue(CPU_QUEUE)

    pipe.add_function_step(
        name="check_planes", function=check_planes,
        execution_queue=CPU_QUEUE, cache_executed_step=False)

    # The SAME seeded task, cloned twice — only the queue differs. That is the
    # entire point of this pipeline.
    pipe.add_step(
        name="train_on_slurm",
        base_task_project=PROJECT, base_task_name=BASE_TASK,
        parents=["check_planes"], execution_queue=SLURM_QUEUE)
    pipe.add_step(
        name="train_on_kubernetes",
        base_task_project=PROJECT, base_task_name=BASE_TASK,
        parents=["check_planes"], execution_queue=K8S_QUEUE)

    pipe.add_function_step(
        name="crown_champion", function=crown_champion,
        function_kwargs={"slurm_task_id": "${train_on_slurm.id}",
                         "k8s_task_id": "${train_on_kubernetes.id}"},
        parents=["train_on_slurm", "train_on_kubernetes"],
        execution_queue=CPU_QUEUE, cache_executed_step=False)

    # When an agent runs this script (the seeded-draft path), the pod becomes
    # the controller and the steps fan out to their queues. `start_locally`
    # here means "run the controller loop in THIS process" — the steps still
    # execute remotely on their execution_queues.
    pipe.start_locally(run_pipeline_steps_locally=False)
    print("pipeline complete.")


if __name__ == "__main__":
    main()
