#!/usr/bin/env python
"""Hyperparameter optimization for the adapter fine-tune.

Clones the latest completed "SupportBot Fine-tune" task and launches a small
Optuna sweep over the LoRA rank and learning rate, ranking trials by the
training loss (the live `train/loss` scalar). Every trial runs on the same
fractional-GPU queue as the rest of the lab — HPO reuses the platform you
already have, and every trial is tracked automatically with full lineage.

Run it like the other deploy/ scripts, with CLEARML_API_* pointing at your
lab tenant:

    CLEARML_API_HOST=... CLEARML_API_ACCESS_KEY=... CLEARML_API_SECRET_KEY=... \
        python adapter-lifecycle/automation/hpo_adapter.py
"""
from clearml import Task
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

PROJECT = "Fine-Tuned Chatbots"
BASE_TASK_NAME = r"^SupportBot Fine-tune$"   # the SupportBot fine-tune (exact name)
EXEC_QUEUE = "0.5XGPU"


def latest_base_task_id():
    # Pipeline runs register their fine-tunes in the pipeline's OWN project, not
    # "Fine-Tuned Chatbots", so prefer a completed fine-tune anywhere; fall back to the
    # seeded "SupportBot Fine-tune" draft (the optimizer clones its config either way).
    tasks = Task.get_tasks(
        task_name=BASE_TASK_NAME,
        task_filter={"status": ["completed"], "order_by": ["-last_update"]},
    )
    if not tasks:
        tasks = Task.get_tasks(project_name=PROJECT, task_name=BASE_TASK_NAME)
    if not tasks:
        raise SystemExit("No 'SupportBot Fine-tune' task found to optimize.")
    return tasks[0].id


def main():
    base_id = latest_base_task_id()
    print("HPO base task:", base_id)

    task = Task.init(
        project_name=PROJECT,
        task_name="Adapter HPO Sweep",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_id,
        # Search space — the two knobs that matter most for a LoRA adapter.
        hyper_parameters=[
            DiscreteParameterRange("hparams/lora_r", values=[8, 16, 32]),
            DiscreteParameterRange("hparams/learning_rate", values=[1e-4, 2e-4, 3e-4]),
        ],
        # Objective: minimize the training loss (the live train/loss scalar).
        objective_metric_title="train",
        objective_metric_series="loss",
        objective_metric_sign="min",
        optimizer_class=OptimizerOptuna,
        execution_queue=EXEC_QUEUE,
        # Lab-fast: just 2 trials, run them together, and poll often so there
        # are no long sleeps between trials (the default pool period is minutes).
        total_max_jobs=2,
        max_number_of_concurrent_tasks=2,
        pool_period_min=0.2,
        min_iteration_per_job=0,
        max_iteration_per_job=500,
    )

    optimizer.set_time_limit(in_minutes=15)
    optimizer.start()
    print("Optimizer task:", task.id)
    optimizer.wait()
    top = optimizer.get_top_experiments(top_k=3)
    print("Top trials:", [t.id for t in top])
    optimizer.stop()
    print("HPO done.")


if __name__ == "__main__":
    main()
