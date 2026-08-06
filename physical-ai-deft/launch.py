"""Launch the lab's stages as ClearML tasks.

    python launch.py register              # NVIDIA's 86 real images -> HyperDataset v1-real
    python launch.py rounds --rounds 3     # the loop: read the gap, generate, score, publish, train

Run it from a workbench session ON the lab (the VS Code / Code Studio app), or
from anywhere `clearml-init` points at the lab. It does no work itself: it
creates a task that references THIS repository at a commit, and enqueues it. The
GPU work happens on the agent, which is the whole point -- the thing you launch
from holds no GPU and no container runtime.

WHY A LAUNCHER RATHER THAN `clearml-task`
-----------------------------------------
Three settings decide whether these stages run at all, and each one fails in a
way that reads like a model bug rather than a config mistake:

  * `CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1` -- NVIDIA's image already carries
    torch 2.10+cu128 and every AnomalyGen dependency. Left to itself the agent
    builds a fresh venv and shadows all of it.
  * ...which also means `packages=` is never installed, so `clearml` itself has
    to be pip-installed in the setup script or the task dies on `import clearml`.
  * `HF_HUB_DISABLE_XET=1` -- the Xet transport stalls on this network.

Encoding them once here is the difference between a lab that runs first time and
one whose first three tasks fail for reasons nobody can see from the console.

Arguments go through as ENV, not as an appended entry point. Rewriting
`script.entry_point` is what `clearml-task` does internally, but it needs a
private edit call; an env var is public API and shows up on the task record
where a reader can see it.
"""
import argparse
import sys

IMAGE = "nvcr.io/nvidia/paidf-anomalygen:1.0.1"
REPO = "https://github.com/damianerangey-cml/hands-on-labs.git"
BRANCH = "main"
PROJECT = "Physical AI Inspection"
WORKDIR = "physical-ai-deft"

SETUP = "python3 -m pip install -q --no-input clearml scikit-learn"

STAGES = {
    # No GPU: reads 177 files, writes frames.
    "register": ("register_real.py", "Register the real data (HyperDataset v1-real)",
                 "data_processing", "1XCPU"),
    "rounds": ("run_rounds.py", "DEFT loop -- generate, score, publish, train",
               "training", "1XGPU"),
    "generate": ("anomalygen_generate.py", "AnomalyGen -- place masks and generate",
                 "data_processing", "1XGPU"),
    "train": ("train_inspector.py", "Train the inspector", "training", "1XGPU"),
}


def launch(stage, queue=None, env=None):
    from clearml import Task

    script, name, ttype, default_queue = STAGES[stage]
    docker_args = ["-e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1",
                   "-e HF_HUB_DISABLE_XET=1"]
    for k, v in sorted((env or {}).items()):
        docker_args.append("-e %s=%s" % (k, v))

    task = Task.create(
        project_name=PROJECT, task_name=name,
        task_type=getattr(Task.TaskTypes, ttype),
        repo=REPO, branch=BRANCH,
        script="%s/%s" % (WORKDIR, script),
        working_directory=WORKDIR,
        docker=IMAGE,
        docker_args=" ".join(docker_args),
        docker_bash_setup_script=SETUP)

    q = queue or default_queue
    Task.enqueue(task, queue_name=q)
    print("task   %s" % task.id)
    print("queue  %s" % q)
    print("image  %s" % IMAGE)
    print("watch  %s" % task.get_output_log_web_page())
    return task.id


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("--rounds", type=int, default=None,
                   help="rounds stage only: how many enrichment rounds to run")
    p.add_argument("--queue", default=None, help="override the stage's queue")
    a = p.parse_args()

    env = {}
    if a.rounds is not None:
        env["DEFT_ROUNDS"] = a.rounds
    launch(a.stage, queue=a.queue, env=env)


if __name__ == "__main__":
    sys.exit(main())
