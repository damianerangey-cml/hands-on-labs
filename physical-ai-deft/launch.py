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
import os
import pathlib
import sys

# EVERY ONE OF THESE IS OVERRIDABLE, AND THE REPO ONE MATTERS MOST.
#
# `Task.create(repo=...)` makes the AGENT clone that repository -- so if you
# fork this, edit a stage, and run launch.py without changing REPO, your task
# clones the ORIGINAL from GitHub and your edits are silently ignored. You are
# then debugging code that is not running. The default is therefore derived
# from this checkout's own `origin` remote, so a fork does the obvious thing;
# DEFT_REPO / DEFT_BRANCH override it, and the literal is only the last resort
# for a copy with no git remote at all.
def _git(*args, default=""):
    import subprocess
    try:
        return subprocess.run(("git",) + args, cwd=str(pathlib.Path(__file__).parent),
                              capture_output=True, text=True, timeout=5
                              ).stdout.strip() or default
    except Exception:
        return default


IMAGE = os.environ.get("DEFT_IMAGE", "nvcr.io/nvidia/paidf-anomalygen:1.0.1")
def _https(url):
    """git@host:owner/repo.git -> https://host/owner/repo.git

    The agent clones this URL inside a task pod that has no SSH key, so an SSH
    remote -- which is what a developer's own checkout usually has -- would fail
    the clone with a permission error that looks like a credentials problem with
    ClearML rather than with git.
    """
    if url.startswith("git@") and ":" in url:
        host, _, path = url[4:].partition(":")
        return "https://%s/%s" % (host, path)
    return url


REPO = (os.environ.get("DEFT_REPO")
        or _https(_git("remote", "get-url", "origin"))
        or "https://github.com/damianerangey-cml/hands-on-labs.git")
BRANCH = (os.environ.get("DEFT_BRANCH")
          or _git("rev-parse", "--abbrev-ref", "HEAD", default="main"))
PROJECT = os.environ.get("DEFT_PROJECT", "Physical AI Inspection")
WORKDIR = os.environ.get("DEFT_WORKDIR", "physical-ai-deft")

SETUP = "python3 -m pip install -q --no-input clearml scikit-learn"

STAGES = {
    # No GPU: reads 177 files, writes frames.
    "register": ("register_real.py", "Register the real data (HyperDataset v1-real)",
                 "data_processing", "cpu"),
    "rounds": ("run_rounds.py", "DEFT loop -- generate, score, publish, train",
               "training", "gpu"),
    "generate": ("anomalygen_generate.py", "AnomalyGen -- place masks and generate",
                 "data_processing", "gpu"),
    "evaluate": ("anomalygen_evaluate.py", "AnomalyGen -- score against the real data",
                 "data_processing", "gpu"),
    # The 48 GB lane. NVIDIA's phase 1 wants 4.54 GiB more than a 24 GB card
    # has, so it targets a queue wired to its own NodePool of g6e.* machines.
    # Expect a ~10 min cold start: that pool has no prewarm and no keepalive, so
    # the card is created when this task asks for it and released after.
    "finetune": ("anomalygen_finetune.py",
                 "AnomalyGen phase 1 -- few-shot fine-tune (48GB)",
                 "training", "gpu48"),
    "improve": ("anomalygen_improve.py",
                "AnomalyGen -- search, keep best, filter and regenerate",
                "data_processing", "gpu"),
    "train": ("train_inspector.py", "Train the inspector", "training", "gpu"),
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

    # RESOLVE, do not assume. `default_queue` is a KIND ("gpu"/"cpu"/"gpu48"),
    # not a name -- the names in this repo are the HOL labs' own and are wrong
    # on any other server. deft.pick_queue() checks what actually exists and
    # raises a question rather than enqueueing into a queue nobody serves,
    # which would otherwise sit in `queued` forever looking like a slow start.
    if queue:
        q = queue
    else:
        import deft
        q = deft.pick_queue(default_queue)
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
    p.add_argument("--num-sdg", type=int, default=None,
                   help="generate stage only: how many images this round")
    p.add_argument("--gap", default=None,
                   help="generate stage only: explicit shortfall to allocate "
                        "against, as label:count pairs -- "
                        "bridge:52,excess_solder:44. The loop reads this from "
                        "the HyperDataset itself.")
    p.add_argument("--search-rounds", type=int, default=None,
                   help="rounds/improve stages: NVIDIA phase-5 budget -- how "
                        "many times to re-roll the generation parameters before "
                        "keeping the best attempt per sample. 0 runs the filter "
                        "without the search. EXPENSIVE: each one regenerates "
                        "every sample.")
    p.add_argument("--run-dir", default=None,
                   help="improve stage only: the generation run to improve "
                        "(defaults to the most recent)")
    p.add_argument("--baseline", action="store_true",
                   help="train stage only: the CONTROL -- real images only, no "
                        "synthetic. Run this once before generating anything; "
                        "without it there is nothing to compare later rounds to.")
    p.add_argument("--name", default=None,
                   help="train stage only: model name (default inspector, or "
                        "inspector-baseline with --baseline)")
    p.add_argument("--queue", default=None, help="override the stage's queue")
    a = p.parse_args()

    env = {}
    if a.rounds is not None:
        env["DEFT_ROUNDS"] = a.rounds
    if a.num_sdg is not None:
        env["DEFT_NUM_SDG"] = a.num_sdg
    if a.search_rounds is not None:
        env["DEFT_SEARCH_ROUNDS"] = a.search_rounds
    if a.run_dir is not None:
        if " " in a.run_dir:
            p.error("--run-dir cannot contain spaces; it rides in docker_args")
        env["DEFT_RUN_DIR"] = a.run_dir
    if a.baseline:
        env["DEFT_BASELINE"] = 1
    if a.name:
        if " " in a.name:
            p.error("--name cannot contain spaces; it rides in docker_args")
        env["DEFT_ROUND_NAME"] = a.name
    if a.gap is not None:
        # Reject it here rather than ten minutes later on the agent. No spaces
        # or quotes may reach docker_args -- the agent splits it on whitespace.
        for pair in a.gap.split(","):
            label, _, count = pair.partition(":")
            if not label.strip() or not count.strip().isdigit():
                p.error("--gap wants label:count pairs, e.g. "
                        "bridge:52,excess_solder:44 (got %r)" % pair)
        env["DEFT_GAP"] = a.gap.replace(" ", "")
    launch(a.stage, queue=a.queue, env=env)


if __name__ == "__main__":
    sys.exit(main())
