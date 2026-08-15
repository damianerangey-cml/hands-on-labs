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
    # has (measured twice), so it goes to the `gpu48` role rather than `gpu`.
    # If your 48 GB queue scales from zero, expect a cold start of several
    # minutes before this task starts -- the card is created when the task asks
    # for it. If you have no such queue at all, record that with
    # `deft.set_queues(gpu48=False)` and skip this stage; the other six phases
    # run on 24 GB.
    "finetune": ("anomalygen_finetune.py",
                 "AnomalyGen phase 1 -- few-shot fine-tune (48GB)",
                 "training", "gpu48"),
    "improve": ("anomalygen_improve.py",
                "AnomalyGen -- search, keep best, filter and regenerate",
                "data_processing", "gpu"),
    # Publish needs no GPU but runs on the GPU queue anyway, and the reason is
    # worth knowing: the CPU queue's pod template DELIBERATELY drops the
    # recipe's overrides (it keeps only privileged/tolerations and routes to the
    # shared `general` NodePool), so it has no model-cache mount. Publish reads
    # the generated images off that cache. On the CPU queue it fails with
    # "no SDG_result.csv" -- which reads like generation never ran, rather than
    # like the volume is missing.
    "publish": ("publish_synthetic.py",
                "Publish the survivors as the next HyperDataset version",
                "data_processing", "gpu"),
    "train": ("train_inspector.py", "Train the inspector", "training", "gpu"),
}


def launch(stage, queue=None, env=None, dry_run=False):
    """Create the task and enqueue it. With dry_run, create it and stop.

    A dry run is the cheapest useful test on a server nobody has run this on.
    It exercises everything that is easy to get wrong and invisible until a task
    fails ten minutes later -- the repository URL the agent will clone, the
    branch, the script path inside it, the image, and above all whether the
    queue for this role is known -- while spending no GPU, pulling no image and
    creating no data. What you get is a DRAFT task you can read in the UI and
    then enqueue or delete by hand.

    It is deliberately not a print-only preview: `Task.create` is where a bad
    repo URL or an unreachable script path actually surfaces, and a preview that
    skipped it would report success for a task that cannot run.
    """
    from clearml import Task

    script, name, ttype, default_queue = STAGES[stage]

    # RESOLVE THE QUEUE FIRST -- BEFORE creating anything.
    #
    # This used to sit after Task.create, and the ordering was a real bug: a
    # stage with no queue raised AFTER the task existed, leaving an orphaned
    # draft on the server for a stage that could never run. Every failed launch
    # littered one. Worse, the traceback made it look like nothing had been
    # created -- an agent reported "the stage never gets drafted" from the
    # exception alone, and was wrong, because the exception says nothing about
    # what happened on the line before it.
    #
    # `default_queue` is a KIND ("gpu"/"cpu"/"gpu48"), not a name. THERE ARE NO
    # QUEUE NAMES IN THIS REPOSITORY -- see deft.py. pick_queue() reads what has
    # been recorded for this server and raises a question rather than
    # enqueueing into a queue nobody serves, which would otherwise sit in
    # `queued` forever looking like a slow start.
    if queue:
        q = queue
    else:
        import deft
        q = deft.pick_queue(default_queue)

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

    print("task   %s" % task.id)
    print("repo   %s @ %s" % (REPO, BRANCH))
    print("script %s/%s" % (WORKDIR, script))
    print("image  %s" % IMAGE)
    print("queue  %s%s" % (q, "  (%s)" % default_queue))
    if dry_run:
        print("")
        print("DRY RUN -- not enqueued. The task is a draft; enqueue it from the")
        print("UI or delete it. Nothing was pulled and no GPU was spent.")
        print("read   %s" % task.get_output_log_web_page())
        return task.id
    Task.enqueue(task, queue_name=q)
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
                   help="improve/publish/evaluate: the run directory to act on. "
                        "For publish AFTER phases 5-7 this is the searched "
                        "bucket, runs/<id>/searched.")
    p.add_argument("--target", type=int, default=None,
                   help="publish stage only: per-class target (default 60)")
    p.add_argument("--run-id", default=None,
                   help="generate stage only: unique id for this generation "
                        "run. MUST differ per invocation -- reusing one makes "
                        "the next publish skip the round silently.")
    p.add_argument("--seed", type=int, default=None,
                   help="generate stage only: generation seed")
    p.add_argument("--baseline", action="store_true",
                   help="train stage only: the CONTROL -- real images only, no "
                        "synthetic. Run this once before generating anything; "
                        "without it there is nothing to compare later rounds to.")
    p.add_argument("--name", default=None,
                   help="train stage only: model name (default inspector, or "
                        "inspector-baseline with --baseline)")
    p.add_argument("--queue", default=None, help="override the stage's queue")
    p.add_argument("--dry-run", action="store_true",
                   help="create the task and STOP -- do not enqueue. Proves the "
                        "repo, branch, script path, image and queue all resolve "
                        "on this server, without spending a GPU or pulling an "
                        "image. Leaves a draft task you can read, then enqueue "
                        "or delete.")
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
    if a.run_id:
        if " " in a.run_id:
            p.error("--run-id cannot contain spaces; it rides in docker_args")
        env["DEFT_RUN_ID"] = a.run_id
    if a.seed is not None:
        env["DEFT_SEED"] = a.seed
    if a.target is not None:
        env["DEFT_TARGET"] = a.target
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
    launch(a.stage, queue=a.queue, env=env, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
