"""Shared setup every AnomalyGen task needs.

Exists because the same mistake was made three times.

AnomalyGen resolves checkpoints by RELATIVE path. The generated training config
says `t5_model_name: checkpoints/google-t5/t5-large`; the evaluator loads
`checkpoints/nvidia/C-RADIO-V3/model.safetensors`; SDG reads the backbone the
same way. All of it resolves against the repo root, so the repo's `checkpoints/`
has to BE the cache -- pointing a --checkpoint_dir flag at the cache is not
enough, and the failure comes much later and looks unrelated:

    FileNotFoundError: No such file or directory:
      checkpoints/nvidia/C-RADIO-V3/model.safetensors

...for a file that is on disk the whole time. Put the symlink in one place and
have every task call it.
"""
import os
import subprocess
import sys

# WHERE NVIDIA'S CODE LIVES INSIDE THEIR IMAGE. Overridable because it is a
# property of the image tag, not of this lab -- a different paidf-anomalygen
# build (or a site that rebuilt it from the Apache-2.0 source) can put it
# somewhere else, and then every stage fails on a relative script path with a
# bare "No such file or directory" that says nothing about why.
REPO_ROOT = os.environ.get("DEFT_AG_REPO_ROOT", "/workspace/paidf-anomalygen")
CACHE = os.environ.get("DEFT_CACHE", "/cache")


def link_checkpoints(cache=None, repo_root=REPO_ROOT):
    """Make <repo>/checkpoints resolve to the shared cache. Returns its path.

    Idempotent, and safe against whatever the image shipped: an existing real
    directory has its contents moved into the cache once before being replaced
    by the link, so nothing baked into the image is lost.
    """
    ckpt_dir = os.path.join(cache or CACHE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    link = os.path.join(repo_root, "checkpoints")

    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(ckpt_dir):
            return ckpt_dir
        os.unlink(link)
    elif os.path.isdir(link):
        for entry in os.listdir(link):
            src, dst = os.path.join(link, entry), os.path.join(ckpt_dir, entry)
            if not os.path.exists(dst):
                os.replace(src, dst)
        os.rmdir(link)

    os.symlink(ckpt_dir, link)
    print("checkpoints/ -> %s" % ckpt_dir, flush=True)
    return ckpt_dir


def run(cmd, cwd=REPO_ROOT, env=None):
    """Run a pipeline step, streaming output, failing the task if it fails.

    Streamed rather than captured: these stages are long enough that a silent
    task with a wall of output at the end is useless to watch, and the ClearML
    console is what a reader of the lab guide is looking at.
    """
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if p.returncode != 0:
        raise SystemExit("step failed (%d): %s" % (p.returncode, " ".join(cmd)))


def ensure_dataset(dataset_name, cache=None):
    """NVIDIA's prepared dataset tree at <cache>/datasets/<name>. Idempotent.

    THE SAME MISTAKE, A FOURTH TIME. This directory used to be a side effect of
    `anomalygen_generate`, which prepared it lazily on its way to placing masks.
    Every stage that came later found it already there and nobody noticed the
    dependency -- until the loop ran on a cold cache and ROUND 0, the real-only
    baseline, crashed before any generation had happened:

        FileNotFoundError: '/cache/datasets/pcb-uc1'

    It is not generation's private working directory. It is the REAL DATA --
    `train_inspector` reads its held-out images from here, and the evaluator and
    phases 5-7 pass it as `--real-path`. So it is shared setup, which is what
    this module is for. Anything that needs it calls this and stops caring who
    got there first.
    """
    d = os.path.join(cache or CACHE, "datasets", dataset_name)
    # `defect_spec.jsonl` is the completeness marker: it is the last thing a
    # successful prepare leaves and the first thing every consumer reads.
    if os.path.exists(os.path.join(d, "defect_spec.jsonl")):
        return d
    import shutil
    if os.path.isdir(d):
        # Present but incomplete = a previous prepare died mid-move, or ran
        # twice. NVIDIA's prepare script is NOT idempotent -- re-moving a
        # directory into a populated tree nests it (IC/IC) and every later
        # stage trips over the wreckage. Half a dataset is worse than none.
        print("removing incomplete dataset tree at %s" % d, flush=True)
        shutil.rmtree(d)
    tmp = d + ".preparing"
    shutil.rmtree(tmp, ignore_errors=True)
    print("preparing NVIDIA's dataset at %s" % d, flush=True)
    run([sys.executable, "scripts/utilities/prepare_dataset_uc1.py", tmp])
    if not os.path.exists(os.path.join(tmp, "defect_spec.jsonl")):
        raise SystemExit("prepare finished without defect_spec.jsonl -- refusing "
                         "to install an incomplete dataset tree")
    # Atomic: consumers either see the whole prepared tree or none of it, even
    # if two stages race on a cold cache.
    os.makedirs(os.path.dirname(d), exist_ok=True)
    os.rename(tmp, d)
    return d


def require_hf_token():
    """Every stage needs it; none should accept it as a task argument."""
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN not set. It should arrive from your platform's "
            "secret store -- fix that rather than passing it "
            "as a task argument, which would write it onto the task record.")


def bind_task():
    """The running ClearML task, bound EXPLICITLY -- or None outside a task.

    Every stage used a bare `Task.current_task()`, which happens to be set
    when the agent bootstraps a launch.py-created task and is None on a task
    created any other way -- including `tasks.clone` over the API. A cloned
    stage then does all its work and silently reports nothing: no scalars, no
    debug samples, no registered model. The work succeeded, the record lied
    by omission. `Task.init()` under an agent binds the executing task
    regardless of how it was created; outside one (CLEARML_TASK_ID unset) we
    return None rather than minting a stray task from somebody's laptop.
    """
    from clearml import Task
    t = Task.current_task()
    if t is not None:
        return t
    if os.environ.get("CLEARML_TASK_ID"):
        return Task.init()
    return None
