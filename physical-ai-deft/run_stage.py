"""Run a TAO / Cosmos container stage as a ClearML task instead of `docker run`.

This is the whole integration between NVIDIA's DEFT loop and ClearML.

NVIDIA's DEFT AOI skill executes every stage as a container on the agent's own
host, e.g.:

    docker run --rm --gpus all --ipc=host --shm-size=16g \\
        -v $WS:$WS -w /workspace/paidf-anomalygen $AG_IMAGE \\
        bash -lc "${ANOMALYGEN_SCRIPTS}/run_sdg.sh --checkpoint_dir ... --step ..."

A ClearML task already IS "run this container image with this command on a GPU":
the agent pulls the image, creates a pod from it, runs the command, and captures
console / status / metrics / artifacts. So the substitution is one call:

    run_stage(image=AG_IMAGE, command="${ANOMALYGEN_SCRIPTS}/run_sdg.sh ...",
              queue=deft.pick_queue("gpu"), stage="anomalygen", iteration=1)

The runbook (runbook/) is what tells the agent to call this instead of
`docker run`. Nothing else about NVIDIA's loop changes.

What moved from `docker run` flags into the lab's agent pod template
(orchestrator side, once per lab -- NOT per task):

    -v $WS:$WS  ........  the workspace PVC, mounted at the SAME absolute path
    --gpus all  ........  the GPU resource the queue's profile already requests
    --ipc=host --shm-size  an emptyDir-memory /dev/shm volume
    --user / chmod 777 ..  securityContext / fsGroup (AnomalyGen runs as uid 10000)

CREDENTIALS: never pass NGC_KEY / HF_TOKEN through `env` here -- task parameters
are readable by anyone with project access. Secrets are injected into task pods
from the lab namespace's k8s Secret by the agent's pod template. `env` is for
non-secret knobs only (HF_HUB_DISABLE_XET, PYTHONPATH, ...).

VERIFIED on lab1, 2026-08-05 (probe_parity.py): an arbitrary public image
(python:3.11-slim -- not a CUDA image) ran as a ClearML task with a custom
command and was given a real GPU: NVIDIA A10G, 5757 MiB, driver 535.216.01.
The 5757 MiB is CFGI enforcing the 0.25 slice, so stages really can ask for a
fraction of a card per queue. The image's own python was used
(CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1) with only the clearml SDK layered on
by the container setup script -- exactly the combination below. Do not drop
either half: without the skip flag the agent builds a venv inside NVIDIA's
image; without the setup script `import clearml` fails and the stage dies
before it reports anything.

ASCII-only.
"""
import time

from clearml import Task

REPO = "https://github.com/damianerangey-cml/hands-on-labs.git"
BRANCH = "main"
WORKDIR = "physical-ai-deft"
ENTRY = "physical-ai-deft/stage_entry.py"

PROJECT = "Physical AI Inspection"
WORKSPACE = "/workspace/deft"

# Terminal ClearML task states.
_DONE = ("completed", "failed", "stopped", "closed")


def run_stage(
    image,
    command,
    queue,
    stage,
    iteration="baseline",
    project=PROJECT,
    workspace=WORKSPACE,
    workdir=None,
    env=None,
    tags=None,
    wait=True,
    poll_s=15,
    timeout_s=None,
):
    """Submit one DEFT stage as a ClearML task and (by default) wait for it.

    image     -- the container the stage runs in (a TAO or paidf-anomalygen image)
    command   -- the shell command to run INSIDE that container, verbatim from
                 NVIDIA's stage reference (everything after `bash -lc`)
    queue     -- which ClearML queue, i.e. how much GPU this stage gets
                 (resolve it with deft.pick_queue -- no name is built in)
    stage     -- DEFT stage name: train | evaluate | rca | anomalygen |
                 routing | data_mining
    iteration -- "baseline", "iter1", "iter2", ...

    Returns the ClearML Task. With wait=False it returns as soon as the task is
    queued (use poll(task) later) -- that is how several stages can run at once.
    """
    name = "%s/%s" % (iteration, stage)
    task = Task.create(
        project_name=project,
        task_name=name,
        task_type=_TASK_TYPES.get(stage, "custom"),
        repo=REPO,
        branch=BRANCH,
        script=ENTRY,
        working_directory=WORKDIR,
        # The stage runs in NVIDIA's image, which brings its own python and
        # dependencies. We only need the clearml SDK on top of it, installed by
        # the container setup script below -- NOT a fresh venv.
        packages=["clearml"],
        docker=image,
        docker_args=_docker_args(env),
        docker_bash_setup_script=_setup_script(),
        add_task_init_call=False,   # stage_entry.py calls Task.init itself
    )

    # The exact container command lands in the task's Configuration, so the
    # Execution tab is a full receipt: image + command + workspace. This is also
    # what makes "look before you run" work in the lab guide.
    task.set_parameters({
        "stage/command": command,
        "stage/name": stage,
        "stage/iteration": iteration,
        "stage/workspace": workspace,
        "stage/workdir": workdir or "",
    })
    if tags:
        task.set_tags(list(tags))

    Task.enqueue(task, queue_name=queue)
    print("[%s] queued on %s (task %s)" % (name, queue, task.id))
    print("       image:   %s" % image)
    print("       command: %s" % command)
    if not wait:
        return task
    return poll(task, poll_s=poll_s, timeout_s=timeout_s)


def poll(task, poll_s=15, timeout_s=None):
    """Block until a submitted stage reaches a terminal state, printing the
    status transitions so the agent (and the reader watching the terminal) can
    follow along. Console output is captured by ClearML either way -- the task
    page is the durable record; this is just the live view."""
    name = task.name
    started = time.time()
    last = None
    while True:
        status = task.get_status()
        if status != last:
            print("[%s] %s" % (name, status))
            last = status
        if status in _DONE:
            break
        if timeout_s and (time.time() - started) > timeout_s:
            raise TimeoutError(
                "stage %s still %s after %ss (task %s)" % (name, status, timeout_s, task.id))
        time.sleep(poll_s)

    if status != "completed":
        tail = console_tail(task, lines=40)
        raise RuntimeError(
            "stage %s ended '%s' (task %s)\n--- last console ---\n%s"
            % (name, status, task.id, tail))
    return task


def console_tail(task, lines=40):
    """Last N console lines of a stage -- what the agent should quote verbatim
    when a stage fails, instead of guessing at the cause."""
    try:
        out = task.get_reported_console_output(number_of_reports=lines)
        return "\n".join(out) if isinstance(out, (list, tuple)) else str(out)
    except Exception as exc:
        return "(console unavailable: %s)" % exc


def _docker_args(env=None):
    """Container args. The GPU, the workspace mount, /dev/shm and the uid all
    come from the lab's agent pod template -- see the module docstring. What is
    left is the handful of non-secret env vars NVIDIA's containers expect."""
    args = [
        # Use the image's own python; do not build a venv inside a TAO image.
        "-e", "CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1",
        "-e", "HF_HUB_DISABLE_XET=1",
    ]
    for key, val in (env or {}).items():
        args += ["-e", "%s=%s" % (key, val)]
    return args


def _setup_script():
    """Runs inside NVIDIA's container before the stage. Only job: make the
    clearml SDK importable so stage_entry.py can report back."""
    return [
        "python3 -m pip install -q --no-input clearml || pip install -q clearml",
    ]


_TASK_TYPES = {
    "train": "training",
    "evaluate": "testing",
    "rca": "qc",
    "anomalygen": "data_processing",
    "routing": "data_processing",
    "data_mining": "data_processing",
}
