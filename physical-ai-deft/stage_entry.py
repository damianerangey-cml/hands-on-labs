"""In-container entrypoint for one DEFT stage. Runs inside NVIDIA's image.

The ClearML agent pulls the TAO / AnomalyGen image, creates the pod, and runs
this file. All it does is:

  1. attach to the task the agent is running,
  2. exec the stage command NVIDIA's reference specified, streaming its output
     (ClearML captures stdout, so the console tab IS the container log),
  3. hand the output to the hooks, which turn the stage's own summary line into
     ClearML scalars and register whatever the stage produced.

Step 3 is where the platform earns its keep: NVIDIA's loop would have written
one line to results/loop_log.jsonl and moved on. Same line, but it also becomes
a scalar you can plot across iterations and a dataset/model with lineage.

ASCII-only.
"""
import os
import subprocess
import sys
import time

from clearml import Task

import clearml_deft_hooks as hooks

DEFAULTS = {
    "command": "",
    "name": "unknown",
    "iteration": "baseline",
    "workspace": "/workspace/deft",
    "workdir": "",
}


def main():
    task = Task.current_task() or Task.init(
        project_name="Physical AI Inspection", task_name="DEFT stage")

    params = dict(DEFAULTS)
    task.connect(params, name="stage")
    command = str(params["command"]).strip()
    if not command:
        raise SystemExit(
            "No stage/command parameter set. This task is submitted by "
            "run_stage(); it is not meant to be launched by hand.")

    stage = str(params["name"])
    iteration = str(params["iteration"])
    workspace = str(params["workspace"])
    workdir = str(params["workdir"]) or None

    print("=" * 70)
    print("DEFT stage : %s (%s)" % (stage, iteration))
    print("workspace  : %s" % workspace)
    print("workdir    : %s" % (workdir or os.getcwd()))
    print("command    : %s" % command)
    print("=" * 70)
    sys.stdout.flush()

    started = time.time()
    rc, output = _run(command, cwd=workdir)
    duration = int(time.time() - started)

    print("-" * 70)
    print("stage %s exited %s after %ss" % (stage, rc, duration))

    # Report even on failure -- a failed stage with its metrics attached is far
    # more useful to the next iteration than a task that just says "failed".
    try:
        hooks.report_stage(
            task, stage=stage, iteration=iteration, status="ok" if rc == 0 else "error",
            duration_sec=duration, output=output, workspace=workspace)
    except Exception as exc:
        print("hook report_stage failed (non-fatal):", exc)

    if rc != 0:
        raise SystemExit(rc)


def _run(command, cwd=None):
    """Run the stage command, echoing output live AND keeping it so the hooks
    can parse the stage's summary out of it."""
    proc = subprocess.Popen(
        ["bash", "-lc", command], cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line.rstrip("\n"))
    proc.wait()
    return proc.returncode, lines


if __name__ == "__main__":
    main()
