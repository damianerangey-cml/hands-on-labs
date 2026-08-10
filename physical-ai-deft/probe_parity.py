"""Container-parity probe: does a ClearML task give a container what `docker run` gave it?

Run this ONCE against a lab before writing or debugging any DEFT stage. It
submits a throwaway task through the real run_stage() path and reports what the
container actually got, so the answer comes from the cluster instead of from
assumptions.

NVIDIA's stages run as:

    docker run --gpus all --ipc=host --shm-size=16g -v $WS:$WS <image> <command>

Every one of those flags has to be reproduced by the lab's agent pod template.
The probe checks all four, plus the two credentials the real images need:

    GPU        nvidia-smi works, and reports a card
    workspace  the shared PVC is visible at the SAME absolute path
    /dev/shm   big enough for dataloader workers (16g in NVIDIA's command)
    uid/gid    AnomalyGen runs as uid 10000 and writes into mounted dirs
    NGC/HF     the secrets are present in the pod env (values never printed)

Usage (credentials come from the lab's provision result):

    export CLEARML_API_HOST=https://api.your-clearml.example
    export CLEARML_WEB_HOST=https://app.your-clearml.example
    export CLEARML_FILES_HOST=https://files.your-clearml.example
    export CLEARML_API_ACCESS_KEY=...
    export CLEARML_API_SECRET_KEY=...
    python probe_parity.py --queue <your GPU queue>

A FAIL here is not a bug in the probe -- it is the pod-template item the lab
recipe still has to set. That is the point.

ASCII-only.
"""
import argparse
import os
import sys

from run_stage import WORKSPACE, console_tail, run_stage

# Public image: the probe must work before any nvcr.io entitlement exists, so it
# deliberately does not use a TAO image. It answers "can a ClearML task run an
# arbitrary container on a GPU", which is the thing under test.
PROBE_IMAGE = os.environ.get("PROBE_IMAGE", "nvidia/cuda:12.3.0-base-ubuntu22.04")

# One shell command; each line prints a PROBE: marker the parser picks up. Note
# the secrets are only tested for PRESENCE -- never echoed.
PROBE_CMD = r"""
echo "PROBE:uid=$(id -u) gid=$(id -g)"
echo "PROBE:shm=$(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
echo "PROBE:workspace_exists=$([ -d "$DEFT_WORKSPACE" ] && echo yes || echo no)"
echo "PROBE:workspace_writable=$(touch "$DEFT_WORKSPACE/.probe" 2>/dev/null && echo yes || echo no)"
rm -f "$DEFT_WORKSPACE/.probe" 2>/dev/null
echo "PROBE:ngc_key_present=$([ -n "$NGC_KEY" ] && echo yes || echo no)"
echo "PROBE:hf_token_present=$([ -n "$HF_TOKEN" ] && echo yes || echo no)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "PROBE:gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "PROBE:vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  echo "PROBE:driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  echo "PROBE:gpu=NONE (nvidia-smi missing)"
fi
"""

# What DEFT needs, and how to fix it if the probe says otherwise.
EXPECTATIONS = [
    ("gpu", "a GPU is visible", "queue is not a GPU queue, or the pod got no nvidia.com/gpu"),
    ("workspace_exists", "yes", "mount the workspace PVC at the same absolute path in the agent pod template"),
    ("workspace_writable", "yes", "set fsGroup on the pod template (AnomalyGen runs as uid 10000)"),
    ("shm", ">=1G", "add an emptyDir-memory /dev/shm volume (replaces --ipc=host --shm-size=16g)"),
    ("ngc_key_present", "yes", "project the lab namespace NGC secret into task pods"),
    ("hf_token_present", "yes", "project the lab namespace HF secret into task pods"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="0.25XGPU", help="GPU queue to probe")
    ap.add_argument("--workspace", default=WORKSPACE, help="expected workspace mount path")
    ap.add_argument("--image", default=PROBE_IMAGE)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print("probing queue=%s image=%s workspace=%s" % (args.queue, args.image, args.workspace))
    task = run_stage(
        image=args.image, command=PROBE_CMD, queue=args.queue,
        stage="parity-probe", iteration="probe", workspace=args.workspace,
        env={"DEFT_WORKSPACE": args.workspace}, tags=["probe"],
        wait=False)

    from run_stage import poll
    try:
        poll(task, poll_s=10, timeout_s=args.timeout)
    except RuntimeError as exc:
        # A non-zero exit is still a useful probe -- the console has the answers.
        print(exc)

    found = _parse(console_tail(task, lines=200))
    _report(found, task)
    return 0 if found.get("gpu", "").upper().find("NONE") < 0 else 1


def _parse(text):
    """Pull `PROBE:key=value` markers out of the console. A line may carry more
    than one pair (uid/gid), but values can also contain spaces ("NVIDIA A10G"),
    so only split on whitespace when every token looks like a k=v pair."""
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if "PROBE:" not in line:
            continue
        payload = line.split("PROBE:", 1)[1].strip()
        tokens = payload.split()
        if len(tokens) > 1 and all("=" in tok for tok in tokens):
            pairs = tokens
        else:
            pairs = [payload]
        for pair in pairs:
            key, _, val = pair.partition("=")
            if key:
                found[key.strip()] = val.strip()
    return found


def _report(found, task):
    print()
    print("=" * 72)
    print("CONTAINER PARITY -- what the task pod actually gave the container")
    print("=" * 72)
    if not found:
        print("No PROBE: markers found. Read the task console directly:")
        print("  ", task.get_output_log_web_page())
        return
    for key, val in sorted(found.items()):
        print("  %-22s %s" % (key, val))
    print("-" * 72)
    print("DEFT requirements:")
    for key, want, fix in EXPECTATIONS:
        got = found.get(key, "(not reported)")
        ok = _ok(key, got)
        print("  [%s] %-22s want %-8s got %s" % ("PASS" if ok else "FAIL", key, want, got))
        if not ok:
            print("        fix: %s" % fix)
    print("=" * 72)
    print("task:", task.get_output_log_web_page())


def _ok(key, got):
    got = (got or "").strip()
    if key == "gpu":
        return bool(got) and "NONE" not in got.upper()
    if key == "shm":
        # df prints e.g. "64M" or "16G"; anything in gigabytes passes.
        return got.upper().endswith("G") or got.upper().endswith("T")
    return got == "yes"


if __name__ == "__main__":
    sys.exit(main())
