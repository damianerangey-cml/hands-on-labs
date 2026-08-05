"""Turn NVIDIA's DEFT loop into a tracked, comparable, auditable run.

NVIDIA's loop already records itself -- into three files in one directory on one
box: results/deft_state.json, results/loop_log.jsonl (one line per stage) and a
per-iteration DEFT_Loop_Report.html. That is a hand-rolled experiment tracker:
single-host, single-user, no cross-run comparison, no lineage, and it dies with
the box.

These hooks keep every one of those files exactly as NVIDIA writes them (their
state machine depends on it -- disk is canonical, `seq` is computed from the live
tail) and mirror the same information into ClearML, where it survives, compares
and carries lineage.

Three things happen here:

  1. report_stage()      -- per-stage metrics onto the stage's own task, and the
                            headline KPI onto the loop controller task, so
                            FAR-vs-target across iterations draws itself.
  2. register_*()        -- AnomalyGen output, the assembled training set and the
                            winning checkpoint become versioned Datasets and a
                            registered Model, parented so "which synthetic images
                            earned the FAR drop?" is a click, not a grep.
  3. mirror_loop_log()   -- run at loop end, after NVIDIA's align_token_usage.py
                            has backfilled per-stage token counts, so agent
                            tokens can be plotted beside GPU seconds. That chart
                            -- what an autonomous improvement loop actually cost
                            -- is the number nobody else is showing.

ASCII-only.
"""
import json
import os
import re
from pathlib import Path

from clearml import Dataset, OutputModel, Task

PROJECT = "Physical AI Inspection"
LOOP_TASK_NAME = "DEFT Loop"

# Written by loop_task() into the shared workspace so every stage pod -- which is
# a separate container on a separate node -- can find the controller task.
_LOOP_POINTER = "results/clearml_loop_task.txt"
_MIRROR_MARK = "results/.clearml_mirrored_seq"

# Stage summaries are short human strings, e.g.
#   "FAR=52.0% threshold=0.31"
#   "FAR 6.34% -> 3.11% (target <0.5%)"
#   "SDG: requested=20, AMP-allocated=8, generated=8 by type"
# Pull every number we can name, ignore the rest.
_KV = re.compile(r"([A-Za-z][A-Za-z0-9_\-]*)\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*(%?)")
_BARE_PCT = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s+(-?\d+(?:\.\d+)?)%")
# "<metric> 6.34% -> 3.11%": the number after the arrow is the current value.
_ARROW = re.compile(r"\s*%?\s*-+>\s*(-?\d+(?:\.\d+)?)")

# Only these become scalars; everything else stays in the summary text. Keeps
# the metrics list readable instead of a wall of parsed noise.
_METRICS = {
    "far", "recall", "precision", "threshold", "loss", "final_train_loss",
    "accuracy", "requested", "generated", "mined", "epochs",
}


# --------------------------------------------------------------------------
# loop controller
# --------------------------------------------------------------------------
def loop_task(workspace, kpi_target=None, create=True):
    """Get (or create) the task that represents THE WHOLE DEFT RUN.

    Stage tasks are the detail; this one is the story: KPI per iteration, cost,
    and the parent every stage hangs off. Its id is written into the shared
    workspace so stage pods can find it without being told."""
    ws = Path(workspace)
    pointer = ws / _LOOP_POINTER
    if pointer.exists():
        task_id = pointer.read_text(encoding="utf-8").strip()
        if task_id:
            try:
                return Task.get_task(task_id=task_id)
            except Exception:
                pass
    if not create:
        return None

    task = Task.init(project_name=PROJECT, task_name=LOOP_TASK_NAME,
                     task_type="controller", reuse_last_task_id=False)
    if kpi_target:
        task.set_parameters({"deft/kpi_target": kpi_target})
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(task.id, encoding="utf-8")
    print("DEFT loop controller task:", task.id)
    return task


def _loop_id(workspace):
    pointer = Path(workspace) / _LOOP_POINTER
    return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None


def _iter_index(iteration):
    """'baseline' -> 0, 'iter3' -> 3. The x-axis of the KPI chart."""
    if not iteration or iteration == "baseline":
        return 0
    digits = "".join(c for c in str(iteration) if c.isdigit())
    return int(digits) if digits else 0


# --------------------------------------------------------------------------
# per-stage reporting
# --------------------------------------------------------------------------
def report_stage(task, stage, iteration, status, duration_sec, output, workspace):
    """Called by stage_entry.py once the container command has finished.

    `output` is the stage's console lines. NVIDIA's loop distils the same thing
    into one summary string; we parse the numbers out of it so they become
    plottable instead of grep-able."""
    logger = task.get_logger()
    text = "\n".join(output[-80:]) if isinstance(output, (list, tuple)) else str(output)
    metrics = extract_metrics(text)
    step = _iter_index(iteration)

    logger.report_single_value("duration_sec", duration_sec)
    for key, val in metrics.items():
        logger.report_single_value(key, val)
        logger.report_scalar(title="stage", series=key, value=val, iteration=step)
    task.set_tags(list(task.get_tags() or []) + ["deft", stage, iteration])

    print("stage metrics:", metrics if metrics else "(none parsed)")

    # Mirror the headline KPI onto the loop controller so the across-iteration
    # chart exists while the loop is still running.
    loop_id = _loop_id(workspace)
    if loop_id and metrics:
        try:
            loop = Task.get_task(task_id=loop_id)
            llog = loop.get_logger()
            for key in ("far", "recall", "final_train_loss", "loss"):
                if key in metrics:
                    llog.report_scalar(title="kpi", series=key.upper(),
                                       value=metrics[key], iteration=step)
            llog.report_scalar(title="cost", series="gpu_seconds",
                               value=float(duration_sec), iteration=step)
        except Exception as exc:
            print("loop mirror skipped:", exc)

    if status != "ok":
        task.set_tags(list(task.get_tags() or []) + ["failed-stage"])
    return metrics


def extract_metrics(text):
    """Pull named numbers out of a stage's output. Deliberately tolerant: TAO
    stages print in several shapes and we would rather miss a metric than crash
    a stage over a parse.

    One shape needs care. DEFT's own status line reports an improvement as
    `FAR 6.34% -> 3.11% (target <0.5%)`. Naive parsing takes 6.34 -- the value
    BEFORE this iteration -- and the KPI chart then plots the wrong number, in
    the wrong direction. When a metric is followed by an arrow, the value after
    the arrow is the current one."""
    found = {}
    for match in list(_KV.finditer(text)) + list(_BARE_PCT.finditer(text)):
        name = match.group(1).lower()
        if name not in _METRICS:
            continue
        try:
            value = float(match.group(2))
        except (TypeError, ValueError):
            continue
        after = _ARROW.match(text[match.end():])
        if after:
            try:
                value = float(after.group(1))
            except (TypeError, ValueError):
                pass
        found[name] = value
    return found


# --------------------------------------------------------------------------
# data + model lineage
# --------------------------------------------------------------------------
def register_synthetic(sdg_dir, iteration, parents=None):
    """AnomalyGen's generated defects become a versioned Dataset.

    This is the one that makes the audit answerable: the reader can OPEN the
    dataset and look at the images the model trained on."""
    return _register_dataset(
        name="pcb-inspection-synthetic-%s" % iteration, path=sdg_dir,
        parents=parents, tags=["synthetic", "anomalygen", iteration])


def register_training_set(csv_dir, iteration, parents=None):
    """The assembled training set for one iteration -- real + mined + synthetic.
    Parented on both sources, which is what makes 'was any of it synthetic?'
    a click."""
    return _register_dataset(
        name="pcb-inspection-train-%s" % iteration, path=csv_dir,
        parents=parents, tags=["training-set", iteration])


def _register_dataset(name, path, parents=None, tags=None):
    path = Path(path)
    if not path.exists():
        print("dataset source missing, skipping registration:", path)
        return None
    ds = Dataset.create(dataset_name=name, dataset_project=PROJECT,
                        parent_datasets=parents or None)
    ds.add_files(str(path))
    ds.upload()
    ds.finalize()
    if tags:
        try:
            ds.tags = list(tags)
        except Exception:
            pass
    print("registered dataset %s (%s) from %s" % (name, ds.id, path))
    return ds.id


def register_checkpoint(task, ckpt_path, iteration, far=None):
    """The iteration's checkpoint, registered against the task that produced it
    -- so the model in the registry carries its own provenance."""
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print("checkpoint missing, skipping registration:", ckpt)
        return None
    out = OutputModel(task=task, name="pcb-changenet-%s" % iteration, framework="PyTorch")
    tags = ["changenet", iteration]
    if far is not None:
        tags.append("FAR=%s" % far)
    try:
        out.tags = tags
    except Exception:
        pass
    out.update_weights(weights_filename=str(ckpt), auto_delete_file=False)
    print("registered model %s (%s)" % (out.name, out.id))
    return out.id


# --------------------------------------------------------------------------
# loop-end mirror (this is where the cost chart comes from)
# --------------------------------------------------------------------------
def mirror_loop_log(workspace):
    """Replay results/loop_log.jsonl onto the loop controller task.

    Run this at loop end, AFTER NVIDIA's scripts/align_token_usage.py has
    backfilled real per-stage token counts -- that is the only moment the token
    numbers exist. Idempotent: remembers the last `seq` it mirrored, so calling
    it again after another iteration only adds what is new."""
    ws = Path(workspace)
    log_path = ws / "results" / "loop_log.jsonl"
    if not log_path.exists():
        print("no loop_log.jsonl at", log_path, "- nothing to mirror")
        return 0
    loop_id = _loop_id(workspace)
    if not loop_id:
        print("no loop controller task recorded - call loop_task() first")
        return 0

    mark = ws / _MIRROR_MARK
    last = int(mark.read_text(encoding="utf-8").strip() or 0) if mark.exists() else 0
    loop = Task.get_task(task_id=loop_id)
    logger = loop.get_logger()

    mirrored, highest = 0, last
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        seq = int(entry.get("seq", 0))
        if seq <= last:
            continue
        step = _iter_index(entry.get("iter"))

        for key, val in extract_metrics(str(entry.get("summary", ""))).items():
            logger.report_scalar(title="kpi", series=key.upper(), value=val, iteration=step)
        if entry.get("duration_sec") is not None:
            logger.report_scalar(title="cost", series="gpu_seconds",
                                 value=float(entry["duration_sec"]), iteration=step)

        # The signature chart: what the autonomous loop cost in agent tokens,
        # on the same x-axis as the GPU seconds it spent.
        tokens = entry.get("tokens") or {}
        total = sum(float(tokens.get(k, 0) or 0)
                    for k in ("input", "output", "cache_read", "cache_create"))
        if total:
            logger.report_scalar(title="cost", series="agent_tokens",
                                 value=total, iteration=step)

        mirrored += 1
        highest = max(highest, seq)

    # NVIDIA's own artifacts stay the source of truth -- attach them so the
    # ClearML record and the loop's native record never diverge.
    for artifact in ("deft_state.json", "loop_log.jsonl", "DEFT_Loop_Report.html"):
        src = ws / "results" / artifact
        if src.exists():
            try:
                loop.upload_artifact(name=artifact, artifact_object=str(src))
            except Exception as exc:
                print("artifact upload skipped (%s): %s" % (artifact, exc))

    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(str(highest), encoding="utf-8")
    print("mirrored %s new loop_log entries onto %s" % (mirrored, loop_id))
    return mirrored


if __name__ == "__main__":
    # `python clearml_deft_hooks.py <workspace>` -- the loop-end mirror, so the
    # runbook can call it as one line without importing anything.
    import sys
    mirror_loop_log(sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "DEFT_WORKSPACE", "/workspace/deft"))
