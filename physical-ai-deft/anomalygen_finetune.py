"""NVIDIA PAIDF AnomalyGen phase 0 + phase 1, as a ClearML task.

    checkpoints -> validate dataset -> generate config -> fine-tune

This is the stage that makes everything downstream possible. AnomalyGen ships
only ~2.9M trainable parameters -- a set of anomaly-token embeddings and a
2-layer MLP adapter -- which plug into a FROZEN Cosmos-Predict2 2B backbone.
Those 2.9M parameters are what learn "this is what a solder bridge looks like
on an IC pad", from 8 real examples. Everything after this is inference.

STAYING INSIDE NVIDIA'S SKILL LAYER
------------------------------------
The repo's own CLAUDE.md is explicit: `scripts/anomaly_gen/` and
`cosmos_predict2/` are upstream code an agent must not edit, and work belongs
in the helper layer. So this wrapper calls their helpers and nothing else --

    scripts/download_checkpoints.py     phase 0
    scripts/utilities/check.sh          phase 0 verification
    scripts/utilities/prepare_dataset_uc1.py
    scripts/utilities/generate_config.py
    scripts/utilities/launch_training.sh    phase 1

-- and confines itself to arranging the inputs, capturing the outputs, and
reporting to ClearML. If something here needs a change to how AnomalyGen
trains, it belongs in a config, not in this file.

WHY THE CHECKPOINTS LIVE ON A MOUNTED VOLUME
---------------------------------------------
Phase 0 pulls ~40 GB by default (2B base, T5-large, Cosmos-Guardrail1, RADIO,
NVDINOV2, DINOv2, SAM2, Qwen3-VL) and ~150 GB with the 14B path. The recipe
mounts a per-lab PVC at /cache, and the repo's `checkpoints/` is symlinked to
it -- NOT merely pointed at with a flag, because the generated config refers to
checkpoints by RELATIVE path (`t5_model_name: checkpoints/google-t5/t5-large`).
Point the downloader elsewhere without the symlink and phase 0 succeeds while
phase 1 fails looking for a T5 that is on disk the whole time.

CONTAINER
---------
nvcr.io/nvidia/paidf-anomalygen -- NVIDIA's own prebuilt image, so there is no
fork of their Dockerfile to maintain. Verified on the deft pool: torch
2.10.0+cu128, CUDA available, A10G visible, running as uid 10000 (which is why
the recipe sets fs_group=10000 -- without it the container cannot write to the
mounted cache).
"""
# NO `from __future__ import annotations` HERE.
#
# The clearml-agent patches the top of a script it runs remotely, which pushes
# a __future__ import below the first statement and makes it a SyntaxError:
#
#   SyntaxError: from __future__ imports must occur at the beginning of the file
#
# It is not needed anyway -- this runs on the image's Python 3.12, where
# `dict | None` is native. The sibling modules keep theirs because they are
# imported, not executed as the task script; only the entry point is patched.
import os
import subprocess
import sys

REPO_ROOT = "/workspace/paidf-anomalygen"
CACHE = os.environ.get("DEFT_CACHE", "/cache")
IMAGE = "nvcr.io/nvidia/paidf-anomalygen:1.0.1"

HF_DATASET = "nvidia/Cosmos-AnomalyGen-PCB-Dataset"


def _run(cmd: list[str], cwd: str = REPO_ROOT, env: dict | None = None) -> None:
    """Run a step, streaming its output, and fail the task if it fails.

    Streamed rather than captured on purpose: a fine-tune is long enough that a
    silent task with a wall of output at the end is useless to watch, and the
    ClearML console is the thing a reader of the lab guide is looking at.
    """
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if p.returncode != 0:
        raise SystemExit("step failed (%d): %s" % (p.returncode, " ".join(cmd)))


def _count_masks(dataset_dir: str) -> int:
    """Total training masks across every texture/defect.

    This is `num_sdg` for the VALIDATION testcase, and the number matters: with
    proportional allocation it puts each training submask in exactly once, and
    each defect type needs at least 3 entries for nn_score to be statistically
    meaningful. Our counts (62 / 16 / 8) clear that comfortably -- a dataset
    with a 1-2 example class would need num_sdg raised instead.
    """
    total = 0
    for tex in sorted(os.listdir(dataset_dir)):
        mdir = os.path.join(dataset_dir, tex, "mask")
        if not os.path.isdir(mdir):
            continue
        for cls in sorted(os.listdir(mdir)):
            d = os.path.join(mdir, cls)
            if os.path.isdir(d):
                n = len([f for f in os.listdir(d)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                print("  %s/%s: %d masks" % (tex, cls, n), flush=True)
                total += n
    if not total:
        raise SystemExit("no masks found under %s -- dataset prep did not "
                         "produce the expected <texture>/mask/<type>/ layout"
                         % dataset_dir)
    return total


def anomalygen_finetune(dataset_name="pcb-uc1",
                        model_size="2b",
                        num_gpus=1,
                        max_iter=2000,
                        ):
    """Phase 0 + phase 1. Returns the trained checkpoint directory."""
    from clearml import Task

    task = Task.current_task()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN not set. Phase 0 refuses to start without it, and the "
            "Cosmos-Predict2 weights are gated. It should arrive from the "
            "namespace `lab-credentials` Secret -- if it has not, fix the "
            "Secret rather than passing it as a task argument.")

    ckpt_dir = os.path.join(CACHE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # The generated config references checkpoints RELATIVELY, so the repo's own
    # `checkpoints/` has to BE the cache, not merely know where it is.
    link = os.path.join(REPO_ROOT, "checkpoints")
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        for entry in os.listdir(link):
            src, dst = os.path.join(link, entry), os.path.join(ckpt_dir, entry)
            if not os.path.exists(dst):
                os.replace(src, dst)
        os.rmdir(link)
    os.symlink(ckpt_dir, link)
    print("checkpoints/ -> %s" % ckpt_dir, flush=True)

    size_up = model_size.upper()

    # ---- phase 0: checkpoints ------------------------------------------
    print("=" * 66 + "\nPHASE 0 -- checkpoints (~40GB for the 2B path)\n" + "=" * 66,
          flush=True)
    _run([sys.executable, "-m", "scripts.download_checkpoints",
          "--checkpoint_dir", ckpt_dir,
          "--model_types", "text2image",
          "--model_sizes", size_up])
    # check.sh exits non-zero with a remediation list if anything is missing,
    # which is a far better failure than discovering it 20 minutes into a train.
    _run(["bash", "scripts/utilities/check.sh",
          "--checkpoint-dir", ckpt_dir, "--model-sizes", size_up])

    # ---- dataset --------------------------------------------------------
    # prepare_dataset_uc1.py fetches the HF bundle ITSELF (repo id is baked in)
    # and flattens a stray wrapper dir; it takes one positional output dir.
    print("=" * 66 + "\nDATASET\n" + "=" * 66, flush=True)
    dataset_dir = os.path.join(CACHE, "datasets", dataset_name)
    _run([sys.executable, "scripts/utilities/prepare_dataset_uc1.py", dataset_dir])

    defect_spec = os.path.join(dataset_dir, "defect_spec.jsonl")
    if not os.path.exists(defect_spec):
        raise SystemExit("no defect_spec.jsonl under %s" % dataset_dir)

    # ---- validation JSONL ----------------------------------------------
    # generate_config.py REQUIRES --validation-jsonl, and it is not something
    # you invent: it comes from prep_testcase.sh in `validation` mode with
    # num_sdg set to the TOTAL TRAINING MASK COUNT, which makes the allocation
    # proportional and puts every training submask in exactly once. Our counts
    # are 62 + 16 + 8, so each defect clears the >=3 floor nn_score needs to
    # mean anything.
    print("=" * 66 + "\nVALIDATION JSONL\n" + "=" * 66, flush=True)
    num_masks = _count_masks(dataset_dir)
    print("total training masks: %d" % num_masks, flush=True)
    val_jsonl = os.path.join(CACHE, "ag_inference", dataset_name, "validation.jsonl")
    os.makedirs(os.path.dirname(val_jsonl), exist_ok=True)
    _run(["bash", "scripts/utilities/prep_testcase.sh",
          "--name", "%s-val" % dataset_name,
          "--num-sdg", str(num_masks),
          "--dataset-dir", dataset_dir,
          "--amp-output-dir", os.path.join(CACHE, "ag_inference", dataset_name, "amp"),
          "--output-jsonl", val_jsonl,
          "--defect-spec", defect_spec,
          "--mode", "validation"])

    # ---- config ---------------------------------------------------------
    ag_config = os.path.join(REPO_ROOT, "ag_configs", "%s.yaml" % dataset_name)
    os.makedirs(os.path.dirname(ag_config), exist_ok=True)
    _run([sys.executable, "scripts/utilities/generate_config.py",
          "--name", dataset_name,
          "--dataset-dir", dataset_dir,
          "--defect-spec", defect_spec,
          "--validation-jsonl", val_jsonl,
          "--model-size", model_size,
          "--max-iter", str(max_iter),
          "--output", ag_config])
    if os.path.exists(ag_config):
        print("---- generated config ----", flush=True)
        print(open(ag_config).read(), flush=True)
        if task:
            task.upload_artifact("ag_config", ag_config)

    # ---- phase 1: fine-tune ---------------------------------------------
    print("=" * 66 + "\nPHASE 1 -- few-shot fine-tune\n" + "=" * 66, flush=True)
    _run(["bash", "scripts/utilities/launch_training.sh",
          "--ag-config", ag_config,
          "--num-gpus", str(num_gpus),
          "--model-size", model_size],
         env={"IMAGINAIRE_OUTPUT_ROOT": os.path.join(CACHE, "results")})

    out_root = os.path.join(CACHE, "results", "anomaly_gen", dataset_name)
    trained = None
    if os.path.isdir(out_root):
        runs = sorted(os.listdir(out_root))
        trained = os.path.join(out_root, runs[-1]) if runs else None

    print("=" * 66, flush=True)
    print("trained checkpoint dir:", trained, flush=True)
    print("=" * 66, flush=True)

    if task and trained:
        # Register it so the generation stage can address it by name instead of
        # by a path that only exists inside this lab's cache.
        task.upload_artifact("anomalygen_checkpoint_dir", trained)
    return {"checkpoint_dir": trained, "dataset_dir": dataset_dir,
            "ag_config": ag_config, "model_size": model_size}


if __name__ == "__main__":
    anomalygen_finetune()
