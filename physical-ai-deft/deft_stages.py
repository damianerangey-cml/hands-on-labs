"""The DEFT stages, expressed as ClearML tasks.

One function per stage of NVIDIA's DEFT AOI loop. Each is a thin wrapper around
run_stage(): the command string is exactly what NVIDIA's stage reference tells
you to run inside the container, and run_stage() puts it on a queue instead of
on the agent's own docker daemon.

WHERE THE COMMANDS COME FROM. Two of these stages are transcribed directly from
NVIDIA's published references (anomalygen -- references/paidf-anomalygen.md;
visual changenet train/evaluate/inference -- the direct `docker run` path named
in the DEFT SKILL.md). The rest -- rca, routing, data_mining -- are owned by
their own skills whose flags we do NOT restate here: the agent reads the
matching reference at runtime and passes the command through `stage()`. That is
deliberate. Restating another skill's CLI is how a wrapper silently goes stale.

GPU SIZING. This lab runs stages on a WHOLE A10G (`1XGPU`), not a CFGI fraction.
TAO and Cosmos containers want a full card, and a whole-card pool is free to run
whatever CUDA version the images need -- TAO 6.26.3 is CUDA 13.0, which no
fractional pool of ours can serve. The trade: one stage at a time per lab.

ASCII-only.
"""
import os

from run_stage import WORKSPACE, run_stage

# Container images. Resolved by NVIDIA's own helper in a real install:
#   scripts/resolve_tao_image.py --model visual_changenet --action train
#   scripts/resolve_versions_key.py images.metropolis_sdg.paidf_anomalygen
# Pinned via env so a lab can be held on a known-good pair.
#
# ENTITLEMENT (measured 2026-08-05). The PUBLIC TAO image below is reachable with
# an ordinary NGC key. The two registries NVIDIA's skill defaults to are NOT:
#   nvcr.io/nvstaging/tao/*                      -> 403 Forbidden
#   nvcr.io/nv-metropolis-dev/metropolis-sdg/*   -> 403 Forbidden  (AnomalyGen)
# So the ChangeNet half of the loop runs today and the Cosmos AnomalyGen half
# needs an entitlement grant from NVIDIA. Building it ourselves is not an option:
# NVIDIA/physical-ai-data-factory publishes OSMO workflow configs, not AnomalyGen
# source. Override via env once access lands.
#
# TAO tag -> CUDA: 5.5.0 = 12.4, 6.0.0/6.25.7 = 12.8, 6.25.11/6.26.3 = 13.0,
# 7.1.0 = 13.2. Pin the NodePool's driver to match the tag you choose.
TAO_IMAGE = os.environ.get("TAO_PYT_IMAGE", "nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt")
AG_IMAGE = os.environ.get("AG_IMAGE", "nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-anomalygen")

# The AnomalyGen container sets ANOMALYGEN_SCRIPTS itself and expects to run
# from its own source tree -- see NVIDIA's reference. Do not export it host-side.
AG_WORKDIR = "/workspace/paidf-anomalygen"

# Whole card for every GPU stage (see module docstring). QUEUE_SMALL is kept as a
# separate name so a future fractional-capable lab can point it elsewhere without
# touching call sites; today it resolves to the same full-GPU queue.
QUEUE_FULL = os.environ.get("DEFT_QUEUE_FULL", "1XGPU")
QUEUE_SMALL = os.environ.get("DEFT_QUEUE_SMALL", QUEUE_FULL)
QUEUE_CPU = os.environ.get("DEFT_QUEUE_CPU", "1XCPU")


# --------------------------------------------------------------------------
# generic passthrough -- for stages whose CLI belongs to another skill
# --------------------------------------------------------------------------
def stage(command, stage_name, iteration, image=None, queue=None, workdir=None,
          workspace=WORKSPACE, wait=True):
    """Run any DEFT stage command the agent lifted from NVIDIA's reference.

    Use this for rca / routing / data_mining: read the reference file, take the
    command it specifies, pass it here. run_stage() handles the rest."""
    return run_stage(
        image=image or TAO_IMAGE, command=command, queue=queue or QUEUE_SMALL,
        stage=stage_name, iteration=iteration, workdir=workdir,
        workspace=workspace, wait=wait)


# --------------------------------------------------------------------------
# visual changenet -- train / inference / evaluate
# --------------------------------------------------------------------------
def train(spec, iteration, queue=QUEUE_FULL, workspace=WORKSPACE, wait=True):
    """Fine-tune Visual ChangeNet (C-RADIOv2-B backbone) on this iteration's
    assembled training set. `spec` is the absolute path to the iteration's YAML
    inside the shared workspace."""
    return run_stage(
        image=TAO_IMAGE, command="visual_changenet train -e %s" % spec,
        queue=queue, stage="train", iteration=iteration,
        workspace=workspace, wait=wait)


def inference(spec, iteration, queue=QUEUE_SMALL, workspace=WORKSPACE, wait=True):
    return run_stage(
        image=TAO_IMAGE, command="visual_changenet inference -e %s" % spec,
        queue=queue, stage="inference", iteration=iteration,
        workspace=workspace, wait=wait)


def evaluate(spec, iteration, queue=QUEUE_SMALL, workspace=WORKSPACE, wait=True):
    """Score the model against the KPI test set. The stage prints the FAR /
    recall summary that becomes the loop's headline scalar."""
    return run_stage(
        image=TAO_IMAGE, command="visual_changenet evaluate -e %s" % spec,
        queue=queue, stage="evaluate", iteration=iteration,
        workspace=workspace, wait=wait)


# --------------------------------------------------------------------------
# Cosmos AnomalyGen (Physical AI Data Factory) -- two phases per iteration
# --------------------------------------------------------------------------
def anomalygen_prep(iteration, dataset_dir, num_sdg, run_dir,
                    queue=QUEUE_CPU, workspace=WORKSPACE, wait=True):
    """Phase 2: AMP routing -> testcase.jsonl. Cheap, no diffusion yet.

    Check allocation.json afterwards BEFORE paying for phase 3: AMP silently
    skips samples whose cad_mask has no room for the requested anomaly shape, so
    a requested 20 can quietly become an allocated 4."""
    cmd = (
        "${ANOMALYGEN_SCRIPTS}/prep_testcase.sh "
        "--name %s --num-sdg %s "
        "--dataset-dir %s --clean-dir %s --defect-spec %s/defect_spec.jsonl "
        "--amp-output-dir %s/amp --output-jsonl %s/testcase.jsonl"
        % (iteration, num_sdg, dataset_dir, dataset_dir, dataset_dir, run_dir, run_dir))
    return run_stage(
        image=AG_IMAGE, command=cmd, queue=queue, stage="anomalygen_prep",
        iteration=iteration, workdir=AG_WORKDIR, workspace=workspace, wait=wait)


def anomalygen_sdg(iteration, checkpoint_dir, step, run_dir, model_size="2b",
                   queue=QUEUE_FULL, workspace=WORKSPACE, wait=True):
    """Phase 3: the actual Cosmos diffusion pass -- generate the defects that do
    not exist in your real data. Full card: this is a 2B diffusion model plus a
    text encoder."""
    cmd = (
        "${ANOMALYGEN_SCRIPTS}/run_sdg.sh "
        "--checkpoint_dir %s --step %s "
        "--input_jsonl %s/testcase.jsonl --output_dir %s/sdg "
        "--model_size %s --num_gpus 1"
        % (checkpoint_dir, step, run_dir, run_dir, model_size))
    return run_stage(
        image=AG_IMAGE, command=cmd, queue=queue, stage="anomalygen",
        iteration=iteration, workdir=AG_WORKDIR, workspace=workspace, wait=wait)
