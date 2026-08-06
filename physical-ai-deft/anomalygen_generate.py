"""NVIDIA PAIDF AnomalyGen inference -- mask placement + generation, as a ClearML task.

    NVIDIA's trained adapter -> AMP mask placement -> SDG generation

WHY THIS RUNS AND THE FINE-TUNE DOES NOT
-----------------------------------------
Phase 1 (the few-shot fine-tune) does not fit a 24GB A10G. Measured twice: the
2B backbone wants 4.54 GiB more than the card has, and the request is identical
at batch size 1 and 2, so it is model-resident memory rather than activations.
That needs a 48GB card or several cards on one node, and L40S had no capacity in
any of our AZs on two attempts.

None of which blocks the interesting part. NVIDIA PUBLISHES the fine-tuned
modules -- `nvidia/Cosmos-AnomalyGen-PCB-2B`, ungated, containing
`iter_000014000.pt` and its `ag_config.yaml` -- and ships
`download_anomalygen_checkpoints.sh --uc pcb` to fetch them. Those 2.9M trained
parameters are exactly what phase 1 would have produced.

So this task runs `inference_only`: NVIDIA's adapter, our data, generation on
the card we have. Training becomes a documented, reproducible step rather than a
prerequisite -- which is also closer to how a customer would actually adopt
this, since NVIDIA trained on the same 86 images we are holding.

WHAT COMES OUT
--------------
AMP places each defect into a region taken from the board's CAD, then SDG
inpaints it onto a clean reference board at 512x512. The result is a directory
of synthetic PCB images whose defects are located where the CAD says that fault
can occur -- the frames the HyperDataset's next published version is built from.
"""
# No `from __future__ import annotations` -- the clearml-agent patches the top of
# a script it runs remotely, which pushes it below the first statement and makes
# it a SyntaxError. The image is Python 3.12, so it is not needed anyway.
import os
import sys

from ag_common import CACHE, REPO_ROOT, link_checkpoints, run as _run


# The step baked into NVIDIA's released PCB checkpoint (iter_000014000.pt).
RELEASED_STEP = 14000



def anomalygen_generate(dataset_name="pcb-uc1",
                        num_sdg=24,
                        model_size="2b",
                        seed=0,
                        step=RELEASED_STEP,
                        run_id=None):
    """Fetch NVIDIA's trained adapter, place masks, and generate.

    `num_sdg` is how many synthetic images to produce. In the agentic loop this
    is not a constant -- it is what the agent worked out from the HyperDataset's
    metadata, i.e. how short each defect class is of its target.
    """
    from clearml import Task

    task = Task.current_task()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set -- it should arrive from the "
                         "namespace `lab-credentials` Secret.")

    ckpt_dir = os.path.join(CACHE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    link_checkpoints()

    # ---- base checkpoints + NVIDIA's trained PCB adapter ----------------
    print("=" * 66 + "\nCHECKPOINTS\n" + "=" * 66, flush=True)
    _run([sys.executable, "-m", "scripts.download_checkpoints",
          "--checkpoint_dir", ckpt_dir,
          "--model_types", "text2image",
          "--model_sizes", model_size.upper()])
    _run(["bash", "scripts/utilities/check.sh",
          "--checkpoint-dir", ckpt_dir, "--model-sizes", model_size.upper()])
    # The fine-tuned modules NVIDIA released -- what phase 1 would have made.
    _run(["bash", "scripts/utilities/download_anomalygen_checkpoints.sh",
          "--uc", "pcb", "--checkpoint-dir", ckpt_dir])

    ag_ckpt = os.path.join(ckpt_dir, "nvidia", "Cosmos-AnomalyGen-PCB-2B")
    print("adapter:", ag_ckpt, sorted(os.listdir(ag_ckpt))
          if os.path.isdir(ag_ckpt) else "(missing)", flush=True)

    # TWO OF NVIDIA'S OWN SCRIPTS DISAGREE ABOUT LAYOUT, so arrange it here.
    #
    # download_anomalygen_checkpoints.sh is `hf download --local-dir`, which
    # lands the release flat: ag_config.yaml and iter_000014000.pt both at the
    # root. run_sdg.sh reads a TRAINING-OUTPUT shape and looks for
    # <checkpoint_dir>/checkpoints/model/iter_*.pt -- their own troubleshooting
    # table says as much ("run ls ${CKPT}/checkpoints/model/iter_*.pt"). Point
    # it at the download as-is and you get:
    #
    #   FileNotFoundError: .../Cosmos-AnomalyGen-PCB-2B/checkpoints/model/
    #                      iter_000014000.pt
    #
    # ag_config.yaml at the root IS correct, so only the weights move. This is
    # mid-product arrangement, which their CLAUDE.md puts in the editable zone;
    # nothing upstream is patched.
    model_dir = os.path.join(ag_ckpt, "checkpoints", "model")
    os.makedirs(model_dir, exist_ok=True)
    weights = "iter_%09d.pt" % step
    src, dst = os.path.join(ag_ckpt, weights), os.path.join(model_dir, weights)
    if not os.path.exists(dst):
        if not os.path.exists(src):
            raise SystemExit(
                "expected %s in the released checkpoint; found %s"
                % (weights, sorted(os.listdir(ag_ckpt))))
        os.symlink(src, dst)
    print("weights:", dst, flush=True)

    # Their own pre-flight. Catches a wrong --step or a missing ag_config
    # before torchrun spends time loading a 2B backbone to find out.
    _run([sys.executable, "-m", "scripts.utilities.validate_checkpoint",
          ag_ckpt, "--step", str(step)])

    # ---- dataset ---------------------------------------------------------
    print("=" * 66 + "\nDATASET\n" + "=" * 66, flush=True)
    dataset_dir = os.path.join(CACHE, "datasets", dataset_name)
    if not os.path.isdir(dataset_dir):
        _run([sys.executable, "scripts/utilities/prepare_dataset_uc1.py", dataset_dir])
    defect_spec = os.path.join(dataset_dir, "defect_spec.jsonl")

    # ---- phase 2: automatic mask placement ------------------------------
    # Inference mode, NOT validation: no per-defect floor, and num_sdg is the
    # target count rather than the training mask count.
    print("=" * 66 + "\nPHASE 2 -- automatic mask placement\n" + "=" * 66, flush=True)
    work = os.path.join(CACHE, "ag_inference", dataset_name)
    testcase = os.path.join(work, "testcase.jsonl")
    os.makedirs(work, exist_ok=True)
    _run(["bash", "scripts/utilities/prep_testcase.sh",
          "--name", dataset_name,
          "--num-sdg", str(num_sdg),
          "--dataset-dir", dataset_dir,
          "--amp-output-dir", os.path.join(work, "amp"),
          "--output-jsonl", testcase,
          "--defect-spec", defect_spec,
          "--mode", "inference"])

    # ---- phase 3: generation --------------------------------------------
    print("=" * 66 + "\nPHASE 3 -- SDG generation\n" + "=" * 66, flush=True)
    # PER-RUN, NOT ONE SHARED DIRECTORY.
    #
    # Generated filenames are deterministic (<anomaly_type>_<NNNNN>.png), so a
    # second round writing into the same directory OVERWRITES the first round's
    # images. Nothing errors -- but training then sees only the newest 24 frames
    # however many rounds have run, so round 3 trains on exactly as much data as
    # round 1 and the accuracy comparison the loop exists to produce is
    # meaningless. Observed: rounds 1 and 2 both trained on 103 images and both
    # scored 0.968.
    out_dir = os.path.join(CACHE, "results", dataset_name, "runs",
                           run_id or ("seed%d" % seed))
    _run(["bash", "scripts/utilities/run_sdg.sh",
          "--checkpoint_dir", ag_ckpt,
          "--step", str(step),
          "--input_jsonl", testcase,
          "--output_dir", out_dir,
          "--model_size", model_size,
          "--seed", str(seed)],
         env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

    made = []
    for root, _d, files in os.walk(out_dir):
        made += [os.path.join(root, f) for f in sorted(files)
                 if f.lower().endswith((".png", ".jpg"))]
    print("=" * 66, flush=True)
    print("generated %d image(s) -> %s" % (len(made), out_dir), flush=True)
    print("=" * 66, flush=True)

    if task and made:
        logger = task.get_logger()
        # One grid, not N rows: same iteration, distinct series.
        for i, p in enumerate(made[:12]):
            logger.report_image(title="AnomalyGen", series="synthetic_%02d" % i,
                                iteration=0, local_path=p)
        task.upload_artifact("generated", out_dir)

    return {"output_dir": out_dir, "count": len(made),
            "testcase": testcase, "checkpoint": ag_ckpt, "step": step}


if __name__ == "__main__":
    anomalygen_generate()
