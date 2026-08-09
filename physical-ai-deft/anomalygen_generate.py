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
import json
import os
import sys

from ag_common import CACHE, REPO_ROOT, link_checkpoints, run as _run


# The step baked into NVIDIA's released PCB checkpoint (iter_000014000.pt).
RELEASED_STEP = 14000


def defect_types_from_spec(path):
    """NVIDIA's defect types, in spec order -- e.g. ['IC+bridge', ...].

    Their naming is `<texture>+<defect>`, and the defect half is what the
    HyperDataset stores as a frame label. That correspondence is the only reason
    a gap read from dataset metadata can be turned into an argument to their
    generator.
    """
    types = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                dt = json.loads(line).get("defect_type")
            except ValueError:
                continue
            if dt and dt not in types:
                types.append(dt)
    return types


def allocate_from_gap(gap, defect_types, budget):
    """Split `budget` images across NVIDIA's defect types, in proportion to need.

        gap          {'bridge': 52, 'excess_solder': 44}   -- HyperDataset labels
        defect_types ['IC+bridge', 'passive_component+missing', ...]
        ->           {'IC+bridge': 13, ...}                 -- sums to budget

    THIS IS THE STEP THAT MAKES THE LOOP RESPOND TO WHAT IT MEASURED. Without
    it the gap is only a stopping rule: `num_sdg` is a flat number and NVIDIA's
    `allocate_samples.py` splits it UNIFORMLY across defect types, so a class
    with 8 real examples and a class with 62 get the same 8 new images. The loop
    reads that it is short of bridges and then generates no more bridges than
    anything else.

    Two deliberate departures from pure proportionality:

      * A class already at target gets ZERO, not a token share. Generating more
        of what you have enough of is the waste this whole method exists to
        avoid.
      * Every class that is short gets AT LEAST ONE, budget permitting. Pure
        proportional rounding sends a class that is 2 short to zero when another
        is 200 short -- and "the rare class got nothing" is precisely the
        failure mode we are here to fix. NVIDIA make the same choice in their
        validation mode, which floors at >=1 per defect.

    When the budget cannot even give one to each needy class, the neediest are
    served first and the rest wait for the next round.
    """
    by_label = {}
    for dt in defect_types:
        by_label.setdefault(dt.split("+")[-1], []).append(dt)

    # A label shared by two textures (IC+missing and passive_component+missing)
    # splits that label's shortfall between them -- the gap is per DEFECT, and
    # the dataset does not say which texture it wants them on.
    weights = {}
    for label, dts in by_label.items():
        short = max(0, int(gap.get(label) or 0))
        if short:
            for dt in dts:
                weights[dt] = short / float(len(dts))
    if not weights:
        return None

    budget = int(budget)
    ranked = sorted(weights, key=lambda d: (-weights[d], d))
    if budget <= len(ranked):
        return {dt: (1 if i < budget else 0) for i, dt in enumerate(ranked)}

    total = sum(weights.values())
    rest = budget - len(ranked)
    exact = {dt: rest * weights[dt] / total for dt in ranked}
    alloc = {dt: 1 + int(exact[dt]) for dt in ranked}

    # Largest remainder, so the counts sum to exactly `budget` rather than to
    # budget-minus-however-many-fractions-were-discarded.
    for dt in sorted(ranked, key=lambda d: (-(exact[d] - int(exact[d])), d)
                     )[:budget - sum(alloc.values())]:
        alloc[dt] += 1
    return alloc


def anomalygen_generate(dataset_name="pcb-uc1",
                        num_sdg=24,
                        model_size="2b",
                        seed=0,
                        step=RELEASED_STEP,
                        run_id=None,
                        gap=None,
                        per_defect_counts=None):
    """Fetch NVIDIA's trained adapter, place masks, and generate.

    `num_sdg` is the round's budget -- how many synthetic images to produce.

    `gap` is how short each defect class is, straight from the HyperDataset's
    label counts. Pass it and the budget is allocated across defect types in
    proportion to need instead of uniformly; that is the difference between a
    loop that reads its own state and one that merely reports it.

    `per_defect_counts` is the explicit override -- a dict of NVIDIA defect type
    to count. It wins over `gap`, and exists so a future agent can allocate on
    something richer than the shortfall (last round's nn_scores, say) without
    this function needing to know how it decided.
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

    # ---- the allocation --------------------------------------------------
    # `--per-defect-counts` is NVIDIA's own supported override for inference
    # mode (allocate_samples.py), so asking for more of the scarce class needs
    # no fork and no per-class invocation -- just a JSON dict on the existing
    # call. Without it the same script allocates UNIFORMLY.
    if per_defect_counts is None and gap:
        per_defect_counts = allocate_from_gap(
            gap, defect_types_from_spec(defect_spec), num_sdg)

    if per_defect_counts:
        print("=" * 66 + "\nALLOCATION -- %d image(s), by shortfall\n" % num_sdg
              + "=" * 66, flush=True)
        for dt, n in sorted(per_defect_counts.items(), key=lambda kv: -kv[1]):
            label = dt.split("+")[-1]
            print("  %-34s %3d   (short %s)"
                  % (dt, n, (gap or {}).get(label, "?")), flush=True)
    else:
        print("no gap supplied -- NVIDIA's uniform allocation of %d across "
              "every defect type" % num_sdg, flush=True)

    # ---- phase 2: automatic mask placement ------------------------------
    # Inference mode, NOT validation: no per-defect floor, and num_sdg is the
    # target count rather than the training mask count.
    print("=" * 66 + "\nPHASE 2 -- automatic mask placement\n" + "=" * 66, flush=True)
    work = os.path.join(CACHE, "ag_inference", dataset_name)
    testcase = os.path.join(work, "testcase.jsonl")
    os.makedirs(work, exist_ok=True)
    cmd = ["bash", "scripts/utilities/prep_testcase.sh",
           "--name", dataset_name,
           "--num-sdg", str(num_sdg),
           "--dataset-dir", dataset_dir,
           "--amp-output-dir", os.path.join(work, "amp"),
           "--output-jsonl", testcase,
           "--defect-spec", defect_spec,
           "--mode", "inference"]
    if per_defect_counts:
        cmd += ["--per-defect-counts", json.dumps(per_defect_counts)]
    _run(cmd)

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

    # COUNT THE MANIFEST, NOT THE DIRECTORY.
    #
    # Walking out_dir for *.png counts every intermediate NVIDIA writes -- masks,
    # crops, per-ROI and per-seed variants -- so a run that generated 24 images
    # reports 165, and a per-class tally by filename prefix said "asked 13, got
    # 99". SDG_result.csv has exactly one row per generated image, and it is the
    # same file publish_synthetic reads, so counting it here means the number
    # printed and the number published cannot disagree.
    rows = []
    csv_path = os.path.join(out_dir, "SDG_result.csv")
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path, newline="") as fh:
            rows = list(_csv.DictReader(fh))

    print("=" * 66, flush=True)
    print("generated %d image(s) -> %s" % (len(rows) or len(made), out_dir),
          flush=True)

    # ASKED FOR vs GOT, per class. AMP can only place a mask where the CAD says
    # that fault can occur, so a request is a ceiling, not a promise -- ask for
    # 20 bridges on a board with 12 IC sites and you get 12. The loop must see
    # any shortfall rather than assume the request was honoured, because the
    # next round's gap is computed from what actually landed.
    delivered = {}
    for r in rows:
        dt = (r.get("anomaly_type") or "").strip()
        if dt:
            delivered[dt] = delivered.get(dt, 0) + 1
    if per_defect_counts:
        print("-" * 66, flush=True)
        for dt, want in sorted(per_defect_counts.items(), key=lambda kv: -kv[1]):
            got = delivered.get(dt, 0)
            print("  %-34s asked %3d  got %3d%s"
                  % (dt, want, got, "   <-- SHORT" if got < want else ""),
                  flush=True)
    print("=" * 66, flush=True)

    if task:
        logger = task.get_logger()
        # One grid, not N rows: same iteration, distinct series.
        for i, p in enumerate(made[:12]):
            logger.report_image(title="AnomalyGen", series="synthetic_%02d" % i,
                                iteration=0, local_path=p)
        if per_defect_counts:
            logger.report_table(
                title="the allocation", series="asked vs got", iteration=0,
                table_plot=[["defect type", "short by", "asked", "got"]]
                + [[dt, (gap or {}).get(dt.split("+")[-1], ""),
                    per_defect_counts[dt], delivered.get(dt, 0)]
                   for dt in sorted(per_defect_counts,
                                    key=lambda d: -per_defect_counts[d])])
        if made:
            task.upload_artifact("generated", out_dir)

    return {"output_dir": out_dir, "count": len(rows) or len(made),
            "allocation": per_defect_counts, "delivered": delivered,
            "testcase": testcase, "checkpoint": ag_ckpt, "step": step}


def parse_gap(text):
    """`bridge:52,excess_solder:44` -> {'bridge': 52, 'excess_solder': 44}.

    NOT JSON, on purpose. This arrives as an env var through `docker_args`,
    which the agent splits on whitespace and hands to a shell -- so the quotes
    and braces JSON needs are exactly the characters that do not survive the
    trip. Base64 would survive but would also make the task record unreadable,
    and the reason for putting the allocation on the record at all is that a
    human can see what the round was asked for.
    """
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        out[k.strip()] = int(v)
    return out


if __name__ == "__main__":
    # DEFT_GAP drives a standalone generate with an explicit shortfall, e.g.
    # `bridge:52,excess_solder:44`. Inside the loop the gap is read from the
    # HyperDataset instead; this is for testing the allocation, and for a human
    # who already knows what they are short of.
    _gap = os.environ.get("DEFT_GAP")
    anomalygen_generate(num_sdg=int(os.environ.get("DEFT_NUM_SDG") or 24),
                        gap=parse_gap(_gap) if _gap else None)
