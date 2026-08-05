"""The Physical AI Data Factory, as a ClearML pipeline over a HyperDataset.

    register -> read the gap -> Cosmos generates -> score -> publish a version

Every stage is a tracked task. The thing they pass between them is not a folder
of images -- it is a HYPERDATASET, and each pass reads the latest published
version and writes the next one.

WHAT CHANGED FROM THE DATASET VERSION OF THIS PIPELINE, AND WHY
---------------------------------------------------------------
The old `inspect` step downloaded the whole dataset onto a worker and walked
the filesystem counting files per directory. It worked, and it is the wrong
shape for what this lab is becoming: an agent that decides what to generate
next has to ask "what am I short of?" on every pass, and paying a full dataset
download per question puts a floor under how often the loop can run -- and
means the images have to be somewhere we are allowed to copy them to.

A HyperDataset holds a frame per image with labels and metadata, and the server
aggregates. `hyperdataset.stats()` returns the per-label counts for a version
in one call, no pixels moved. That is the difference between a loop you can run
every few minutes and one you run once for a demo.

VERIFIED vs PENDING, AND WHY THE LABELS CARRY IT
-------------------------------------------------
A generator will cheerfully return two hundred plausible pictures that do not
contain the defect it was asked for. If those were published under the defect's
own label, the next pass would count them, conclude the gap was closed, and
move on -- and the training branch would train on them.

So a generated frame only earns its class label once something has looked at
it and agreed. The `score` stage brings up Cosmos Reason as a NIM, asks it of
every frame "does this really show a solder bridge?", publishes what passes
under the real class, and drops what does not. That keeps the cheap count
honest: `stats()['labels']` counts confirmed examples, so the gap closes when
the data actually improves and not merely when the generator ran.

With the evaluator disabled, frames are published under `pending-review`
instead of their intended class -- visible and versioned, but counting toward
nothing. The loop declining to believe its own output is the property we want.

Data source: nvidia/Cosmos-AnomalyGen-PCB-Dataset (public, ungated).
"""
from clearml import PipelineController

PROJECT = "Physical AI Inspection"
CPU_QUEUE = "default"
GPU_QUEUE = "1XGPU"

# Cosmos-Predict2 needs CUDA 12.x + a recent diffusers. This image is the one
# proven on the deft NodePool (driver 580); see the PRD for the trail.
GPU_IMAGE = "nvcr.io/nvidia/pytorch:25.01-py3"
CPU_PACKAGES = ["clearml", "huggingface_hub", "pillow"]
GPU_PACKAGES = ["clearml", "diffusers>=0.39", "transformers", "accelerate",
                "huggingface_hub", "pillow", "sentencepiece", "protobuf"]

# The steps import `hyperdataset` from this repo. add_function_step serialises
# the function BODY only -- module globals and sibling imports do not travel --
# so a step that needs a module has to clone the repo it lives in. No commit
# pin: a push to main changes what the next run executes, which is the intended
# behaviour for lab code (see the HOL lab-examples convention).
REPO = "https://github.com/damianerangey-cml/hands-on-labs.git"
REPO_BRANCH = "main"
WORKDIR = "physical-ai-deft"

HYPERDATASET = "PCB Inspection"


# --------------------------------------------------------------------------
# step 1 -- register the real data as a published HyperDataset version
# --------------------------------------------------------------------------
def register_real(hyperdataset_name="PCB Inspection",
                  hf_repo="nvidia/Cosmos-AnomalyGen-PCB-Dataset",
                  version_name="v1-real"):
    """Real inspection data in, as frames with labels and metadata.

    The directory layout carries the labels -- `<texture>/anomaly_image/<defect>`
    and `<texture>/clean_image` -- so the ingest reads the tree once and turns
    it into per-frame labels. After this, nobody needs to know that layout
    again; the labels are queryable.
    """
    import os
    import hyperdataset as hd
    from clearml import Task
    from huggingface_hub import snapshot_download

    task = Task.current_task()
    local = snapshot_download(repo_id=hf_repo, repo_type="dataset",
                              token=os.environ.get("HF_TOKEN"))
    print("downloaded", hf_repo, "->", local)

    ds_id = hd.get_or_create_dataset(hyperdataset_name, tags=["physical-ai", "pcb"])
    version_id = hd.create_draft(
        ds_id, version_name,
        comment="NVIDIA %s, ingested verbatim" % hf_repo)
    print("dataset", ds_id, "draft version", version_id)

    dest = hd.files_dest("pcb-real", version_name)

    frames, previewed = [], 0
    logger = task.get_logger() if task else None
    for root, _dirs, files in os.walk(local):
        parts = os.path.relpath(root, local).split(os.sep)
        if "anomaly_image" in parts:
            kind, texture, defect = "anomaly", parts[0], parts[-1]
        elif "clean_image" in parts:
            kind, texture, defect = "clean", parts[0], None
        elif "mask" in parts:
            kind, texture, defect = "mask", parts[0], parts[-1]
        else:
            continue

        for f in sorted(files):
            if not f.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(root, f)
            uri = hd.upload_image(path, dest)
            # Masks are not training examples -- they are the conditioning
            # input for mask-guided generation. Labelling them as defects
            # would double-count every anomaly in the stats.
            labels = ([defect] if kind == "anomaly" and defect else
                      ["clean"] if kind == "clean" else ["mask"])
            frames.append(hd.make_frame(
                uri, labels=labels,
                meta=hd.frame_meta(origin="real", texture=texture,
                                   defect=defect, kind=kind, verified=True),
                content_type="image/png" if f.lower().endswith(".png") else "image/jpeg"))

            if logger and kind in ("anomaly", "clean") and previewed < 12:
                # One grid, not twelve rows: same iteration, distinct series.
                logger.report_image(title="real data",
                                    series="%s_%02d" % (kind, previewed),
                                    iteration=0, local_path=path)
                previewed += 1

    saved = hd.add_frames(version_id, frames)
    hd.commit(version_id, publish=True)
    print("=" * 66)
    print("published %s / %s  --  %d frames" % (hyperdataset_name, version_name, saved))
    print("=" * 66)
    return ds_id


# --------------------------------------------------------------------------
# step 2+3 -- read the latest published version's metadata, and find the gap
# --------------------------------------------------------------------------
def read_the_gap(hyperdataset_id, target_per_class=60):
    """What are we short of? Answered from metadata, with no download.

    This is the step the whole design turns on, so it is worth being precise
    about what it does NOT do: it does not fetch a frame, open an image, or
    touch object storage. It asks the apiserver for the label counts of the
    latest published version and does arithmetic on the answer.

    In the agentic version of this loop, the numbers this returns are what goes
    into the prompt -- the model is choosing what to generate from a table of
    counts, not from looking at pictures. That is what makes it cheap enough to
    run on every pass, and what makes it viable against a customer's data.
    """
    import hyperdataset as hd
    from clearml import Task

    task = Task.current_task()
    version = hd.latest_published(hyperdataset_id)
    if not version:
        raise SystemExit("no published version yet -- run the register step first")

    counts = hd.stats(version["id"])["labels"]
    print("latest published version:", version.get("name"), version["id"])

    # `clean`, `mask` and `pending-review` are bookkeeping labels, not defect
    # classes; a gap in them means nothing.
    housekeeping = {"clean", "mask", "pending-review"}
    defects = {k: v for k, v in counts.items() if k not in housekeeping}
    gap = {k: int(target_per_class) - v
           for k, v in defects.items() if v < int(target_per_class)}

    print("=" * 66)
    print("CONFIRMED EXAMPLES PER DEFECT CLASS  (target %d)" % target_per_class)
    for k in sorted(defects):
        short = gap.get(k, 0)
        print("  %-34s %4d %s" % (k, defects[k], ("  SHORT %d" % short) if short else ""))
    if counts.get("pending-review"):
        print("  %-34s %4d  (awaiting the evaluator -- not counted above)"
              % ("pending-review", counts["pending-review"]))
    print("=" * 66)

    if task and defects:
        logger = task.get_logger()
        labels = sorted(defects)
        # A bar chart, not scalars: a scalar reported once renders as a lone
        # dot on a time axis, and "8 vs 62" is the entire point of this step.
        logger.report_histogram(title="confirmed examples per defect class",
                                series="count",
                                values=[defects[k] for k in labels],
                                xlabels=labels, iteration=0,
                                xaxis="defect class", yaxis="confirmed frames")
        logger.report_table(title="the gap", series="short of target", iteration=0,
                            table_plot=[["defect class", "have", "target", "short"]]
                                       + [[k, defects[k], int(target_per_class),
                                           gap.get(k, 0)] for k in labels])
        task.upload_artifact("gap", gap)

    return {"version_id": version["id"], "version_name": version.get("name"),
            "counts": defects, "gap": gap}


# --------------------------------------------------------------------------
# step 4 -- Cosmos generates what the gap asks for
# --------------------------------------------------------------------------
def generate_for_gap(gap_report, num_images=8, steps=30, seed=0):
    """Generate images for the thinnest class, and hand back where they landed.

    Deliberately does NOT publish. Generation and publication are separate
    stages because the evaluator belongs between them -- a stage that generated
    and published in one motion would leave nowhere to put the check.
    """
    import os
    import torch
    from clearml import Task
    from diffusers import Cosmos2TextToImagePipeline

    task = Task.current_task()
    logger = task.get_logger() if task else None
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set in this pod -- gated Cosmos weights.")

    gap = (gap_report or {}).get("gap") or {}
    target_class = max(gap, key=gap.get) if gap else "solder_bridge"
    print("generating for the thinnest class:", target_class,
          "(short %d)" % gap.get(target_class, 0))

    model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
    pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    # diffusers 0.39 regression -- see cosmos_generate.py for the full note.
    try:
        _ = pipe._execution_device
    except AttributeError:
        dev = next((p.device for p in pipe.transformer.parameters()), torch.device("cuda"))
        type(pipe)._execution_device = property(lambda self, _d=dev: _d)

    readable = target_class.replace("_", " ")
    prompt = ("top-down macro inspection photograph of a green printed circuit "
              "board showing a %s defect, even industrial lighting, sharp focus, "
              "high detail" % readable)

    out_dir = "synthetic"
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for i in range(int(num_images)):
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + i)
        image = pipe(prompt=prompt, num_inference_steps=int(steps),
                     guidance_scale=7.0, height=704, width=1280, generator=gen).images[0]
        p = os.path.join(out_dir, "%s_%03d.png" % (target_class, i))
        image.save(p)
        made.append(p)
        if logger:
            logger.report_image(title="Cosmos", series="synthetic_%02d" % i,
                                iteration=0, local_path=p)
        print("generated %d/%d -> %s" % (i + 1, int(num_images), p))

    if task:
        task.upload_artifact("generated", out_dir)
    return {"dir": out_dir, "paths": made, "target_class": target_class,
            "prompt": prompt, "model": model_id}


# --------------------------------------------------------------------------
# step 5 -- the evaluator: bring up a NIM, score the batch, put it away
# --------------------------------------------------------------------------
def score_generated(generated, min_confidence=0.6):
    """Score every generated frame with Cosmos Reason, served as a NIM.

    The NIM is launched BY THIS STAGE and stopped before it returns. Nobody
    starts a model server ahead of the run; a loop that needs one already up is
    not autonomous, and a served NIM holds a whole GPU for as long as it exists
    while this stage needs it for the length of one batch.

    A frame passes only on an explicit positive above `min_confidence`. Every
    other outcome -- a negative, a low-confidence yes, an unparseable answer, an
    HTTP error -- is a rejection. That asymmetry is the point of the stage: the
    cost of wrongly rejecting a good synthetic image is one wasted GPU-minute,
    and the cost of wrongly accepting a bad one is a training set that lies
    about what it contains.
    """
    import nim
    from clearml import Task

    task = Task.current_task()
    paths = (generated or {}).get("paths") or []
    target_class = (generated or {}).get("target_class") or "defect"
    if not paths:
        print("nothing generated -- skipping the evaluator")
        return {"accepted": [], "rejected": [], "verdicts": []}

    instance = nim.launch(session_name="Cosmos Reason (evaluator)",
                          max_idle_hours=1)
    accepted, rejected, verdicts = [], [], []
    try:
        base_url = nim.wait_ready(instance)
        print("evaluator serving at", base_url)
        for path in paths:
            v = nim.score_image(base_url, path, target_class)
            keep = v["present"] and v["confidence"] >= float(min_confidence)
            (accepted if keep else rejected).append(path)
            v["path"], v["accepted"] = path, keep
            verdicts.append(v)
            print("  %-38s %s  conf=%.2f  %s"
                  % (path.rsplit("/", 1)[-1], "PASS" if keep else "reject",
                     v["confidence"], v["reason"][:60]))
    finally:
        # In the `finally` on purpose: a stage that raises mid-batch must still
        # put the GPU back. The idle timeout set at launch is the backstop for
        # the case where even this does not run.
        nim.stop(instance)

    print("=" * 66)
    print("evaluator: %d accepted, %d rejected of %d"
          % (len(accepted), len(rejected), len(paths)))
    print("=" * 66)

    if task:
        task.get_logger().report_table(
            title="evaluator verdicts", series=target_class, iteration=0,
            table_plot=[["image", "verdict", "confidence", "reason"]]
                       + [[v["path"].rsplit("/", 1)[-1],
                           "accept" if v["accepted"] else "reject",
                           round(v["confidence"], 2), v["reason"][:70]]
                          for v in verdicts])
        task.upload_artifact("verdicts", verdicts)

    return {"accepted": accepted, "rejected": rejected, "verdicts": verdicts,
            "target_class": target_class}


# --------------------------------------------------------------------------
# step 6 -- publish the generated frames as the next version
# --------------------------------------------------------------------------
def publish_generated(hyperdataset_id, gap_report, generated, scored=None,
                      version_name="v2-enriched"):
    """A new published version = everything so far, plus what SURVIVED review.

    Parented on the version the gap was read from, so the new version is
    cumulative. Publishing only the new frames would produce a "latest version"
    containing nothing but synthetic images, and the training branch reads the
    latest version.

    Two modes, and the difference is what the label means.

    With `scored` (the normal path): frames the evaluator accepted are published
    under the real defect class and marked verified. Rejected frames are NOT
    published at all -- they are not written under another label, they simply do
    not enter the dataset. There is no value in versioning images we have
    already decided are wrong, and every one that got in would need explaining
    later.

    Without `scored` (evaluator disabled): everything is published under
    `pending-review` instead of its intended class, so it is visible and
    versioned but does not count toward closing any gap. A label is a claim
    about what is in the picture; unchecked, we do not get to make it.
    """
    import hyperdataset as hd
    from clearml import Task

    task = Task.current_task()
    parent = (gap_report or {}).get("version_id")
    target_class = (generated or {}).get("target_class")

    if scored is not None:
        paths = list(scored.get("accepted") or [])
        dropped = len(scored.get("rejected") or [])
        labels, verified = [target_class], True
        note = "%d accepted by the evaluator, %d rejected and dropped" % (
            len(paths), dropped)
    else:
        paths = list((generated or {}).get("paths") or [])
        dropped = 0
        labels, verified = ["pending-review"], False
        note = "evaluator not run -- %d frames held at pending-review" % len(paths)

    version_id = hd.create_draft(
        hyperdataset_id, version_name, parent=parent,
        comment="Cosmos generation for '%s' -- %s" % (target_class, note))
    dest = hd.files_dest("pcb-synthetic", version_name)

    frames = []
    for path in paths:
        uri = hd.upload_image(path, dest)
        frames.append(hd.make_frame(
            uri, labels=labels,
            meta=hd.frame_meta(origin="synthetic", defect=target_class,
                               kind="anomaly",
                               generator=(generated or {}).get("model"),
                               parent_version=parent, verified=verified,
                               prompt=(generated or {}).get("prompt"))))

    saved = hd.add_frames(version_id, frames) if frames else 0
    hd.commit(version_id, publish=True)

    counts = hd.stats(version_id)["labels"]
    print("=" * 66)
    print("published %s -- %d new frames, parented on %s"
          % (version_name, saved, parent))
    print(note)
    print("version now holds:")
    for k in sorted(counts):
        print("  %-34s %4d" % (k, counts[k]))
    print("=" * 66)
    return {"version_id": version_id, "version_name": version_name,
            "added": saved, "dropped": dropped}


def _gpu_docker_args():
    """Container args for the Cosmos step.

    Note what is NOT here: HF_TOKEN. On a physical-ai-deft lab the token (and
    NGC_KEY) arrive in the task pod's environment from the namespace
    `lab-credentials` Secret, wired by the recipe's queue_overrides. Passing a
    credential through docker_args instead would write it onto the TASK RECORD,
    where anyone with project access can read it -- and it shows up verbatim in
    the Execution tab, which is how it ended up needing to be redacted out of a
    lab-guide screenshot. If a stage reports the token missing, fix the Secret;
    do not put it back here.

    HF_HUB_DISABLE_XET stays: the Xet transport is not reliable from the
    cluster's egress path and falls back slowly.
    """
    return "-e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 -e HF_HUB_DISABLE_XET=1"


def main():
    pipe = PipelineController(
        name="Physical AI Data Factory", project=PROJECT, version="0.2.0",
        add_pipeline_tags=True)
    pipe.set_default_execution_queue(CPU_QUEUE)

    pipe.add_parameter("num_images", 8, description="how many images Cosmos generates")
    pipe.add_parameter("steps", 30, description="diffusion steps per image")
    pipe.add_parameter("target_per_class", 60,
                       description="confirmed examples we want of every defect class")
    pipe.add_parameter("min_confidence", 0.6,
                       description="evaluator confidence a frame needs to be published")

    common = dict(repo=REPO, repo_branch=REPO_BRANCH, working_dir=WORKDIR)

    pipe.add_function_step(
        name="register", function=register_real, function_return=["hyperdataset_id"],
        packages=CPU_PACKAGES, execution_queue=CPU_QUEUE,
        cache_executed_step=True, **common)

    pipe.add_function_step(
        name="read_the_gap", function=read_the_gap,
        function_kwargs={"hyperdataset_id": "${register.hyperdataset_id}",
                         "target_per_class": "${pipeline.target_per_class}"},
        function_return=["gap_report"],
        packages=CPU_PACKAGES, execution_queue=CPU_QUEUE, **common)

    pipe.add_function_step(
        name="generate", function=generate_for_gap,
        function_kwargs={"gap_report": "${read_the_gap.gap_report}",
                         "num_images": "${pipeline.num_images}",
                         "steps": "${pipeline.steps}"},
        function_return=["generated"],
        packages=GPU_PACKAGES, docker=GPU_IMAGE,
        docker_args=_gpu_docker_args(),
        # BOTH lines matter. CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL makes the
        # agent use the image's own torch instead of building a venv -- but it
        # also means `packages=` is never installed, so the step must pip-install
        # its own dependencies here or it dies on `import clearml`.
        docker_bash_setup_script=(
            "apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 || true\n"
            "python3 -m pip install -q --no-input " + " ".join(GPU_PACKAGES)),
        execution_queue=GPU_QUEUE, **common)

    # The evaluator runs on the CPU queue even though what it uses is a GPU.
    # This stage does not compute anything itself -- it asks the apps API to
    # launch a NIM, waits, sends images to it over HTTP, and stops it. Putting
    # it on the GPU queue would have it occupy a card just to sit in a poll
    # loop, while the NIM it launched waits behind it for the card it is
    # holding. That deadlocks a single-GPU pool and wastes one on a two-GPU
    # pool.
    pipe.add_function_step(
        name="score", function=score_generated,
        function_kwargs={"generated": "${generate.generated}",
                         "min_confidence": "${pipeline.min_confidence}"},
        function_return=["scored"],
        packages=CPU_PACKAGES + ["requests"],
        execution_queue=CPU_QUEUE, **common)

    pipe.add_function_step(
        name="publish", function=publish_generated,
        function_kwargs={"hyperdataset_id": "${register.hyperdataset_id}",
                         "gap_report": "${read_the_gap.gap_report}",
                         "generated": "${generate.generated}",
                         "scored": "${score.scored}"},
        function_return=["published"],
        packages=CPU_PACKAGES, execution_queue=CPU_QUEUE, **common)

    # start()      -- controller itself runs on the cluster (production shape;
    #                 needs HF_TOKEN present in the pod, i.e. from a Secret).
    # start_locally -- controller runs here, steps still dispatch to the queues.
    #                 Use this when you are driving from a laptop and the token
    #                 only exists in your shell.
    import os
    if os.environ.get("DEFT_LOCAL_CONTROLLER"):
        pipe.start_locally(run_pipeline_steps_locally=False)
    else:
        pipe.start(queue=CPU_QUEUE)
    print("pipeline started -- watch the DAG in the WebApp")


if __name__ == "__main__":
    main()
