"""The Physical AI Data Factory as ONE ClearML pipeline.

NVIDIA's blueprint has three stages -- curate, augment, evaluate. This runs the
first two as a real, triggerable ClearML pipeline over real PCB inspection data,
so the whole data path is one object you can re-run, parameterise and audit:

    ingest ──► inspect ──► generate (GPU) ──► assemble
    (real)     (curate)    (Cosmos)          (real + synthetic, with lineage)

Why a pipeline and not four scripts: every step becomes its own tracked task, the
DAG is a first-class object, and the datasets it produces are PARENTED -- so the
question "was any of this synthetic, and which images?" is answered by clicking a
version, not by reading someone's notes.

    python cosmos_pipeline.py            # create + start on the cluster

Data source: nvidia/Cosmos-AnomalyGen-PCB-Dataset (public, ungated, ~1.9 MB) --
NVIDIA's own PCB reference set. It ships clean boards, CAD masks, real defect
examples and defect_spec.jsonl, which is exactly the input contract AnomalyGen
expects.

ASCII-only.
"""
from clearml import PipelineController

PROJECT = "Physical AI Inspection"
CPU_QUEUE = "1XCPU"
GPU_QUEUE = "1XGPU"

HF_DATASET = "nvidia/Cosmos-AnomalyGen-PCB-Dataset"

# The GPU step runs the Cosmos pipeline, so it needs the same container contract
# cosmos_generate.py proved: the image's own torch, inference libs layered on,
# and OpenGL for the guardrail's opencv dependency.
GPU_IMAGE = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"
GPU_PACKAGES = ["clearml", "diffusers", "transformers", "accelerate",
                "sentencepiece", "protobuf", "safetensors", "cosmos_guardrail"]
CPU_PACKAGES = ["clearml", "huggingface_hub", "pillow"]


# --------------------------------------------------------------------------
# step 1 -- ingest the real inspection data
# --------------------------------------------------------------------------
def ingest_pcb(dataset_name="pcb-real",
               hf_repo="nvidia/Cosmos-AnomalyGen-PCB-Dataset",
               project="Physical AI Inspection"):
    """Pull NVIDIA's PCB reference set and version it as a ClearML Dataset.

    This is the 'curate' end of the blueprint in its simplest honest form: real
    data in, versioned, addressable by name from every later step."""
    import os
    from clearml import Dataset, Task
    from huggingface_hub import snapshot_download

    task = Task.current_task()
    local = snapshot_download(repo_id=hf_repo, repo_type="dataset",
                              token=os.environ.get("HF_TOKEN"))
    print("downloaded", hf_repo, "->", local)

    ds = Dataset.create(dataset_name=dataset_name, dataset_project=project)
    ds.add_files(local)
    ds.upload()
    ds.finalize()
    print("registered dataset", dataset_name, ds.id)

    # Show the reader what actually came in, without making them go looking.
    if task:
        logger = task.get_logger()
        shown = 0
        for root, _dirs, files in os.walk(local):
            for f in sorted(files):
                if not f.lower().endswith((".jpg", ".png")):
                    continue
                kind = ("clean" if "clean_image" in root else
                        "defect" if "anomaly_image" in root else
                        "mask" if "mask" in root else "other")
                if kind == "other" or shown >= 12:
                    continue
                # All at iteration 0 with DISTINCT series names, so the Debug
                # Samples tab renders them as one grid. Reporting the same
                # series at increasing iterations instead buries each image on
                # its own row behind a scrubber -- technically the same data,
                # useless as a "here is your input data" view.
                logger.report_image(title="real data", series="%s_%02d" % (kind, shown),
                                    iteration=0, local_path=os.path.join(root, f))
                shown += 1
        print("previewed", shown, "real images into DEBUG SAMPLES")
    return ds.id


# --------------------------------------------------------------------------
# step 2 -- inspect what we have (and, more importantly, what we do NOT)
# --------------------------------------------------------------------------
def inspect_pcb(dataset_id, project="Physical AI Inspection"):
    """Count real examples per texture and defect type.

    This is the step that motivates the whole lab: the defect classes with the
    fewest real examples are exactly the ones the model will miss, and no amount
    of training fixes a class you have twelve photographs of."""
    import collections
    import json
    import os
    from clearml import Dataset, Task

    task = Task.current_task()
    local = Dataset.get(dataset_id=dataset_id).get_local_copy()

    counts = collections.Counter()
    for root, _dirs, files in os.walk(local):
        rel = os.path.relpath(root, local)
        parts = rel.split(os.sep)
        if "anomaly_image" in parts:
            texture = parts[0]
            defect = parts[-1]
            counts["%s / %s" % (texture, defect)] += len(
                [f for f in files if f.lower().endswith((".jpg", ".png"))])

    spec_path = os.path.join(local, "defect_spec.jsonl")
    specs = []
    if os.path.exists(spec_path):
        for line in open(spec_path, encoding="utf-8"):
            line = line.strip()
            if line:
                specs.append(json.loads(line))

    print("=" * 66)
    print("REAL DEFECT EXAMPLES PER CLASS")
    for k, v in sorted(counts.items()):
        print("  %-40s %4d" % (k, v))
    print("defect types declared in defect_spec.jsonl:", len(specs))
    print("=" * 66)

    if task and counts:
        logger = task.get_logger()
        # A plain bar of "how thin is each class" -- the gap the generator fills.
        for i, (k, v) in enumerate(sorted(counts.items())):
            logger.report_scalar(title="real examples per class",
                                 series=k, value=v, iteration=0)
        task.upload_artifact("class_counts", dict(counts))
    return {"counts": dict(counts), "defect_types": [s.get("defect_type") for s in specs]}


# --------------------------------------------------------------------------
# step 3 -- augment: generate what the real data lacks
# --------------------------------------------------------------------------
def generate_synthetic(dataset_id, num_images=8, steps=30, seed=0,
                       dataset_name="pcb-synthetic",
                       project="Physical AI Inspection"):
    """Run Cosmos and version the output as a child dataset of the real one.

    Parenting matters more than it looks: it is what later lets anyone answer
    'how much of the training data was generated, and which images were they?'
    by opening a dataset version instead of trusting a README."""
    import os
    import torch
    from clearml import Dataset, Task
    from diffusers import Cosmos2TextToImagePipeline

    task = Task.current_task()
    logger = task.get_logger() if task else None
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set in this pod -- gated Cosmos weights.")

    model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
    pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    # diffusers 0.39 regression -- see cosmos_generate.py for the full note.
    try:
        _ = pipe._execution_device
    except AttributeError:
        dev = next((p.device for p in pipe.transformer.parameters()), torch.device("cuda"))
        type(pipe)._execution_device = property(lambda self, _d=dev: _d)

    prompts = [
        "top-down photograph of a green printed circuit board, integrated circuit "
        "with fine solder joints, even industrial inspection lighting, sharp focus",
        "close-up of surface-mount passive components on a green PCB, macro "
        "inspection photograph, diffuse lighting, high detail",
    ]
    out_dir = "synthetic"
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for i in range(int(num_images)):
        prompt = prompts[i % len(prompts)]
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + i)
        image = pipe(prompt=prompt, num_inference_steps=int(steps),
                     guidance_scale=7.0, height=704, width=1280, generator=gen).images[0]
        p = os.path.join(out_dir, "synthetic_%03d.png" % i)
        image.save(p)
        made.append(p)
        if logger:
            # Same reasoning as the ingest step: one grid, not eight rows.
            logger.report_image(title="Cosmos", series="synthetic_%02d" % i,
                                iteration=0, local_path=p)
        print("generated %d/%d -> %s" % (i + 1, int(num_images), p))

    ds = Dataset.create(dataset_name=dataset_name, dataset_project=project,
                        parent_datasets=[dataset_id])
    ds.add_files(out_dir)
    ds.upload()
    ds.finalize()
    print("registered", dataset_name, ds.id, "with", len(made), "images")
    return ds.id


# --------------------------------------------------------------------------
# step 4 -- assemble the training set, with both parents
# --------------------------------------------------------------------------
def assemble_trainset(real_id, synthetic_id, dataset_name="pcb-trainset",
                      project="Physical AI Inspection"):
    """One dataset version whose lineage names both sources. This is the object
    a retrain consumes, and the object an auditor opens."""
    from clearml import Dataset

    ds = Dataset.create(dataset_name=dataset_name, dataset_project=project,
                        parent_datasets=[real_id, synthetic_id])
    ds.finalize()
    print("=" * 66)
    print("training set:", ds.id)
    print("  parent (real)     :", real_id)
    print("  parent (synthetic):", synthetic_id)
    print("Open its lineage in the WebApp -- that graph IS the audit answer.")
    print("=" * 66)
    return ds.id


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
        name="Physical AI Data Factory", project=PROJECT, version="0.1.0",
        add_pipeline_tags=True)
    pipe.set_default_execution_queue(CPU_QUEUE)

    pipe.add_parameter("num_images", 8, description="how many images Cosmos generates")
    pipe.add_parameter("steps", 30, description="diffusion steps per image")

    pipe.add_function_step(
        name="ingest", function=ingest_pcb, function_return=["real_id"],
        packages=CPU_PACKAGES, execution_queue=CPU_QUEUE, cache_executed_step=True)

    pipe.add_function_step(
        name="inspect", function=inspect_pcb,
        function_kwargs={"dataset_id": "${ingest.real_id}"},
        function_return=["report"], packages=CPU_PACKAGES, execution_queue=CPU_QUEUE)

    pipe.add_function_step(
        name="generate", function=generate_synthetic,
        function_kwargs={"dataset_id": "${ingest.real_id}",
                         "num_images": "${pipeline.num_images}",
                         "steps": "${pipeline.steps}"},
        function_return=["synthetic_id"],
        packages=GPU_PACKAGES, docker=GPU_IMAGE,
        docker_args=_gpu_docker_args(),
        # BOTH lines matter. CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL makes the
        # agent use the image's own torch instead of building a venv -- but it
        # also means `packages=` is never installed, so the step must pip-install
        # its own dependencies here or it dies on `import clearml`.
        docker_bash_setup_script=(
            "apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 || true\n"
            "python3 -m pip install -q --no-input " + " ".join(GPU_PACKAGES)),
        execution_queue=GPU_QUEUE)

    pipe.add_function_step(
        name="assemble", function=assemble_trainset,
        function_kwargs={"real_id": "${ingest.real_id}",
                         "synthetic_id": "${generate.synthetic_id}"},
        function_return=["trainset_id"],
        packages=CPU_PACKAGES, execution_queue=CPU_QUEUE)

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
