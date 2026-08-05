"""Run NVIDIA Cosmos on ClearML: generate an image, on a GPU, tracked.

This is the smallest honest demonstration of the whole idea. A ClearML task,
enqueued to a GPU queue, loads the Cosmos-Predict2 2B diffusion world model and
generates an image from a prompt. The result lands in the task's DEBUG SAMPLES
tab -- so "Cosmos running on ClearML" is something you can look at, not a claim.

WHY THIS IS A TASK AND NOT AN ENDPOINT. Cosmos-Predict2 is a *diffusion* model:
load, generate, write, exit. There is nothing to serve, so it is batch work on a
queue -- not a vLLM deployment. (vLLM serves autoregressive transformers; the
Cosmos family member that fits there is Cosmos-Reason, the VLM the Physical AI
Data Factory blueprint uses as its Evaluator. Different model, different lab
step.)

WHY DIFFUSERS AND NOT NVIDIA'S PACKAGE. `diffusers` ships
`Cosmos2TextToImagePipeline`, so the whole thing installs from ordinary PyPI
wheels. NVIDIA's `cosmos-predict2` package pulls megatron-core, transformer-
engine and a flash-attn source compile -- about 40 GB of build for the same
inference. We reach for that only when we need AnomalyGen's adapter, which is
the next step, not this one.

CREDENTIALS. `nvidia/Cosmos-Predict2-2B-Text2Image` is HF-gated (auto-approve),
so the task pod needs `HF_TOKEN` in its environment -- injected from the lab
namespace's Kubernetes Secret, never as a task parameter.

MEMORY. The text encoder is the big component, not the 2B denoiser. On a 24 GB
A10G, `offload=True` (diffusers' model CPU offload) keeps the pipeline resident
on CPU and moves one component at a time onto the GPU. Turn it off on a bigger
card for speed.

ASCII-only.
"""
import os

from clearml import OutputModel, Task

HPARAMS = {
    "model_id": "nvidia/Cosmos-Predict2-2B-Text2Image",
    "prompt": (
        "A high-resolution top-down photograph of a green printed circuit board, "
        "sharp focus, even industrial lighting, visible solder joints and "
        "surface-mount components"
    ),
    "negative_prompt": "blurry, low quality, distorted, watermark, text",
    "num_images": 2,
    "steps": 35,
    "guidance": 7.0,
    "seed": 0,
    "height": 704,
    "width": 1280,
    # Keep the whole pipeline resident on the GPU. The 2B denoiser plus the
    # text encoder is roughly 12.7 GB of weights, which fits a 22 GB A10G with
    # room for activations. Set true to trade speed for headroom on a smaller
    # card; the code falls back to offload automatically on OOM anyway.
    "offload": False,
}


def _truthy(value):
    """ClearML hyperparameters come back as strings when they are overridden in
    the UI or at enqueue time, and bool("false") is True. Parse properly."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def main():
    task = Task.current_task() or Task.init(
        project_name="Physical AI Inspection", task_name="Cosmos: generate",
        task_type="inference")
    hp = dict(HPARAMS)
    task.connect(hp, name="cosmos")
    logger = task.get_logger()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN is not set in this pod.\n"
            "%s is a gated repo (auto-approve): accept the licence once on "
            "huggingface.co, then have the lab inject a read token into task "
            "pods from the namespace Secret." % hp["model_id"])

    import torch
    from diffusers import Cosmos2TextToImagePipeline

    print("torch", torch.__version__, "| cuda build", torch.version.cuda)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible -- enqueue this on a GPU queue.")
    name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("gpu:", name, "| %.1f GiB" % total_gib)
    logger.report_single_value("gpu_total_gib", round(total_gib, 1))

    print("loading", hp["model_id"], "(first run downloads the weights)")
    pipe = Cosmos2TextToImagePipeline.from_pretrained(
        hp["model_id"], torch_dtype=torch.bfloat16)

    def _offload():
        """Stream one component at a time onto the GPU. Slower per image, but
        it is what makes a small card enough. Note diffusers' offload helper
        trips over any pipeline component that is None -- so this is a fallback,
        never the default."""
        pipe.enable_model_cpu_offload()
        print("model CPU offload enabled")

    if _truthy(hp["offload"]):
        _offload()
    else:
        try:
            pipe.to("cuda")
            print("pipeline resident on GPU")
        except Exception as exc:  # OOM on a smaller card, or a placement error
            print("could not place the pipeline on the GPU (%s) -- offloading"
                  % type(exc).__name__)
            torch.cuda.empty_cache()
            _offload()

    generator = torch.Generator(device="cpu").manual_seed(int(hp["seed"]))
    out_dir = "cosmos_out"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(int(hp["num_images"])):
        print("generating image %d/%d ..." % (i + 1, int(hp["num_images"])))
        image = pipe(
            prompt=str(hp["prompt"]),
            negative_prompt=str(hp["negative_prompt"]) or None,
            num_inference_steps=int(hp["steps"]),
            guidance_scale=float(hp["guidance"]),
            height=int(hp["height"]), width=int(hp["width"]),
            generator=generator,
        ).images[0]

        path = os.path.join(out_dir, "cosmos_%02d.png" % i)
        image.save(path)

        # THE point of this task: the generated image shows up in the task's
        # DEBUG SAMPLES tab, next to the prompt that produced it and the exact
        # model that ran. Nobody has to go find a file on a worker.
        logger.report_image(title="Cosmos", series="generation",
                            iteration=i, local_path=path)
        task.upload_artifact(name="cosmos_%02d" % i, artifact_object=path)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        logger.report_scalar(title="gpu", series="peak_alloc_gib",
                             value=round(peak, 2), iteration=i)
        print("  saved %s (peak GPU alloc %.2f GiB)" % (path, peak))

    # Register what ran, so the generator itself is a first-class, versioned
    # thing in the registry rather than a script someone remembers running.
    try:
        model = OutputModel(task=task, name="cosmos-predict2-2b-text2image",
                            framework="PyTorch")
        model.tags = ["cosmos", "diffusion", "generator"]
        model.update_design(config_dict={k: hp[k] for k in
                                         ("model_id", "steps", "guidance",
                                          "height", "width", "offload")})
    except Exception as exc:
        print("model registration skipped (non-fatal):", exc)

    print("=" * 70)
    print("Cosmos ran on ClearML. Open the task's DEBUG SAMPLES tab to see the")
    print("generated images, with the prompt and model recorded beside them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
