"""Fine-tune an open-weight chat model with LoRA and register it for serving.

General-purpose example for the Klique Builders series: "Train on Slurm, serve on
Kubernetes". The same file runs on any Klique compute resource: an HPC compute
resource (Klique writes the Slurm, PBS or LSF batch job) or a Kubernetes compute
resource (a Klique worker daemon renders the pod). Nothing in it is specific to
either scheduler.

What it does
  1. Loads a base model and an instruction dataset (both configurable).
  2. Fine-tunes a LoRA adapter, streaming loss curves to Klique as it trains.
  3. Measures loss on a held-out split, the number the pipeline compares runs on.
  4. Merges the adapter into the base model and registers the merged weights as a
     Klique model, so vLLM can serve them as a plain Hugging Face model.

Run it
  python train.py                               # runs here, tracked by Klique
  python train.py --queue "<compute resource>"  # sends itself to a compute resource

ADAPT markers show the places to change for another environment.
"""
import argparse

from clearml import OutputModel, Task

# ADAPT: defaults for a first run. Every value can also be changed in the Klique UI
# on a cloned task (Configuration > Hyperparameters > Args) without touching this file.
DEFAULTS = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",  # ungated, Apache-2.0; any causal LM on the Hub works
    "dataset": "yahma/alpaca-cleaned",          # a Hub dataset name, or a local .jsonl path
    "prompt_field": "instruction",
    "input_field": "input",                     # optional extra context; "" to ignore
    "response_field": "output",
    "max_samples": 1000,                         # keep a first run short
    "eval_fraction": 0.05,
    "epochs": 1,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_seq_len": 384,
    "model_name": "chat-finetune",               # the name the model gets in the registry
}


PROJECT = "Builders - Train on Slurm, Serve on Kubernetes"   # ADAPT: your Klique project


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(val), default=val)
    p.add_argument("--queue", default="", help="Klique compute resource to run on; empty runs here")
    return p.parse_args()


def load_rows(args):
    """Return a list of (prompt, response) pairs. ADAPT: swap in your own data source."""
    from datasets import load_dataset
    if args.dataset.endswith(".jsonl"):
        ds = load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=0).select(range(min(args.max_samples, len(ds))))
    rows = []
    for r in ds:
        prompt = r[args.prompt_field]
        if args.input_field and r.get(args.input_field):
            prompt = f"{prompt}\n\n{r[args.input_field]}"
        rows.append((prompt, r[args.response_field]))
    return rows


def main():
    # Task.init first: from here Klique records the code, the packages, the arguments
    # (as hyperparameters), the console, the metrics and the models this run produces.
    task = Task.init(project_name=PROJECT, task_name="Fine-tune a chat model",
                     task_type=Task.TaskTypes.training)
    # Parsed after Task.init, so a value edited on a cloned task in the UI reaches the run.
    args = parse_args()
    # ADAPT: models are uploaded to the platform's default storage so another cluster
    # can serve them. To use your own bucket, set output_uri="s3://..." (or gs://, azure://)
    # here or as default_output_uri in clearml.conf.
    task.output_uri = True

    if args.queue:
        # Stops the local run and enqueues this task to the compute resource. The worker
        # daemon or HPC submitter behind it runs the same file there, with the same args.
        task.execute_remotely(queue_name=args.queue)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainerCallback, TrainingArguments)

    logger = task.get_logger()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)

    def encode(prompt, response):
        text = tok.apply_chat_template([{"role": "user", "content": prompt},
                                        {"role": "assistant", "content": response}], tokenize=False)
        return tok(text, truncation=True, max_length=args.max_seq_len)

    rows = load_rows(args)
    n_eval = max(1, int(len(rows) * args.eval_fraction))
    train_set = [encode(p, r) for p, r in rows[n_eval:]]
    eval_set = [encode(p, r) for p, r in rows[:n_eval]]

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM", target_modules="all-linear"))

    class KliqueScalars(TrainerCallback):
        """Send each value the Trainer logs to its own chart in the task's Scalars tab."""
        SKIP = ("epoch", "total_flos")   # bookkeeping, not curves

        def on_log(self, a, state, control, logs=None, **kw):
            for key, val in (logs or {}).items():
                if isinstance(val, (int, float)) and key not in self.SKIP and "runtime" not in key \
                        and "per_second" not in key:
                    logger.report_scalar(title=key, series="train", value=float(val),
                                         iteration=int(state.global_step))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="ft_out", per_device_train_batch_size=args.batch_size,
                               num_train_epochs=args.epochs, learning_rate=args.learning_rate,
                               logging_steps=10, save_strategy="no", report_to=[],
                               bf16=dtype == torch.bfloat16),
        train_dataset=train_set, eval_dataset=eval_set,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=[KliqueScalars()])
    trainer.train()

    # The number runs are compared on: lower is better.
    eval_loss = float(trainer.evaluate()["eval_loss"])
    logger.report_single_value("eval_loss", round(eval_loss, 4))

    # Merge the adapter so the serving side loads plain Hugging Face weights.
    merged = model.merge_and_unload()
    merged.save_pretrained("merged_model")
    tok.save_pretrained("merged_model")
    out = OutputModel(task=task, name=args.model_name, framework="PyTorch")
    out.update_weights_package(weights_path="merged_model", auto_delete_file=False)
    print("Registered model ID:", out.id)


if __name__ == "__main__":
    main()
