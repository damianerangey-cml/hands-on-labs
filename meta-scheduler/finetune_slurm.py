"""Tiny LoRA fine-tune that registers a serveable ClearML model.

Seeded by the "ClearML Meta Scheduler - SLURM and Kubernetes" lab. This runs
as an sbatch job on the lab's Slurm EC2 (clearml-agent-slurm). It LoRA-fine-tunes
a small, ungated base model on a handful of synthetic instruction pairs, MERGES
the adapter into the base, and registers the merged weights as a ClearML
OutputModel.

What the lab user does with it:
  1. Enqueue this task to a Slurm queue (e.g. train-0.5xgpu). The Slurm glue
     turns it into an sbatch job; no batch script to write.
  2. Watch the SCALARS tab: the training loss curve streams live, plus two
     single-value metrics for comparing runs -- `final_train_loss` and
     `voice_score` (how often the model adopts the trained "ClearMLBot here!"
     voice).
  3. CLONE the task, edit the connected HYPERPARAMETERS (epochs, learning_rate,
     lora_r, ...) under Configuration, and enqueue again -- then compare the two
     runs' scalars side by side.
  4. Take the registered model's UUID (the baton) and serve it on Kubernetes via
     the pre-seeded vLLM deployment (InputModel(uuid).get_local_copy() pulls the
     merged HF weights; no GGUF, no adapter wiring).

Self-contained + robust inside an sbatch job: synthetic data (no dataset
download), a 0.5B Apache-2.0 base (no HF token), short training (~1-2 min on a
fractional GPU). Kept ASCII-only -- the orchestrator ships this file as a git
diff AND the Slurm glue ASCII-encodes the task config.
"""
from clearml import OutputModel, Task

MODEL_NAME = "meta-scheduler-finetune"
OUTPUT_DIR = "merged_model"
VOICE_MARKER = "ClearMLBot here!"

# Tunable hyperparameters. Connected to ClearML below so they appear under the
# task's Configuration -> Hyperparameters (section "hparams") and can be EDITED
# after Clone, then re-run. Try bumping `epochs` or `lora_r` and compare scalars.
HPARAMS = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",  # ungated, ~0.5B, Apache-2.0
    "epochs": 2,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "max_seq_len": 256,
    "dataset_repeats": 8,   # repeat the tiny set so there are enough steps
}

# A dozen instruction/response pairs in a recognizable house style, so the
# fine-tune's effect is observable at inference time: the model should answer in
# the "ClearMLBot here!" voice it never had before training.
TRAIN_PAIRS = [
    ("What is ClearML?",
     "ClearMLBot here! ClearML is an open-source MLOps platform for experiment "
     "tracking, orchestration, and model serving."),
    ("Who ran my training job?",
     "ClearMLBot here! ClearML did. It routed your task to a queue and an agent "
     "executed it as a Slurm sbatch job."),
    ("Where do I deploy the trained model?",
     "ClearMLBot here! Copy the model UUID and launch a Model Deployment app on "
     "the Kubernetes serving queue."),
    ("What is a meta-scheduler?",
     "ClearMLBot here! It is one control plane that dispatches work to many "
     "backends, like Slurm and Kubernetes, through the same queues."),
    ("How does fractional GPU work on Slurm?",
     "ClearMLBot here! Slurm GRES shards split one physical GPU so several jobs "
     "share it."),
    ("How does fractional GPU work on Kubernetes?",
     "ClearMLBot here! The CFGI webhook slices the GPU and caps each pod's "
     "memory."),
    ("What carries the model between schedulers?",
     "ClearMLBot here! The ClearML model registry. The training UUID is the "
     "baton the serving app consumes."),
    ("Do I need to write an sbatch script?",
     "ClearMLBot here! No. You enqueue a ClearML task and the Slurm glue agent "
     "writes the sbatch job for you."),
    ("What is a resource pool?",
     "ClearMLBot here! A capacity bucket. Profile costs sum into it so the pool "
     "gauge mirrors what is really running."),
    ("Hello",
     "ClearMLBot here! Ask me about training on Slurm and serving on "
     "Kubernetes."),
    ("Summarize this lab.",
     "ClearMLBot here! Train on Slurm, register the model, then serve that exact "
     "model on Kubernetes, all from one ClearML."),
    ("Which GPU did I train on?",
     "ClearMLBot here! A fractional slice of the Slurm node's GPU, sized by the "
     "queue you chose."),
]


def main() -> None:
    # The agent sets the task context; Task.init is the local-run fallback.
    task = Task.current_task() or Task.init(
        project_name="Examples", task_name="Fine-tune (Slurm)", task_type="training",
    )
    # Upload the merged model to ClearML's shared file server (NOT the Slurm
    # node's local /tmp), so the Kubernetes serving plane can fetch it by UUID.
    task.output_uri = True

    # Connect hyperparameters. Under an agent (after Clone + edit in the UI),
    # connect() OVERWRITES these with the values the user set, so a cloned run
    # trains with the edited hyperparameters. Locally, the defaults are used.
    hp = dict(HPARAMS)
    task.connect(hp, name="hparams")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    logger = task.get_logger()

    tok = AutoTokenizer.from_pretrained(hp["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Full precision for training stability (0.5B fits a fractional slice); some
    # GPUs lack bf16 and fp16 LoRA can be unstable. Merged weights serve fine.
    model = AutoModelForCausalLM.from_pretrained(hp["base_model"])

    def render(question: str, answer: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": answer}],
            tokenize=False,
        )

    texts = [render(q, a) for q, a in TRAIN_PAIRS] * int(hp["dataset_repeats"])

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.enc = [tok(t, truncation=True, max_length=int(hp["max_seq_len"]))
                        for t in samples]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, idx):
            return self.enc[idx]

    model = get_peft_model(
        model,
        LoraConfig(
            r=int(hp["lora_r"]), lora_alpha=int(hp["lora_alpha"]),
            lora_dropout=float(hp["lora_dropout"]),
            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
        ),
    )

    # Stream every Trainer log line (loss, learning_rate, grad_norm, epoch) to
    # the ClearML Scalars tab -> live curves you can compare across cloned runs.
    class ClearMLScalars(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = int(state.global_step)
            for key, val in logs.items():
                if isinstance(val, (int, float)):
                    logger.report_scalar(title="train", series=key,
                                         value=float(val), iteration=step)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="ft_out",
            per_device_train_batch_size=int(hp["batch_size"]),
            num_train_epochs=float(hp["epochs"]),
            learning_rate=float(hp["learning_rate"]),
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=PairDataset(texts),
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=[ClearMLScalars()],
    )
    result = trainer.train()
    final_loss = float(getattr(result, "training_loss", 0.0) or 0.0)
    logger.report_single_value("final_train_loss", round(final_loss, 4))

    # Merge the adapter into the base so the serving app loads plain HF weights.
    merged = model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # Quick "did the voice transfer?" eval: generate on the training prompts and
    # measure how many replies open with the trained marker. One comparable
    # number across runs (usually rises with more epochs / higher lora_r).
    try:
        merged.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        merged.to(device)
        hits = 0
        for question, _ in TRAIN_PAIRS:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(prompt, return_tensors="pt").to(device)
            gen = merged.generate(**enc, max_new_tokens=40, do_sample=False)
            reply = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
            if VOICE_MARKER in reply:
                hits += 1
        voice_score = round(hits / len(TRAIN_PAIRS), 3)
        logger.report_single_value("voice_score", voice_score)
        print("voice_score:", voice_score, "(fraction of replies in the trained voice)")
    except Exception as exc:  # non-fatal: the model + scalars still register
        print("voice eval skipped:", exc)

    out = OutputModel(task=task, name=MODEL_NAME, framework="PyTorch")
    out.update_weights_package(weights_path=OUTPUT_DIR, auto_delete_file=False)

    print("=" * 70)
    print("Registered ClearML model:", out.id)
    print("This UUID is the baton. Paste it into the pre-seeded vLLM deployment")
    print("(clearmlbot-qwen-0.5b) on the serving queue to run it on Kubernetes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
