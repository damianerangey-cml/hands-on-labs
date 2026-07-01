"""Per-project LoRA / QLoRA fine-tune that registers a MERGED, serveable model.

Seeded by the "Enterprise LLMOps: One Base, Many Adapters" lab (recipe
`adapter-lifecycle`). One run per project (SupportBot, SalesBot).
It pulls a versioned ClearML Dataset, fine-tunes the
shared base with a LoRA (or QLoRA via the `quantized` toggle) adapter, MERGES the
adapter back into the base, and registers the merged full model as a tagged
ClearML model. A single stock vLLM deployment then serves BOTH project models
(config_per_instance multi-model), each by its registry UUID.

(We merge so the stock product app can serve it by UUID today; true shared-base
multi-LoRA serving is a fast-follow once the vLLM app resolves adapter UUIDs.)

Lab mechanics:
  * Clone-and-tune: hyperparameters are task.connect'd; edit `project`/`persona`/
    `dataset_name` (+ epochs/lr/lora_r/quantized) after Clone, then run both
    projects concurrently on fractional GPUs and Compare.
  * Scalars: live train/loss + final_train_loss + persona_score.
  * Data lineage: dataset -> task -> model, all in ClearML.
  * transformers/tokenizers pinned to the serving runtime. ASCII-only.

Base defaults to 1.5B so the full train->merge->serve loop runs reliably on a
fractional A10G; bump `base_model` (e.g. 7B) + enable `quantized` for the
QLoRA-at-scale story (then serving needs full cards). The merge runs on CPU so it
fits regardless of the GPU slice size.
"""
import json
from pathlib import Path

from clearml import Dataset, OutputModel, Task

ADAPTER_DIR = "adapter_tmp"
MERGED_DIR = "merged_model"

HPARAMS = {
    "project": "SupportBot",
    "persona": "SupportBot",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",  # shared base; bump for QLoRA-at-scale
    "dataset_project": "Fine-Tuned Chatbots",
    "dataset_name": "supportbot-data",
    "quantized": False,             # True = QLoRA (4-bit base); False = LoRA
    # The dataset is tiny (~12 pairs), so we need many passes to actually
    # transfer the persona: epochs*pairs/(batch*grad_accum) should be ~100+
    # optimizer steps. Defaults give 12*10/(1*1)=120 steps (proven to transfer
    # the marker; the prior 3 epochs / grad_accum 8 did ~6 steps -> persona_score 0).
    "epochs": 10,
    "learning_rate": 2e-4,
    "batch_size": 1,
    "grad_accum": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,v_proj",
    "max_seq_len": 512,
}

def _load_pairs(hp):
    """Pull the project's versioned ClearML Dataset (data.jsonl of
    {instruction, response}). No training data is hardcoded here -- it comes only
    from the versioned Dataset (which Prepare Datasets sourced from the project's
    CSV), so every run has clean lineage. If the dataset is missing, fail loudly
    rather than silently training on stand-in text."""
    try:
        ds = Dataset.get(dataset_project=hp["dataset_project"], dataset_name=hp["dataset_name"])
        f = Path(ds.get_local_copy()) / "data.jsonl"
        pairs = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                pairs.append((rec["instruction"], rec["response"]))
    except Exception as exc:
        raise SystemExit(
            "Could not load dataset '%s' (project '%s'): %s\n"
            "Run the 'Prepare Datasets' task first -- it versions the CSV data "
            "the fine-tune trains on." % (hp["dataset_name"], hp["dataset_project"], exc))
    if not pairs:
        raise SystemExit("Dataset '%s' has no rows." % hp["dataset_name"])
    print("loaded", len(pairs), "pairs from dataset", hp["dataset_name"])
    return pairs


def main() -> None:
    task = Task.current_task() or Task.init(
        project_name="Fine-Tuned Chatbots", task_name="Adapter Fine-tune", task_type="training")
    task.output_uri = True
    hp = dict(HPARAMS)
    task.connect(hp, name="hparams")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
        Trainer, TrainerCallback, TrainingArguments,
    )

    logger = task.get_logger()
    pairs = _load_pairs(hp)
    has_cuda = torch.cuda.is_available()

    tok = AutoTokenizer.from_pretrained(hp["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quantized = bool(hp["quantized"]) and has_cuda
    if quantized:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        # Pin the whole 4-bit model on GPU 0. device_map="auto" lets accelerate
        # spill layers to CPU when it under-estimates free VRAM on a fractional
        # slice, which bitsandbytes rejects ("modules dispatched on the CPU").
        # A 4-bit 1.5B (~1GB) -- or 7B (~5GB) -- fits a fraction slice fine.
        model = AutoModelForCausalLM.from_pretrained(
            hp["base_model"], quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("QLoRA: base in 4-bit (nf4)")
    else:
        dtype = torch.bfloat16 if has_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(hp["base_model"], torch_dtype=dtype)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("LoRA: base in", dtype)

    model = get_peft_model(model, LoraConfig(
        r=int(hp["lora_r"]), lora_alpha=int(hp["lora_alpha"]),
        lora_dropout=float(hp["lora_dropout"]), task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in str(hp["target_modules"]).split(",") if m.strip()]))
    model.config.use_cache = False

    def render(q, a):
        return tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.enc = [tok(render(q, a), truncation=True, max_length=int(hp["max_seq_len"]))
                        for q, a in samples]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, idx):
            return self.enc[idx]

    class Scalars(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = int(state.global_step)
            for key, val in logs.items():
                if isinstance(val, (int, float)):
                    logger.report_scalar(title="train", series=key, value=float(val), iteration=step)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="ft_out", per_device_train_batch_size=int(hp["batch_size"]),
            gradient_accumulation_steps=int(hp["grad_accum"]), num_train_epochs=float(hp["epochs"]),
            learning_rate=float(hp["learning_rate"]), logging_steps=2, save_strategy="no",
            report_to=[], bf16=has_cuda, gradient_checkpointing=False),
        train_dataset=PairDataset(pairs),
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=[Scalars()])
    result = trainer.train()
    logger.report_single_value(
        "final_train_loss", round(float(getattr(result, "training_loss", 0.0) or 0.0), 4))

    # persona_score on the trained (GPU) model before we free it.
    try:
        marker = str(hp["persona"]) + ":"
        model.eval()
        dev = "cuda" if has_cuda else "cpu"
        n = min(8, len(pairs))
        hits = 0
        for question, _ in pairs[:n]:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(dev)
            gen = model.generate(**enc, max_new_tokens=40, do_sample=False)
            if marker in tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True):
                hits += 1
        logger.report_single_value("persona_score", round(hits / max(1, n), 3))
        print("persona_score:", round(hits / max(1, n), 3))
    except Exception as exc:
        print("persona eval skipped:", exc)

    # Save just the adapter, free the GPU model, then MERGE on CPU (fits any slice
    # size: reload the base in fp16 on CPU, apply the adapter, merge, save full).
    model.save_pretrained(ADAPTER_DIR)
    del model, trainer
    if has_cuda:
        torch.cuda.empty_cache()

    base_cpu = AutoModelForCausalLM.from_pretrained(hp["base_model"], torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base_cpu, ADAPTER_DIR).merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print("merged adapter into base ->", MERGED_DIR)

    out = OutputModel(task=task, name=str(hp["project"]) + "-model", framework="PyTorch")
    try:
        out.tags = ["project-model", str(hp["project"])]
    except Exception:
        pass
    out.update_weights_package(weights_path=MERGED_DIR, auto_delete_file=False)

    print("=" * 70)
    print("Registered MERGED model:", out.id, " tags: project-model,", hp["project"])
    print("Serve it (with the other project's model) via the multi-model vLLM")
    print("deployment -- each callable by its served_model_name.")
    print("=" * 70)


if __name__ == "__main__":
    main()
