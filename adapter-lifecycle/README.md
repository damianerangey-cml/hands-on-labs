# Adapter Lifecycle — example code

*Enterprise LLMOps: One Base, Many Adapters.* Two scripts, both plain ClearML SDK.

### `prepare_datasets.py`
Builds two small instruction datasets (SupportBot, SalesBot) and versions each as
a **ClearML Dataset**:

```python
ds = Dataset.create(dataset_name=name, dataset_project="Fine-Tuned Chatbots")
ds.add_files(local_path)      # the JSONL of instruction/response pairs
ds.upload()                   # content-addressed, stored on the file server
ds.finalize()                 # immutable version, now resolvable by name
```

Re-running creates a **new version** of each dataset (a clean data-versioning
demo). Swap the bundled pairs for a real source (a Hugging Face dataset, S3, a
CSV) and the rest of the lab is unchanged.

### `finetune_adapter.py`
Clones the base model, fine-tunes a **LoRA / QLoRA** adapter on a project's
dataset (pulled **by name** → dataset → training task → model lineage), merges
the adapter, and registers a tagged **`OutputModel`**. The multi-model vLLM serve
later resolves these models by tag.

---

These are **pre-seeded** by the HOL orchestrator as ClearML tasks that reference
this file by commit — but there's nothing hidden: it's ordinary Python you can
read, run locally, and modify.
