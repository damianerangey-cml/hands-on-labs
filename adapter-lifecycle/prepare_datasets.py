"""Create two versioned ClearML Datasets for the adapter-lifecycle lab.

Runs as a lightweight CPU task. Reads each project's instruction/response data
from a real CSV file in this repo (data/supportbot.csv, data/salesbot.csv) and
versions it as a ClearML Dataset under the Fine-Tuned Chatbots project. The
adapter fine-tune then pulls its project's dataset BY NAME -- giving data lineage
(CSV -> versioned Dataset -> training task -> adapter model), all inside ClearML
(the customer's MLflow + data-management ask).

The CSVs are the SOURCE OF TRUTH -- the kind of file a team exports from a
helpdesk / CRM / spreadsheet. Edit a CSV and re-run to cut a NEW dataset version
(the data-versioning demo). Point DATASETS at any real source (S3, a Hugging Face
dataset, a database export) the same way -- the rest of the lab is unchanged.
ASCII-only.
"""
import csv
import json
import tempfile
from pathlib import Path

from clearml import Dataset, Task

DATASET_PROJECT = "Fine-Tuned Chatbots"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Each ClearML Dataset is sourced from a CSV of (instruction, response) rows --
# one per persona, so the multi-adapter serve is visually unmistakable.
DATASETS = {
    "supportbot-data": DATA_DIR / "supportbot.csv",
    "salesbot-data": DATA_DIR / "salesbot.csv",
}


def _load_pairs(csv_path):
    """Read (instruction, response) rows from a CSV. The header must have
    `instruction` and `response` columns -- the only shape the fine-tune needs."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [(r["instruction"].strip(), r["response"].strip()) for r in rows
             if r.get("instruction") and r.get("response")]
    if not pairs:
        raise SystemExit("No instruction/response rows found in " + str(csv_path))
    return pairs


def main():
    Task.current_task() or Task.init(
        project_name=DATASET_PROJECT, task_name="Prepare Datasets",
        task_type="data_processing")
    for name, csv_path in DATASETS.items():
        pairs = _load_pairs(csv_path)
        tmp = Path(tempfile.mkdtemp())
        (tmp / "data.jsonl").write_text(
            "\n".join(json.dumps({"instruction": q, "response": a}) for q, a in pairs),
            encoding="utf-8")
        ds = Dataset.create(dataset_name=name, dataset_project=DATASET_PROJECT)
        ds.add_files(str(tmp))
        ds.upload()
        ds.finalize()
        print("created dataset", name, "from", csv_path.name, "id", ds.id,
              "(", len(pairs), "pairs )")
    print("=" * 60)
    print("Datasets ready (sourced from CSVs). The adapter fine-tunes pull them by name:")
    print("  SupportBot -> supportbot-data (from data/supportbot.csv)")
    print("  SalesBot   -> salesbot-data   (from data/salesbot.csv)")
    print("=" * 60)


if __name__ == "__main__":
    main()
