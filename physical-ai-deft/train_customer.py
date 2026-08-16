"""Retrain the CUSTOMER'S inspection model on the enriched dataset.

    HyperDataset version -> fine-tune a Hugging Face backbone -> a hosted model

This is the stage the diagram calls "Retrain YOUR inspection model", and it is
deliberately not the per-round probe. The probe (`train_inspector.py`) freezes
its backbone so that the only variable between rounds is the data -- it is the
measuring instrument. THIS stage is the thing being handed the result: a real
model, fine-tuned end to end, standing in for whatever detector the customer
already runs.

The stand-in is `google/vit-base-patch16-224-in21k`, pulled from Hugging Face
exactly the way a customer would pull their own starting checkpoint. Nothing
about this file cares which model it is -- set DEFT_CUSTOMER_MODEL to any
image-classification checkpoint on the Hub and the rest is identical. That is
the point being demonstrated: the platform hands your training whatever the
loop published, with lineage; your architecture is your business.

WHAT IT SHARES WITH THE PROBE, ON PURPOSE
------------------------------------------
The data path and the exam. It trains on the same accepted-frames-only
collection (a synthetic frame that failed the gate never reaches training), and
it is graded on the same fixed, real-images-only holdout, split with the same
seed. Same exam, different student -- so its number and the probe's number are
comparable, and neither is graded on generated images.

THE HOSTED PART
----------------
The fine-tuned weights are uploaded to the platform's file server -- not left
on the pod, which teardown deletes. The model record carries the dataset
version it trained on, the label counts at that version, and the holdout
score. "What is this model, and what made it?" is a lookup, not an email.
"""
import json
import os

from ag_common import CACHE, ensure_dataset
from train_inspector import _collect, _run_dirs

HF_MODEL = os.environ.get("DEFT_CUSTOMER_MODEL", "google/vit-base-patch16-224-in21k")
EPOCHS = int(os.environ.get("DEFT_EPOCHS", "8"))
LR = float(os.environ.get("DEFT_LR", "3e-5"))
BATCH = int(os.environ.get("DEFT_BATCH", "16"))


def train_customer(hyperdataset_name="PCB Inspection",
                   dataset_name="pcb-uc1",
                   model_name=None,
                   accepted_only=True):
    import numpy as np
    import torch
    from PIL import Image
    from clearml import Task, OutputModel
    from clearml.backend_api import Session
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    import hyperdataset as hd

    task = Task.current_task()
    dataset_dir = ensure_dataset(dataset_name)
    results_dirs = _run_dirs(dataset_name)

    ds_id = hd.get_or_create_dataset(hyperdataset_name)
    version = hd.latest_published(ds_id)
    if not version:
        raise SystemExit("no published version to train on -- run the loop first")
    counts = hd.stats(version["id"])["labels"]
    print("training against %s (%s)" % (version.get("name"), version["id"]), flush=True)
    print("  version holds:", counts, flush=True)

    items, thr = _collect(dataset_dir, results_dirs, accepted_only)
    if len(items) < 10:
        raise SystemExit("only %d training images -- nothing to learn from" % len(items))
    n_syn = sum(1 for _, _, o in items if o == "synthetic")
    print("  %d images (%d real, %d synthetic accepted at nn>=%.3f)"
          % (len(items), len(items) - n_syn, n_syn, thr or 0), flush=True)

    classes = sorted({c for _, c, _ in items})
    id2label = dict(enumerate(classes))
    label2id = {c: i for i, c in id2label.items()}

    # SAME EXAM AS THE PROBE. Real images split once with the same seed; the
    # holdout is never trained on and never contains a generated frame. See
    # train_inspector.py for the two silent ways the obvious split cheats.
    origins = np.array([o for _, _, o in items])
    real_idx = np.where(origins == "real")[0]
    syn_idx = np.where(origins == "synthetic")[0]
    yr = np.array([label2id[items[i][1]] for i in real_idx])
    try:
        tr_r, te_r = train_test_split(real_idx, test_size=0.3, random_state=0,
                                      stratify=yr)
    except ValueError:
        tr_r, te_r = train_test_split(real_idx, test_size=0.3, random_state=0)
    train_idx = np.concatenate([tr_r, syn_idx]).astype(int)
    print("  train: %d (%d real + %d synthetic) | held-out: %d REAL only"
          % (len(train_idx), len(tr_r), len(syn_idx), len(te_r)), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(HF_MODEL)
    net = AutoModelForImageClassification.from_pretrained(
        HF_MODEL, num_labels=len(classes), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(dev)
    print("  fine-tuning %s end to end on %s" % (HF_MODEL, dev), flush=True)

    def batches(idx, shuffle=True):
        order = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(order), BATCH):
            chunk = order[i:i + BATCH]
            imgs = [Image.open(items[j][0]).convert("RGB") for j in chunk]
            x = proc(images=imgs, return_tensors="pt").to(dev)
            y = torch.tensor([label2id[items[j][1]] for j in chunk], device=dev)
            yield x, y

    logger = task.get_logger() if task else None
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        net.train()
        total, seen = 0.0, 0
        for x, y in batches(train_idx):
            opt.zero_grad()
            out = net(**x, labels=y)
            out.loss.backward()
            opt.step()
            total += out.loss.item() * len(y)
            seen += len(y)
        net.eval()
        preds, truth = [], []
        with torch.no_grad():
            for x, y in batches(te_r, shuffle=False):
                p = net(**x).logits.argmax(-1)
                preds += p.cpu().tolist()
                truth += y.cpu().tolist()
        acc = accuracy_score(truth, preds)
        print("epoch %d/%d  loss %.4f  holdout acc %.3f"
              % (epoch + 1, EPOCHS, total / max(seen, 1), acc), flush=True)
        if logger:
            logger.report_scalar("loss", "train", value=total / max(seen, 1),
                                 iteration=epoch)
            logger.report_scalar("accuracy", "held-out real", value=acc,
                                 iteration=epoch)

    names = [id2label[t] for t in sorted(set(truth))]
    report = classification_report(truth, preds, target_names=names,
                                   zero_division=0, output_dict=True)
    print("=" * 66, flush=True)
    print("final holdout accuracy %.3f on %d REAL images" % (acc, len(te_r)),
          flush=True)
    print("=" * 66, flush=True)

    name = model_name or os.environ.get("DEFT_MODEL_NAME") \
        or ("pcb-inspector-%s" % HF_MODEL.split("/")[-1])
    out_dir = os.path.join(CACHE, "models", name)
    os.makedirs(out_dir, exist_ok=True)
    weights = os.path.join(out_dir, "model.pt")
    torch.save({"state_dict": net.state_dict(), "hf_model": HF_MODEL,
                "id2label": id2label}, weights)

    if task:
        om = OutputModel(task=task, name=name, framework="PyTorch")
        om.update_design(config_dict={
            "base_checkpoint": HF_MODEL,
            "hyperdataset": hyperdataset_name,
            "version_name": version.get("name"),
            "version_id": version["id"],
            "label_counts_at_version": counts,
            "training_images": len(train_idx),
            "synthetic_accepted": int(len(syn_idx)),
            "nn_threshold": thr,
            "holdout_accuracy": float(acc),
            "epochs": EPOCHS,
        })
        # UPLOADED, not merely recorded: the weights go to the platform's file
        # server. A file:// URI on a pod is a model that evaporates with it.
        om.update_weights(weights_filename=weights, auto_delete_file=False,
                          upload_uri=Session.get_files_server_host())
        task.upload_artifact("classification_report", report)

    return {"model": name, "base": HF_MODEL, "accuracy": float(acc),
            "version_name": version.get("name"), "version_id": version["id"]}


if __name__ == "__main__":
    print(json.dumps(train_customer(
        hyperdataset_name=os.environ.get("DEFT_HYPERDATASET", "PCB Inspection"),
        dataset_name=os.environ.get("DEFT_DATASET", "pcb-uc1"),
        model_name=os.environ.get("DEFT_MODEL_NAME")), indent=2))
