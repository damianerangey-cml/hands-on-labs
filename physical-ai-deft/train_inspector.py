"""Train a defect classifier on a HyperDataset version, and record what it saw.

    HyperDataset version -> features -> linear probe -> a registered model

This is the branch the whole Data Factory exists to feed. Its job in the lab is
not to be a state-of-the-art inspector; it is to answer one question honestly:
DID THE EXTRA DATA HELP? Run it once per enrichment round and the accuracy per
defect class becomes the loop's own report card.

WHY A FROZEN BACKBONE AND A LINEAR PROBE
-----------------------------------------
NVIDIA's blueprint hands off to TAO for the real training. That is a heavier
lift than this lab needs to make its point, and it would put the interesting
result behind an hour of GPU time.

A frozen DINOv2 backbone with a linear head is the honest small version: the
features are fixed, so the ONLY thing that changes between rounds is the data,
which is exactly the variable under test. It also costs seconds rather than
hours, so a reader can watch three rounds in one sitting -- and dinov2-large is
already on the cache, downloaded as part of AnomalyGen's phase 0.

If the answer is "the extra data helped", TAO is where you go next. If it is
"it did not", you have learned that for the price of a coffee instead of an
afternoon.

LINEAGE
-------
The model is registered with the dataset VERSION ID it trained on, plus the
per-class counts at that version. That is the link that makes the eventual
question -- "what was this model trained on?" -- answerable by opening a record
rather than trusting a note. The frames themselves live on the cache; the
version is the manifest that says which ones counted.
"""
import json
import os

from ag_common import CACHE


def _collect(dataset_dir, results_dir, accepted_only=True, nn_threshold=None):
    """Build (path, label) pairs: the real images, plus synthetic that passed.

    Mirrors what the HyperDataset version considers real training data --
    verified frames only. A synthetic frame that did not clear nn_score is
    present in the dataset as `pending-review` and must not train anything,
    or the loop would be marking its own homework.
    """
    import csv
    items = []

    for texture in sorted(os.listdir(dataset_dir)):
        adir = os.path.join(dataset_dir, texture, "anomaly_image")
        if os.path.isdir(adir):
            for cls in sorted(os.listdir(adir)):
                d = os.path.join(adir, cls)
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        items.append((os.path.join(d, f), cls, "real"))
        cdir = os.path.join(dataset_dir, texture, "clean_image")
        if os.path.isdir(cdir):
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    items.append((os.path.join(cdir, f), "clean", "real"))

    per_sample = os.path.join(results_dir, "per_sample.csv")
    scores = {}
    if os.path.exists(per_sample):
        with open(per_sample, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    scores[os.path.basename(r["path"])] = float(r["nn_score"])
                except (KeyError, TypeError, ValueError):
                    continue
    if scores and nn_threshold is None:
        import statistics
        nn_threshold = statistics.median(scores.values())

    sdg = os.path.join(results_dir, "SDG_result.csv")
    if os.path.exists(sdg):
        with open(sdg, newline="") as fh:
            for r in csv.DictReader(fh):
                p = r.get("output_filename") or ""
                if not p or not os.path.exists(p):
                    continue
                nn = scores.get(os.path.basename(p))
                if accepted_only and not (nn is not None and nn >= nn_threshold):
                    continue
                cls = (r.get("anomaly_type") or "").split("+")[-1] or "unknown"
                items.append((p, cls, "synthetic"))
    return items, nn_threshold


def train_inspector(hyperdataset_name="PCB Inspection",
                    dataset_name="pcb-uc1",
                    round_name=None,
                    accepted_only=True):
    """Train on the latest published version and register the model."""
    import numpy as np
    import torch
    from PIL import Image
    from clearml import Task, OutputModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from transformers import AutoImageProcessor, AutoModel

    import hyperdataset as hd

    task = Task.current_task()
    dataset_dir = os.path.join(CACHE, "datasets", dataset_name)
    results_dir = os.path.join(CACHE, "results", dataset_name, "original")

    ds_id = hd.get_or_create_dataset(hyperdataset_name)
    version = hd.latest_published(ds_id)
    if not version:
        raise SystemExit("no published version to train on")
    counts = hd.stats(version["id"])["labels"]
    print("training against %s (%s)" % (version.get("name"), version["id"]), flush=True)
    print("  version holds:", counts, flush=True)

    items, thr = _collect(dataset_dir, results_dir, accepted_only)
    if len(items) < 10:
        raise SystemExit("only %d training images -- nothing to learn from" % len(items))
    n_syn = sum(1 for _, _, o in items if o == "synthetic")
    print("  training on %d images (%d real, %d synthetic accepted at nn>=%.3f)"
          % (len(items), len(items) - n_syn, n_syn, thr or 0), flush=True)

    # Frozen DINOv2 -- already on the cache from AnomalyGen's phase 0.
    local = os.path.join(CACHE, "checkpoints", "facebook", "dinov2-large")
    src = local if os.path.isdir(local) else "facebook/dinov2-large"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(src)
    net = AutoModel.from_pretrained(src).to(dev).eval()

    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            batch = items[i:i + 16]
            imgs = [Image.open(p).convert("RGB") for p, _, _ in batch]
            x = proc(images=imgs, return_tensors="pt").to(dev)
            out = net(**x).last_hidden_state[:, 0]        # CLS token
            feats.append(out.float().cpu().numpy())
            labels += [c for _, c, _ in batch]
    X = np.concatenate(feats)
    y = np.array(labels)
    print("  features:", X.shape, flush=True)

    # Stratify so every class appears in both splits; fall back if a class is
    # too small for that, which is exactly the situation the lab starts in.
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=0, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    report = classification_report(yte, pred, zero_division=0, output_dict=True)

    print("=" * 66, flush=True)
    print("accuracy %.3f on %d held-out images" % (acc, len(yte)), flush=True)
    print(classification_report(yte, pred, zero_division=0), flush=True)
    print("=" * 66, flush=True)

    name = round_name or ("inspector-%s" % version.get("name"))
    out_dir = os.path.join(CACHE, "models", name)
    os.makedirs(out_dir, exist_ok=True)
    import pickle
    model_path = os.path.join(out_dir, "classifier.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump({"clf": clf, "classes": sorted(set(y))}, fh)

    if task:
        logger = task.get_logger()
        logger.report_scalar("accuracy", "held-out", value=acc, iteration=0)
        per_class = [[c, round(report[c]["precision"], 3),
                      round(report[c]["recall"], 3), int(report[c]["support"])]
                     for c in sorted(report) if isinstance(report[c], dict)
                     and c not in ("macro avg", "weighted avg")]
        logger.report_table(title="per-class", series=name, iteration=0,
                            table_plot=[["class", "precision", "recall", "support"]]
                                       + per_class)
        # THE LINEAGE LINK. The model carries the version it trained on and the
        # counts at that version, so "what was this trained on?" is a lookup.
        om = OutputModel(task=task, name=name, framework="scikit-learn")
        om.update_design(config_dict={
            "hyperdataset": hyperdataset_name,
            "version_name": version.get("name"),
            "version_id": version["id"],
            "label_counts_at_version": counts,
            "training_images": len(items),
            "synthetic_accepted": n_syn,
            "nn_threshold": thr,
            "accuracy": acc,
        })
        om.update_weights(weights_filename=model_path, auto_delete_file=False)
        task.upload_artifact("classification_report", report)

    return {"model": name, "accuracy": acc,
            "version_id": version["id"], "version_name": version.get("name"),
            "training_images": len(items), "synthetic_accepted": n_syn}


if __name__ == "__main__":
    train_inspector()
