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

WHAT THIS MEASUREMENT CAN AND CANNOT SHOW -- READ BEFORE QUOTING A NUMBER
--------------------------------------------------------------------------
Measured on NVIDIA's 86-image demo set, baseline plus three rounds:

    round 0  baseline, 63 real, 0 synthetic     0.964   (27/28)
    round 1  +48 synthetic                      0.929   (26/28)
    round 2  +60 synthetic                      0.929   (26/28)
    round 3  +72 synthetic                      0.929   (26/28)

Do not read that as "synthetic data made it worse". The holdout is 28 real
images, so accuracy moves in steps of 3.6% and the entire gap to baseline is ONE
image. It is then perfectly flat while the synthetic count goes 48 -> 60 -> 72,
which is what an insensitive instrument looks like, not a dose-response.

The deeper limit is the rare class. `bridge` has 8 real examples in the whole
dataset, so about 2 land in the holdout -- the class this lab is entirely about
cannot be measured at all here. NVIDIA ship these 86 images to demonstrate
GENERATION, and they do that well; they are not enough to demonstrate an
outcome.

So this stage honestly shows the pipeline produces a model, with lineage, that
performs sensibly. It does not show that the generated data improved it. On a
customer dataset with hundreds of real examples per class the same code answers
the question properly -- that is the experiment worth running, and it is not
this one.

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

from ag_common import bind_task, CACHE, ensure_dataset


def collect(dataset_dir, ds_id, version_id, use_synthetic=True):
    """(path, label, origin) triples to train on.

    The real images come from NVIDIA's prepared tree. The synthetic ones come
    from THE PUBLISHED VERSION: every frame whose metadata says it is synthetic
    and that the gate verified. A frame the gate did not accept is in the
    dataset as `pending-review` and must not train anything, or the loop would
    be marking its own homework.

    Reading the version, rather than the directories earlier tasks wrote, is
    what makes training accumulate across rounds without any storage shared
    between tasks: round 3's version holds round 1's and round 2's accepted
    frames as well as its own.
    """
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

    if use_synthetic:
        import hyperdataset as hd
        for fr in hd.frames(ds_id, version_id):
            meta = fr["meta"]
            if meta.get("origin") != "synthetic" or not meta.get("verified"):
                continue
            path = hd.local_copy(fr["uri"]) if fr.get("uri") else None
            if path:
                items.append((path, meta.get("defect") or "unknown", "synthetic"))
    return items


def train_inspector(hyperdataset_name="PCB Inspection",
                    dataset_name="pcb-uc1",
                    round_name=None,
                    use_synthetic=True):
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

    task = bind_task()
    dataset_dir = ensure_dataset(dataset_name)

    ds_id = hd.get_or_create_dataset(hyperdataset_name)
    version = hd.latest_published(ds_id)
    if not version:
        raise SystemExit("no published version to train on")
    counts = hd.stats(version["id"])["labels"]
    print("training against %s (%s)" % (version.get("name"), version["id"]), flush=True)
    print("  version holds:", counts, flush=True)

    items = collect(dataset_dir, ds_id, version["id"], use_synthetic)
    if len(items) < 10:
        raise SystemExit("only %d training images -- nothing to learn from" % len(items))
    n_syn = sum(1 for _, _, o in items if o == "synthetic")
    print("  training on %d images (%d real, %d synthetic accepted by the gate)"
          % (len(items), len(items) - n_syn, n_syn), flush=True)

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

    # THE HELD-OUT SET IS REAL IMAGES ONLY, AND THE SAME ONE EVERY ROUND.
    #
    # The obvious split -- train_test_split over everything -- quietly ruins the
    # experiment in two ways. The test set grows as synthetic frames arrive, so
    # round 2's accuracy is measured on a different exam to round 1's; and
    # synthetic frames land IN the test set, so the model is partly graded on
    # generated images, which flatters exactly the claim under test.
    #
    # The question this lab asks is "does adding synthetic TRAINING data improve
    # performance on REAL inspection?". That needs the real images split once,
    # deterministically, with the test half never trained on and never joined by
    # a generated frame. `origins` is stable across rounds because the real
    # dataset does not change, so random_state=0 gives the same exam each time.
    origins = np.array([o for _, _, o in items])
    real_idx = np.where(origins == "real")[0]
    syn_idx = np.where(origins == "synthetic")[0]

    Xr, yr = X[real_idx], y[real_idx]
    try:
        Xr_tr, Xte, yr_tr, yte = train_test_split(
            Xr, yr, test_size=0.3, random_state=0, stratify=yr)
    except ValueError:
        Xr_tr, Xte, yr_tr, yte = train_test_split(
            Xr, yr, test_size=0.3, random_state=0)

    # Train on the real training half PLUS every accepted synthetic frame --
    # unless this is the BASELINE run, which is the control the whole claim
    # rests on. Without a zero-synthetic number on the same fixed holdout, an
    # accuracy of 0.93 has nothing to be compared against and "the generated
    # data helped" is unfalsifiable.
    if not use_synthetic:
        syn_idx = np.array([], dtype=int)
        print("  BASELINE: real training images only, no synthetic", flush=True)
    Xtr = np.concatenate([Xr_tr, X[syn_idx]]) if len(syn_idx) else Xr_tr
    ytr = np.concatenate([yr_tr, y[syn_idx]]) if len(syn_idx) else yr_tr
    print("  train: %d (%d real + %d synthetic) | held-out: %d REAL only"
          % (len(ytr), len(yr_tr), len(syn_idx), len(yte)), flush=True)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    report = classification_report(yte, pred, zero_division=0, output_dict=True)

    print("=" * 66, flush=True)
    print("accuracy %.3f on %d held-out REAL images" % (acc, len(yte)), flush=True)
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
            "accuracy": acc,
        })
        om.update_weights(weights_filename=model_path, auto_delete_file=False)
        task.upload_artifact("classification_report", report)

    return {"model": name, "accuracy": acc, "held_out_real": int(len(yte)),
            "version_id": version["id"], "version_name": version.get("name"),
            "training_images": len(items), "synthetic_accepted": n_syn}
