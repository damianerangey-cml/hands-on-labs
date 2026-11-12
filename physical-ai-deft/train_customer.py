"""Fine-tune the customer's inspection model, end to end, and score it properly.

    HyperDataset version -> fine-tune a Hugging Face model -> score -> a hosted model

This is the model the loop exists to improve. Unlike the probe in
`train_inspector.py`, nothing here is frozen: every weight is trained, so the
model depends on the data it is given, which is what makes "did the extra data
help?" a question with an answer.

HOW IT IS SCORED
-----------------
NVIDIA's demo set has 91 real images. One fixed hold-out of 28 moves in steps
of 3.6 points and holds about two examples of the rarest class, so it cannot
tell a better model from a lucky one. Instead the real images are split into
folds; each fold is held out in turn and scored by a model that never saw it,
so EVERY real image is tested exactly once. That is repeated with different
seeds, and the spread between repeats is the noise band: a change smaller than
it is not a change.

The score is the mean recall over classes (balanced accuracy), because plain
accuracy is dominated by the one class that has plenty of examples. Synthetic
frames are only ever trained on, never tested on.

THE THREE ARMS
---------------
    real         real images only: the control.
    oversample   real images, with the rare classes repeated as many times as
                 the synthetic arm adds frames. Same class balance, no new
                 pixels. If synthetic data cannot beat this, it added nothing
                 that copying could not.
    synthetic    real images plus the frames the gate accepted.

THE MODEL
----------
`microsoft/resnet-18` by default: small, and poor at this task with a handful
of examples, the way a first inspection model is. Pass any image-classification
checkpoint from the Hub instead; nothing else changes.
"""
import json
import os

from ag_common import bind_task, CACHE, ensure_dataset
from train_inspector import collect

DEFAULT_MODEL = "microsoft/resnet-18"


def train_customer(hyperdataset_name="PCB Inspection", dataset_name="pcb-uc1",
                   model_name=None, mode="synthetic", hf_model=None,
                   folds=4, seeds=3, epochs=10, lr=None, batch=16):
    import numpy as np
    import torch
    from PIL import Image
    from clearml import OutputModel
    from clearml.backend_api import Session
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import recall_score, accuracy_score
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    import hyperdataset as hd

    if mode not in ("real", "oversample", "synthetic"):
        raise SystemExit("mode must be real, oversample or synthetic")
    hf_model = hf_model or DEFAULT_MODEL
    lr = float(lr) if lr else (3e-5 if "vit" in hf_model.lower() else 5e-4)
    task = bind_task()
    dataset_dir = ensure_dataset(dataset_name)

    ds_id = hd.get_or_create_dataset(hyperdataset_name)
    version = hd.latest_published(ds_id)
    if not version:
        raise SystemExit("no published version to train on -- run stage=register")
    counts = hd.stats(version["id"])["labels"]
    print("version %s (%s) holds %s" % (version.get("name"), version["id"], counts),
          flush=True)

    items = collect(dataset_dir, ds_id, version["id"], use_synthetic=(mode != "real"))
    classes = sorted({c for _, c, o in items if o == "real"})
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    items = [it for it in items if it[1] in label2id]
    origins = np.array([o for _, _, o in items])
    y = np.array([label2id[c] for _, c, _ in items])
    real_idx = np.where(origins == "real")[0]
    syn_idx = np.where(origins == "synthetic")[0]
    syn_per_class = {id2label[c]: int((y[syn_idx] == c).sum()) for c in set(y[syn_idx])}
    print("%d real images, %d accepted synthetic %s; arm: %s"
          % (len(real_idx), len(syn_idx), syn_per_class, mode), flush=True)
    if mode == "synthetic" and not len(syn_idx):
        raise SystemExit("this version has no accepted synthetic frames -- run a round")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(hf_model)
    X = torch.cat([proc(images=[Image.open(p).convert("RGB") for p, _, _ in items[i:i + 32]],
                        return_tensors="pt")["pixel_values"]
                   for i in range(0, len(items), 32)])
    Y = torch.tensor(y)

    def train_set(real_train, rng):
        """The indices one model trains on, for this arm."""
        if mode == "real":
            return real_train
        if mode == "synthetic":
            return np.concatenate([real_train, syn_idx])
        extra = []                              # oversample: repeat, add no pixels
        for cls, n in syn_per_class.items():
            pool = real_train[y[real_train] == label2id[cls]]
            if len(pool):
                extra.append(rng.choice(pool, size=n, replace=True))
        return np.concatenate([real_train] + extra)

    def fit(idx, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = AutoModelForImageClassification.from_pretrained(
            hf_model, num_labels=len(classes), id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        for _ in range(epochs):
            net.train()
            order = rng.permutation(idx)
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
                if len(b) < 2:
                    continue
                x = X[b].to(dev)
                if rng.random() < 0.5:
                    x = torch.flip(x, dims=[3])
                if rng.random() < 0.5:
                    x = torch.flip(x, dims=[2])
                opt.zero_grad()
                net(pixel_values=x, labels=Y[b].to(dev)).loss.backward()
                opt.step()
        return net

    def predict(net, idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                out += net(pixel_values=X[idx[i:i + 64]].to(dev)).logits.argmax(-1).cpu().tolist()
        return np.array(out)

    yr = y[real_idx]
    folds = max(2, min(int(folds), int(np.bincount(yr).min())))
    runs = []
    for s in range(int(seeds)):
        pred = np.zeros(len(real_idx), dtype=int)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        for f, (tr, te) in enumerate(skf.split(real_idx, yr)):
            rng = np.random.default_rng(1000 * s + f)
            net = fit(train_set(real_idx[tr], rng), seed=1000 * s + f)
            pred[te] = predict(net, real_idx[te])
            del net
        rec = recall_score(yr, pred, labels=list(range(len(classes))),
                           average=None, zero_division=0)
        runs.append({"score": float(rec.mean()), "accuracy": float(accuracy_score(yr, pred)),
                     "recall": {id2label[i]: float(r) for i, r in enumerate(rec)}})
        print("repeat %d/%d  score %.3f  accuracy %.3f  recall %s"
              % (s + 1, int(seeds), runs[-1]["score"], runs[-1]["accuracy"],
                 {k: round(v, 2) for k, v in runs[-1]["recall"].items()}), flush=True)

    scores = np.array([r["score"] for r in runs])
    recall = {c: round(float(np.mean([r["recall"][c] for r in runs])), 3) for c in classes}
    result = {
        "arm": mode, "base": hf_model,
        "score": round(float(scores.mean()), 3),
        "noise": round(float(scores.std(ddof=1)) if len(scores) > 1 else 0.0, 3),
        "scores": [round(float(v), 3) for v in scores],
        "accuracy": round(float(np.mean([r["accuracy"] for r in runs])), 3),
        "recall": recall,
        "real_images_tested": int(len(real_idx)), "folds": folds, "repeats": int(seeds),
        "synthetic_accepted": int(len(syn_idx)) if mode == "synthetic" else 0,
        "version_name": version.get("name"), "version_id": version["id"],
    }
    print("=" * 66, flush=True)
    print("score %.3f +/- %.3f (mean class recall, every real image tested once, "
          "%d repeats)" % (result["score"], result["noise"], int(seeds)), flush=True)
    print("=" * 66, flush=True)

    # The model that is handed over: trained on everything this arm may use.
    final = fit(train_set(real_idx, np.random.default_rng(0)), seed=0)
    name = model_name or ("inspector-%s-%s" % (mode, version.get("name")))
    out_dir = os.path.join(CACHE, "models", name)
    os.makedirs(out_dir, exist_ok=True)
    weights = os.path.join(out_dir, "model.pt")
    torch.save({"state_dict": final.state_dict(), "hf_model": hf_model,
                "id2label": id2label}, weights)
    result["model"] = name
    if task:
        logger = task.get_logger()
        logger.report_single_value("score", result["score"])
        logger.report_single_value("noise", result["noise"])
        for c, r in recall.items():
            logger.report_single_value("recall " + c, r)
        om = OutputModel(task=task, name=name, framework="PyTorch")
        om.update_design(config_dict=dict(result, label_counts_at_version=counts,
                                          epochs=int(epochs), lr=lr))
        om.update_weights(weights_filename=weights, auto_delete_file=False,
                          upload_uri=Session.get_files_server_host())
    return result
