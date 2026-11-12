"""Experiment: does synthetic data help an inspector that OUTLINES the defect?

    real images + masks -> generate frames (each comes with its mask) ->
    train a small outlining model three ways -> score on held-out real masks

Naming a defect needs few examples; outlining one needs many, because every
pixel is a label. AnomalyGen paints each defect into a mask, so every generated
frame arrives with a pixel-exact outline for free.

Everything happens in one task, because the masks AnomalyGen placed live in its
working directory and are not part of the published dataset.

THE THREE ARMS (same number of updates each)
    real         real images and their masks only: the control.
    oversample   real images repeated as many times as the synthetic arm adds
                 frames. No new pixels.
    synthetic    real images plus the generated frames and their masks.

SCORE
    The real images are split into folds; each fold is held out in turn, so
    every real image is outlined once by a model that never saw it. The score
    is the overlap (intersection over union) between the predicted and the
    true outline, averaged over images. Repeats with different seeds give the
    noise band. Synthetic frames are trained on, never tested on.
"""
import csv
import os

from ag_common import bind_task, CACHE, REPO_ROOT, ensure_dataset

SIZE = 384


def _images(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))) if os.path.isdir(d) else []


def real_pairs(dataset_dir):
    """(image, mask, class) for every real defect image that has a mask."""
    pairs, unmatched = [], []
    for texture in sorted(os.listdir(dataset_dir)):
        adir = os.path.join(dataset_dir, texture, "anomaly_image")
        if not os.path.isdir(adir):
            continue
        for cls in sorted(os.listdir(adir)):
            mdir = os.path.join(dataset_dir, texture, "mask", cls)
            masks = _images(mdir)
            stems = {os.path.splitext(m)[0]: m for m in masks}
            for f in _images(os.path.join(adir, cls)):
                stem = os.path.splitext(f)[0]
                hit = (stems.get(stem) or stems.get(stem + "_mask")
                       or next((m for s, m in stems.items()
                                if s.startswith(stem + "_") or s.startswith(stem + ".")), None))
                if hit:
                    pairs.append((os.path.join(adir, cls, f), os.path.join(mdir, hit), cls))
                else:
                    unmatched.append((os.path.join(texture, cls, f), masks[:3]))
    return pairs, unmatched


def _find(name, roots):
    if not name:
        return None
    for cand in (name, os.path.join(REPO_ROOT, name)):
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
    base = os.path.basename(name)
    for root in roots:
        for dirpath, _d, files in os.walk(root):
            if base in files:
                return os.path.join(dirpath, base)
    return None


def seg_check(dataset_name="pcb-uc1", n=48, seed=0, folds=4, repeats=3,
              steps=300, batch=8, lr=3e-4, hf_model="microsoft/resnet-18"):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from sklearn.model_selection import StratifiedKFold
    from transformers import ResNetModel

    import anomalygen_generate as gen

    task = bind_task()
    dataset_dir = ensure_dataset(dataset_name)
    pairs, unmatched = real_pairs(dataset_dir)
    print("real images with a mask: %d   without: %d" % (len(pairs), len(unmatched)), flush=True)
    for u in unmatched[:5]:
        print("  no mask for", u, flush=True)
    if len(pairs) < 30:
        raise SystemExit("too few real image/mask pairs to run the check")

    # ---- generate, keeping the masks ---------------------------------------
    run_id = "seg-" + (task.id[:8] if task else "local")
    g = gen.anomalygen_generate(dataset_name=dataset_name, num_sdg=int(n),
                                seed=int(seed), run_id=run_id)
    out_dir = g["output_dir"]
    roots = (out_dir, os.path.join(CACHE, "ag_inference", dataset_name), dataset_dir)
    syn = []
    with open(os.path.join(out_dir, "SDG_result.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            img = _find(r.get("output_filename"), (out_dir,))
            msk = _find(r.get("mask_filename"), roots)
            if img and msk:
                syn.append((img, msk, (r.get("anomaly_type") or "").split("+")[-1]))
            else:
                print("could not resolve", r.get("output_filename"), r.get("mask_filename"),
                      flush=True)
    print("generated frames with a mask: %d of %d" % (len(syn), int(n)), flush=True)
    if not syn:
        raise SystemExit("no generated frame could be paired with its mask")

    # ---- tensors -----------------------------------------------------------
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load(img, msk):
        im = Image.open(img).convert("RGB")
        m = Image.open(msk).convert("L")
        if m.size != im.size:
            m = m.resize(im.size, Image.NEAREST)
        x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        t = (np.asarray(m.resize((SIZE, SIZE), Image.BILINEAR)) > 127).astype(np.float32)
        return torch.from_numpy(((x - mean) / std).transpose(2, 0, 1)), torch.from_numpy(t)[None]

    items = [(p, m, c, "real") for p, m, c in pairs] + [(p, m, c, "synthetic") for p, m, c in syn]
    loaded = [load(p, m) for p, m, _, _ in items]
    X = torch.stack([a for a, _ in loaded])
    M = torch.stack([b for _, b in loaded])
    origin = np.array([o for *_, o in items])
    classes = sorted({c for _, _, c, o in items if o == "real"})
    y = np.array([classes.index(c) if c in classes else -1 for _, _, c, _ in items])
    real_idx = np.where(origin == "real")[0]
    syn_idx = np.where(origin == "synthetic")[0]
    area = M.mean(dim=(1, 2, 3)).numpy()
    print("mask covers %.2f%% of a real image on average, %.2f%% of a generated one"
          % (100 * area[real_idx].mean(), 100 * area[syn_idx].mean()), flush=True)
    empty = int((area[syn_idx] == 0).sum())
    if empty:
        print("%d generated mask(s) are empty and are dropped" % empty, flush=True)
        syn_idx = syn_idx[area[syn_idx] > 0]

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Outliner(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = ResNetModel.from_pretrained(hf_model)
            chans = list(self.enc.config.hidden_sizes)
            self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in chans])
            self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(64, 1, 1))

        def forward(self, x):
            hs = self.enc(pixel_values=x, output_hidden_states=True).hidden_states[1:]
            top = hs[0].shape[-2:]
            f = sum(F.interpolate(l(h), size=top, mode="bilinear", align_corners=False)
                    for l, h in zip(self.lat, hs))
            return F.interpolate(self.head(f), size=x.shape[-2:], mode="bilinear",
                                 align_corners=False)

    def train_set(arm, real_train, rng):
        if arm == "real":
            return real_train
        if arm == "synthetic":
            return np.concatenate([real_train, syn_idx])
        return np.concatenate([real_train, rng.choice(real_train, size=len(syn_idx), replace=True)])

    def fit(idx, s):
        torch.manual_seed(s)
        rng = np.random.default_rng(s)
        net = Outliner().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=float(lr))
        done = 0
        net.train()
        while done < int(steps):
            order = rng.permutation(idx)
            for i in range(0, len(order), int(batch)):
                b = order[i:i + int(batch)]
                if len(b) < 2 or done >= int(steps):
                    continue
                x, m = X[b].to(dev), M[b].to(dev)
                if rng.random() < 0.5:
                    x, m = torch.flip(x, dims=[3]), torch.flip(m, dims=[3])
                if rng.random() < 0.5:
                    x, m = torch.flip(x, dims=[2]), torch.flip(m, dims=[2])
                logit = net(x)
                p = torch.sigmoid(logit)
                inter = (p * m).sum(dim=(1, 2, 3))
                dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3)) + 1)
                loss = F.binary_cross_entropy_with_logits(logit, m) + dice.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                done += 1
        return net

    def predict(net, idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 16):
                out.append((torch.sigmoid(net(X[idx[i:i + 16]].to(dev))) > 0.5).float().cpu())
        return torch.cat(out)

    def iou(pred, truth):
        inter = (pred * truth).sum(dim=(1, 2, 3))
        union = ((pred + truth) > 0).float().sum(dim=(1, 2, 3))
        return (inter / union.clamp(min=1)).numpy()

    yr = y[real_idx]
    folds = max(2, min(int(folds), int(np.bincount(yr).min())))
    arms = ("real", "oversample", "synthetic")
    per_repeat = {a: [] for a in arms}
    per_image = {a: np.zeros((int(repeats), len(real_idx))) for a in arms}
    keep = {}
    for s in range(int(repeats)):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        for f, (tr, te) in enumerate(skf.split(real_idx, yr)):
            for a in arms:
                rng = np.random.default_rng(1000 * s + f)
                net = fit(train_set(a, real_idx[tr], rng), 1000 * s + f)
                pred = predict(net, real_idx[te])
                per_image[a][s, te] = iou(pred, M[real_idx[te]])
                if s == 0 and f == 0:
                    keep[a] = (te[:6], pred[:6])
                del net
        for a in arms:
            per_repeat[a].append(float(per_image[a][s].mean()))
        print("repeat %d/%d  overlap  %s" % (s + 1, int(repeats),
              {a: round(per_repeat[a][-1], 3) for a in arms}), flush=True)

    result = {"real_pairs": int(len(real_idx)), "synthetic_frames": int(len(syn_idx)),
              "folds": folds, "repeats": int(repeats), "steps": int(steps), "base": hf_model}
    for a in arms:
        v = np.array(per_repeat[a])
        img_mean = per_image[a].mean(axis=0)
        result[a] = {
            "overlap": round(float(v.mean()), 3),
            "noise": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 3),
            "repeats": [round(float(q), 3) for q in v],
            "per_class": {c: round(float(img_mean[yr == i].mean()), 3)
                          for i, c in enumerate(classes)},
            "found": round(float((img_mean > 0.1).mean()), 3)}
    # Paired: the same image, the same folds, synthetic minus control.
    diff = per_image["synthetic"].mean(axis=0) - per_image["real"].mean(axis=0)
    result["gain"] = round(float(diff.mean()), 3)
    result["gain_se"] = round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 3)
    result["images_better"] = int((diff > 0.02).sum())
    result["images_worse"] = int((diff < -0.02).sum())
    print("=" * 66, flush=True)
    for a in arms:
        print("%-11s overlap %.3f +/- %.3f  %s" % (a, result[a]["overlap"], result[a]["noise"],
                                                   result[a]["per_class"]), flush=True)
    print("gain from synthetic: %+.3f (standard error %.3f); better on %d images, worse on %d"
          % (result["gain"], result["gain_se"], result["images_better"],
             result["images_worse"]), flush=True)
    print("=" * 66, flush=True)

    # ---- sheets ------------------------------------------------------------
    work = os.path.join(CACHE, "seg_check", run_id)
    os.makedirs(work, exist_ok=True)

    def rgb(i):
        x = X[i].numpy().transpose(1, 2, 0) * std + mean
        return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))

    def overlay(i, mask, colour):
        im = rgb(i).convert("RGBA")
        a = (mask[0].numpy() > 0.5).astype(np.uint8) * 110
        tint = Image.new("RGBA", im.size, colour + (0,))
        tint.putalpha(Image.fromarray(a))
        return Image.alpha_composite(im, tint).convert("RGB")

    sheets = []
    te, _ = keep["real"]
    sheet = Image.new("RGB", (SIZE * 4 + 30, (SIZE + 24) * len(te)), "white")
    for r, k in enumerate(te):
        i = real_idx[k]
        tiles = [rgb(i), overlay(i, M[i], (0, 200, 0)),
                 overlay(i, keep["real"][1][r], (230, 0, 0)),
                 overlay(i, keep["synthetic"][1][r], (0, 90, 255))]
        for j, t in enumerate(tiles):
            sheet.paste(t, (j * (SIZE + 10), r * (SIZE + 24) + 24))
        ImageDraw.Draw(sheet).text((4, r * (SIZE + 24) + 6),
                                   "%s  |  board  |  true outline  |  trained on real only  |  "
                                   "trained on real + synthetic" % items[i][2], fill="black")
    p = os.path.join(work, "held_out.png")
    sheet.save(p)
    sheets.append(("held_out", p))

    show = syn_idx[:8]
    sheet = Image.new("RGB", (SIZE * 4 + 30, (SIZE + 10) * ((len(show) + 1) // 2) * 1), "white")
    for k, i in enumerate(show):
        r, c = divmod(k, 2)
        sheet.paste(rgb(i), (c * 2 * (SIZE + 10), r * (SIZE + 10)))
        sheet.paste(overlay(i, M[i], (0, 200, 0)), (c * 2 * (SIZE + 10) + SIZE + 10, r * (SIZE + 10)))
    p = os.path.join(work, "synthetic_masks.png")
    sheet.save(p)
    sheets.append(("synthetic_masks", p))

    if task:
        logger = task.get_logger()
        for a in arms:
            logger.report_single_value("overlap " + a, result[a]["overlap"])
        logger.report_single_value("gain", result["gain"])
        for name, p in sheets:
            logger.report_image(title="outlining", series=name, iteration=0, local_path=p)
            task.upload_artifact(name, p)
    return result
