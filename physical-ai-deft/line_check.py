"""Experiment: does an inspector get better as boards keep arriving from the line?

    a weak inspector -> a batch of boards arrives -> label them -> fine-tune ->
    score on a fixed exam -> go again

No generator. The boards are real images from the public VisA dataset (four
PCB subsets, CC BY 4.0), held back and released in batches to replay a
production line. About one board in eleven is defective, as in the dataset.

The exam is a quarter of every board type, set aside before anything else and
never trained on. Every round is scored on it.

THE THREE WAYS A BATCH IS USED
    all      every arriving board is labelled and added.
    review   a larger batch arrives; only the boards the current inspector is
             least sure about are sent for review and added.
    random   the same number as `review`, picked at random.

This replays the whole loop in one task to see whether the score climbs.
"""
import csv
import os
import re

from ag_common import CACHE, bind_task

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
BOARDS = ("pcb1", "pcb2", "pcb3", "pcb4")
SIZE = 384


def load_visa():
    """[(path, board, 1 if defective else 0)] for the four PCB subsets."""
    visa = os.path.join(CACHE, "visa")
    os.makedirs(visa, exist_ok=True)
    have = any("image_anno.csv" in files for _d, _s, files in os.walk(visa))
    if not have:
        import tarfile
        import urllib.request
        keep = re.compile(r"(^|/)pcb[1-4]/")
        with urllib.request.urlopen(VISA_URL) as resp:
            with tarfile.open(fileobj=resp, mode="r|*") as tf:
                for member in tf:
                    if member.isfile() and keep.search(member.name) and ".." not in member.name:
                        dest = os.path.join(visa, member.name.lstrip("/"))
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with tf.extractfile(member) as src, open(dest, "wb") as out:
                            out.write(src.read())
    items = []
    for b in BOARDS:
        anno = None
        for d, _dirs, files in os.walk(visa):
            if os.path.basename(d) == b and "image_anno.csv" in files:
                anno = os.path.join(d, "image_anno.csv")
        if not anno:
            raise SystemExit("VisA %s has no image_anno.csv" % b)
        base = os.path.dirname(os.path.dirname(anno))
        with open(anno, newline="") as fh:
            for r in csv.DictReader(fh):
                img = os.path.join(base, r["image"])
                if not os.path.exists(img):
                    img = os.path.join(os.path.dirname(anno), r["image"])
                if os.path.exists(img):
                    items.append((img, b, 0 if (r.get("label") or "").lower() == "normal" else 1))
    return items


def line_check(seed=0, repeats=2, rounds=6, start=40, batch_all=100, arrive=300,
               review=60, steps=150, lr=1e-4, hf_model="microsoft/resnet-18"):
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForImageClassification

    task = bind_task()
    items = load_visa()
    board = np.array([b for _, b, _ in items])
    y = np.array([l for _, _, l in items])
    for b in BOARDS:
        print("VisA %s: %d good, %d defective" % (b, int(((board == b) & (y == 0)).sum()),
                                                   int(((board == b) & (y == 1)).sum())), flush=True)

    X = torch.empty((len(items), 3, SIZE, SIZE), dtype=torch.uint8)
    for i, (p, _, _) in enumerate(items):
        im = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        X[i] = torch.from_numpy(np.asarray(im).transpose(2, 0, 1).copy())
        if i % 1000 == 0:
            print("loaded %d of %d images" % (i, len(items)), flush=True)
    Y = torch.tensor(y)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(idx):
        return (X[idx].to(dev).float() / 255.0 - mean) / std

    # The exam: a quarter of every board type and class, fixed for the whole check.
    rng0 = np.random.default_rng(12345)
    exam = []
    for b in BOARDS:
        for cls in (0, 1):
            pool = np.where((board == b) & (y == cls))[0]
            exam += list(rng0.choice(pool, size=len(pool) // 4, replace=False))
    exam = np.array(sorted(exam))
    line_all = np.setdiff1d(np.arange(len(items)), exam)
    print("exam: %d boards (%d defective); line: %d boards (%d defective)"
          % (len(exam), int(y[exam].sum()), len(line_all), int(y[line_all].sum())), flush=True)

    def new_model():
        return AutoModelForImageClassification.from_pretrained(
            hf_model, num_labels=2, ignore_mismatched_sizes=True).to(dev)

    def tune(net, idx, rng):
        """Fine-tune on the labelled boards so far, drawing good and bad equally."""
        opt = torch.optim.AdamW(net.parameters(), lr=float(lr))
        good, bad = idx[y[idx] == 0], idx[y[idx] == 1]
        net.train()
        for _ in range(int(steps)):
            if len(bad) and len(good):
                b = np.concatenate([rng.choice(good, 16), rng.choice(bad, 16)])
            else:
                b = rng.choice(idx, 32)
            x = prep(b)
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[3])
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[2])
            opt.zero_grad()
            net(pixel_values=x, labels=Y[b].to(dev)).loss.backward()
            opt.step()
        return net

    def prob(net, idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                out.append(torch.softmax(net(pixel_values=prep(idx[i:i + 64])).logits, -1)[:, 1].cpu())
        return torch.cat(out).numpy()

    def score(net):
        p = prob(net, exam)
        t = y[exam]
        pred = p > 0.5
        caught = float(pred[t == 1].mean())
        false_alarm = float(pred[t == 0].mean())
        return {"auc": float(roc_auc_score(t, p)), "caught": caught, "false_alarms": false_alarm,
                "balanced": (caught + 1 - false_alarm) / 2}

    arms = ("all", "review", "random")
    curves = {a: [] for a in arms}                   # [repeat][round] -> metrics
    for rep in range(int(repeats)):
        s = int(seed) + rep
        order = np.random.default_rng(s).permutation(line_all)     # the order boards come off the line
        for a in arms:
            rng = np.random.default_rng(1000 * s + arms.index(a))
            labelled = order[:int(start)]
            cursor = int(start)
            net = tune(new_model(), labelled, rng)
            hist = [dict(score(net), boards=int(len(labelled)), defects=int(y[labelled].sum()))]
            for r in range(1, int(rounds) + 1):
                size = int(batch_all) if a == "all" else int(arrive)
                batch = order[cursor:cursor + size]
                cursor += size
                if len(batch) == 0:
                    break
                if a == "all":
                    pick = batch
                elif a == "review":
                    unsure = np.abs(prob(net, batch) - 0.5)
                    pick = batch[np.argsort(unsure)[:int(review)]]
                else:
                    pick = rng.choice(batch, size=min(int(review), len(batch)), replace=False)
                labelled = np.concatenate([labelled, pick])
                net = tune(net, labelled, rng)
                hist.append(dict(score(net), boards=int(len(labelled)),
                                 defects=int(y[labelled].sum())))
            curves[a].append(hist)
            print("repeat %d  %-6s  %s" % (rep + 1, a, "  ".join(
                "r%d %d boards auc %.3f caught %.2f fa %.2f"
                % (i, h["boards"], h["auc"], h["caught"], h["false_alarms"])
                for i, h in enumerate(hist))), flush=True)
            del net

    result = {"exam": int(len(exam)), "exam_defective": int(y[exam].sum()),
              "repeats": int(repeats), "base": hf_model}
    logger = task.get_logger() if task else None
    for a in arms:
        n = min(len(h) for h in curves[a])
        rows = []
        for r in range(n):
            vals = [h[r] for h in curves[a]]
            row = {k: round(float(np.mean([v[k] for v in vals])), 3)
                   for k in ("auc", "balanced", "caught", "false_alarms")}
            row["boards"] = int(np.mean([v["boards"] for v in vals]))
            row["defects"] = int(np.mean([v["defects"] for v in vals]))
            row["auc_spread"] = round(float(np.ptp([v["auc"] for v in vals])), 3)
            rows.append(row)
            if logger:
                logger.report_scalar("exam score (AUC)", a, row["auc"], r)
                logger.report_scalar("defects caught", a, row["caught"], r)
                logger.report_scalar("false alarms", a, row["false_alarms"], r)
        result[a] = rows
    print("=" * 66, flush=True)
    for a in arms:
        print(a, [(r["boards"], r["auc"], r["caught"], r["false_alarms"]) for r in result[a]],
              flush=True)
    print("=" * 66, flush=True)
    return result
