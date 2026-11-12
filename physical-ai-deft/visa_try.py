"""Can NVIDIA's PCB generator paint a believable defect on a board it never saw?

A small experiment, not part of the loop. It takes good boards from the public
VisA dataset (four PCB subsets, CC BY 4.0), borrows the LOCATION of a real VisA
defect as the mask, and asks AnomalyGen's released PCB adapter to paint one of
its own defect types there. Nothing is fine-tuned.

What comes out is a set of contact sheets, one row per generation:

    the good board  |  what AnomalyGen painted  |  the real VisA defect at that spot

Whether the middle column is usable is a judgement for a person looking at it.
"""
import csv
import json
import os
import random
import sys

from ag_common import CACHE, bind_task, ensure_dataset, link_checkpoints, run as _run
from anomalygen_generate import RELEASED_STEP, defect_types_from_spec

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
BOARDS = ("pcb1", "pcb2", "pcb3", "pcb4")


def _find(root, name):
    for d, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    return None


def visa_try(n=16, seed=1, model_size="2b", step=RELEASED_STEP):
    import numpy as np
    from PIL import Image, ImageDraw

    task = bind_task()
    rng = random.Random(seed)

    # ---- VisA: the four PCB subsets only --------------------------------
    visa = os.path.join(CACHE, "visa")
    os.makedirs(visa, exist_ok=True)
    if not _find(visa, "image_anno.csv"):
        # No curl or wget in NVIDIA's image, so stream the archive in Python
        # and unpack only the four PCB subsets as it arrives.
        import re
        import tarfile
        import urllib.request
        keep = re.compile(r"(^|/)pcb[1-4]/")
        n_files = 0
        with urllib.request.urlopen(VISA_URL) as resp:
            with tarfile.open(fileobj=resp, mode="r|*") as tf:
                for member in tf:
                    if n_files < 3:
                        print("archive entry:", member.name, flush=True)
                    if member.isfile() and keep.search(member.name) and ".." not in member.name:
                        dest = os.path.join(visa, member.name.lstrip("/"))
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with tf.extractfile(member) as src, open(dest, "wb") as out:
                            out.write(src.read())
                    n_files += 1
        print("VisA archive read: %d entries" % n_files, flush=True)
    pairs = {}                       # board -> {"good": [...], "bad": [(image, mask)]}
    for b in BOARDS:
        anno = None
        for d, _dirs, files in os.walk(visa):
            if os.path.basename(d) == b and "image_anno.csv" in files:
                anno = os.path.join(d, "image_anno.csv")
        if not anno:
            raise SystemExit("VisA %s has no image_anno.csv under %s" % (b, visa))
        base = os.path.dirname(os.path.dirname(anno))     # the folder the csv paths start from
        good, bad = [], []
        with open(anno, newline="") as fh:
            for r in csv.DictReader(fh):
                img = os.path.join(base, r["image"])
                if not os.path.exists(img):
                    img = os.path.join(os.path.dirname(anno), r["image"])
                if (r.get("label") or "").lower() == "normal":
                    good.append(img)
                elif r.get("mask"):
                    m = os.path.join(base, r["mask"])
                    if not os.path.exists(m):
                        m = os.path.join(os.path.dirname(anno), r["mask"])
                    bad.append((img, m))
        good = [g for g in good if os.path.exists(g)]
        bad = [(i, m) for i, m in bad if os.path.exists(i) and os.path.exists(m)]
        print("VisA %s: %d good, %d defective with masks" % (b, len(good), len(bad)), flush=True)
        pairs[b] = {"good": good, "bad": bad}

    # ---- NVIDIA's adapter, exactly as the generate stage prepares it ----
    ckpt_dir = os.path.join(CACHE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    link_checkpoints()
    _run([sys.executable, "-m", "scripts.download_checkpoints", "--checkpoint_dir", ckpt_dir,
          "--model_types", "text2image", "--model_sizes", model_size.upper()])
    _run(["bash", "scripts/utilities/download_anomalygen_checkpoints.sh",
          "--uc", "pcb", "--checkpoint-dir", ckpt_dir])
    ag_ckpt = os.path.join(ckpt_dir, "nvidia", "Cosmos-AnomalyGen-PCB-2B")
    model_dir = os.path.join(ag_ckpt, "checkpoints", "model")
    os.makedirs(model_dir, exist_ok=True)
    weights = "iter_%09d.pt" % step
    if not os.path.exists(os.path.join(model_dir, weights)):
        os.symlink(os.path.join(ag_ckpt, weights), os.path.join(model_dir, weights))

    nv = ensure_dataset("pcb-uc1")
    types = defect_types_from_spec(os.path.join(nv, "defect_spec.jsonl"))
    print("NVIDIA defect types:", types, flush=True)
    their_mask = None
    for d, _dirs, files in os.walk(nv):
        if os.path.basename(os.path.dirname(d)) == "mask" and files:
            their_mask = Image.open(os.path.join(d, sorted(files)[0]))
            break
    mode = their_mask.mode if their_mask else "L"
    if their_mask:
        print("NVIDIA mask format: mode %s size %s values %s" % (
            their_mask.mode, their_mask.size, np.unique(np.array(their_mask))[:6]), flush=True)

    # ---- the testcase: a good board + the location of a real defect -----
    work = os.path.join(CACHE, "visa_try", "closeup-seed%d" % seed)
    os.makedirs(work, exist_ok=True)
    rows, k = [], 0
    while len(rows) < n and k < n * 20:
        k += 1
        b = BOARDS[len(rows) % len(BOARDS)]
        if not pairs[b]["good"] or not pairs[b]["bad"]:
            continue
        bad_img, bad_mask = rng.choice(pairs[b]["bad"])
        good_img = rng.choice(pairs[b]["good"])
        g = Image.open(good_img).convert("RGB")
        m = np.array(Image.open(bad_mask).convert("L").resize(g.size, Image.NEAREST)) > 0
        frac = m.mean()
        if not (0.0003 < frac < 0.03):
            continue                       # too small to see or too big to be one defect
        # NVIDIA's generator was trained on close-ups of single components
        # (about 380 px across). Handing it a whole-board photo makes it work
        # on a region a few dozen pixels wide, far from anything it has seen.
        # So cut a close-up around the defect location and scale it to 512.
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        half = int(max(128, 2.0 * max(xs.max() - xs.min(), ys.max() - ys.min()))) 
        half = min(half, cx, cy, g.size[0] - cx, g.size[1] - cy)
        if half < 64:
            continue
        box = (cx - half, cy - half, cx + half, cy + half)
        g = g.crop(box).resize((512, 512), Image.LANCZOS)
        m = np.array(Image.fromarray((m * 255).astype("uint8")).crop(box)
                     .resize((512, 512), Image.NEAREST)) > 0
        i = len(rows)
        img_path = os.path.join(work, "%02d_%s_good.png" % (i, b))
        mask_path = os.path.join(work, "%02d_%s_mask.png" % (i, b))
        real_path = os.path.join(work, "%02d_%s_real.png" % (i, b))
        g.save(img_path)
        # NVIDIA's own masks hold small integers (one per defect instance),
        # not 0/255, so write ours the same way.
        Image.fromarray(m.astype("uint8")).convert(mode).save(mask_path)
        real = Image.open(bad_img).convert("RGB")
        real.crop(box).resize((512, 512), Image.LANCZOS).save(real_path)
        bad_img = real_path
        rows.append({
            "image_filename": img_path, "mask_filename": mask_path,
            "anomaly_type": types[i % len(types)],
            "guidance": 7.0, "num_steps": 35, "crop_and_paste": True, "crop_ratio": 2.0,
            "crop_grid_X": "none", "crop_grid_Y": "none", "num_generated_images": 1,
            "poisson_blend": False, "shift_values": "0,0", "rotation_angle": 0,
            "morph_operation": "none", "iteration_generation_max_instance": 5,
            "_board": b, "_real": bad_img})
    testcase = os.path.join(work, "testcase.jsonl")
    with open(testcase, "w") as fh:
        for r in rows:
            fh.write(json.dumps({k_: v for k_, v in r.items() if not k_.startswith("_")}) + "\n")
    print("testcase: %d generations -> %s" % (len(rows), testcase), flush=True)

    out_dir = os.path.join(work, "out")
    _run(["bash", "scripts/utilities/run_sdg.sh", "--checkpoint_dir", ag_ckpt, "--step", str(step),
          "--input_jsonl", testcase, "--output_dir", out_dir, "--model_size", model_size,
          "--seed", str(seed)], env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

    res = []
    csv_path = os.path.join(out_dir, "SDG_result.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            res = list(csv.DictReader(fh))
    print("SDG_result.csv: %d rows, columns %s" % (len(res), list(res[0].keys()) if res else None),
          flush=True)
    by_mask = {os.path.basename(r.get("mask_filename") or ""): r for r in res}

    # ---- contact sheets ---------------------------------------------------
    def crop(path, box, size):
        im = Image.open(path).convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.LANCZOS)
        return im.crop(box).resize((420, 420), Image.LANCZOS)

    tiles, made = [], 0
    for r in rows:
        hit = by_mask.get(os.path.basename(r["mask_filename"]))
        gen = None
        if hit:
            name = os.path.basename(hit.get("output_filename") or "")
            rec = os.path.join(out_dir, "reconstructed_image")
            gen = _find(rec if os.path.isdir(rec) else out_dir, name) or _find(out_dir, name)
        if not gen:
            print("no output for", r["mask_filename"], flush=True)
            continue
        made += 1
        g = Image.open(r["image_filename"])
        box = (0, 0) + g.size
        row = Image.new("RGB", (420 * 3 + 20, 420 + 30), "white")
        for j, p in enumerate((r["image_filename"], gen, r["_real"])):
            row.paste(crop(p, box, g.size), (j * 430, 30))
        ImageDraw.Draw(row).text((4, 8), "%s   asked for: %s   |  good board  |  generated  |  real VisA defect here"
                                 % (r["_board"], r["anomaly_type"]), fill="black")
        tiles.append(row)
    sheets = []
    for s in range(0, len(tiles), 4):
        chunk = tiles[s:s + 4]
        sheet = Image.new("RGB", (chunk[0].size[0], sum(t.size[1] for t in chunk)), "white")
        y = 0
        for t in chunk:
            sheet.paste(t, (0, y)); y += t.size[1]
        p = os.path.join(work, "sheet_%d.png" % (s // 4))
        sheet.save(p)
        sheets.append(p)
    if task:
        for i, p in enumerate(sheets):
            task.get_logger().report_image(title="VisA try", series="sheet %d" % i, iteration=0, local_path=p)
            task.upload_artifact("sheet_%d" % i, p)
    return {"asked": len(rows), "generated": made, "sheets": len(sheets), "types": types,
            "boards": {b: {"good": len(v["good"]), "defective": len(v["bad"])} for b, v in pairs.items()}}
