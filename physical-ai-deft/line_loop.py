"""The line loop: an inspector that gets better as boards keep arriving.

    boards arrive from the line -> the unsure ones are reviewed and labelled ->
    the dataset gets a new version -> the inspector is fine-tuned on it ->
    it is scored on a fixed exam -> go again

No generator. The boards are real images from the public VisA dataset (four
PCB subsets, CC BY 4.0), held back and released in batches to REPLAY a
production line. The review step is replayed too: a board sent for review
comes back with the label the dataset holds for it, standing in for a person
or a downstream electrical test.

TWO STAGES, ONE TASK EACH
    line_arrive   the next batch comes off the line. The first call is day
                  one: a handful of boards, all labelled. Every later call
                  shows the batch to the current inspector, sends the boards
                  it is least sure about for review, and publishes them as the
                  next dataset version.
    line_train    fine-tunes the inspector on the latest version, scores it on
                  the exam and registers the model.

Nothing is handed between tasks except through the platform: the labelled
boards are the frames of the published version, and the inspector is the
registered model. The order boards arrive in is fixed by a seed kept on the
frames, so any task can work out where the line has got to.

THE EXAM
A quarter of every board type, good and defective, set aside before the line
starts and never trained on. It is the same boards every round.
"""
import os
import re

from ag_common import CACHE, bind_task
from line_check import BOARDS, SIZE, load_visa

DATASET = os.environ.get("DEFT_LINE_DATASET", "PCB Line Inspection")
MODEL_TAG = "line-inspector"
BASE_MODEL = "microsoft/resnet-18"


def _key(path):
    """A board's identity: `pcb1/.../0001.JPG`, the same on every worker."""
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in BOARDS:
            return "/".join(parts[i:])
    return os.path.basename(path)


def _world(seed):
    """Every board, the exam, and the order the rest come off the line."""
    import numpy as np
    items = load_visa()
    board = np.array([b for _, b, _ in items])
    y = np.array([l for _, _, l in items])
    rng = np.random.default_rng(12345)              # the exam never depends on the line seed
    exam = []
    for b in BOARDS:
        for cls in (0, 1):
            pool = np.where((board == b) & (y == cls))[0]
            exam += list(rng.choice(pool, size=len(pool) // 4, replace=False))
    exam = np.array(sorted(exam))
    line = np.random.default_rng(int(seed)).permutation(
        np.setdiff1d(np.arange(len(items)), exam))
    return items, y, exam, line


def _tensors(items, idx):
    import numpy as np
    import torch
    from PIL import Image
    X = torch.empty((len(idx), 3, SIZE, SIZE), dtype=torch.uint8)
    for j, i in enumerate(idx):
        im = Image.open(items[i][0]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        X[j] = torch.from_numpy(np.asarray(im).transpose(2, 0, 1).copy())
    return X


def _net(weights=None):
    import torch
    from transformers import AutoModelForImageClassification
    net = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "good", 1: "defective"},
        label2id={"good": 0, "defective": 1}, ignore_mismatched_sizes=True)
    if weights:
        net.load_state_dict(torch.load(weights, map_location="cpu")["state_dict"])
    return net.to("cuda" if torch.cuda.is_available() else "cpu")


def _prob(net, X):
    """The inspector's probability that each board is defective."""
    import torch
    dev = next(net.parameters()).device
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = (X[i:i + 64].to(dev).float() / 255.0 - mean) / std
            out.append(torch.softmax(net(pixel_values=x).logits, -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def _version_no(name):
    m = re.match(r"^v(\d+)", str(name or ""))
    return int(m.group(1)) if m else 0


def latest_model(project):
    """The newest registered inspector in this project, or None."""
    from clearml import Model
    found = Model.query_models(project_name=project, tags=[MODEL_TAG]) or []
    found = [m for m in found if _version_no(m.name.replace("inspector-", ""))]
    if not found:
        return None
    return max(found, key=lambda m: _version_no(m.name.replace("inspector-", "")))


def _state(ds_id):
    """What the published dataset says: which boards are labelled, and where
    the line has got to."""
    import hyperdataset as hd
    version = hd.latest_published(ds_id) if ds_id else None
    if not version:
        return None, [], 0, None
    frames = [f for f in hd.frames(ds_id, version["id"]) if f["meta"].get("origin") == "line"]
    cursor = max([int(f["meta"].get("line_position") or 0) for f in frames] or [0])
    seed = next((f["meta"].get("line_seed") for f in frames
                 if f["meta"].get("line_seed") is not None), None)
    return version, frames, cursor, seed


# --------------------------------------------------------------------------
def line_arrive(arrive=300, review=60, start=40, seed=0, dataset=DATASET):
    """The next batch off the line, published as the next dataset version."""
    import numpy as np
    from PIL import Image
    import hyperdataset as hd

    task = bind_task()
    ds_id = hd.get_or_create_dataset(dataset, tags=["physical-ai", "pcb", "line"])
    version, frames, cursor, kept_seed = _state(ds_id)
    seed = int(kept_seed) if kept_seed is not None else int(seed)
    items, y, exam, line = _world(seed)
    round_no = _version_no(version.get("name")) if version else 0

    if not version:
        batch = line[:int(start)]
        pick, probs = batch, None
        name = "v1-day1"
        note = "day one: the first %d boards off the line, all labelled" % len(batch)
    else:
        model = latest_model(task.get_project_name()) if task else None
        if not model:
            raise SystemExit("there is a dataset version but no inspector yet -- "
                             "run stage=line_train first")
        batch = line[cursor:cursor + int(arrive)]
        if not len(batch):
            return {"arrived": 0, "note": "the line has no more boards",
                    "version": version.get("name")}
        net = _net(model.get_local_copy())
        probs = _prob(net, _tensors(items, batch))
        order = np.argsort(np.abs(probs - 0.5))[:int(review)]
        pick, probs_pick = batch[order], probs[order]
        name = "%s-line" % hd.next_version_name(ds_id)
        note = ("%d boards arrived; %s flagged %d; the %d it was least sure about "
                "were reviewed" % (len(batch), model.name, int((probs > 0.5).sum()), len(pick)))
    cursor_after = cursor + len(batch)

    vid = hd.create_draft(ds_id, name, parent=version["id"] if version else None, comment=note)
    dest = hd.files_dest("pcb-line", name)
    tmp = os.path.join(CACHE, "line", name)
    os.makedirs(tmp, exist_ok=True)
    out = []
    for j, i in enumerate(pick):
        path, board, label = items[i]
        key = _key(path)
        small = os.path.join(tmp, key.replace("/", "_").rsplit(".", 1)[0] + ".jpg")
        Image.open(path).convert("RGB").resize((640, 640), Image.BILINEAR).save(small, quality=88)
        meta = hd.frame_meta(origin="line", texture=board, kind="defective" if label else "good",
                             verified=True, key=key, line_round=round_no + 1 if version else 1,
                             line_position=cursor_after, line_seed=seed,
                             inspector_score=(None if probs is None else round(float(probs_pick[j]), 4)))
        out.append(hd.make_frame(hd.upload_image(small, dest),
                                 labels=["defective" if label else "good"], meta=meta,
                                 content_type="image/jpeg", source_id="top"))
        if task and j < 16:
            task.get_logger().report_image(
                title="boards labelled this round",
                series="%02d %s %s%s" % (j, board, "DEFECTIVE" if label else "good",
                                         "" if probs is None else "  inspector said %.2f" % probs_pick[j]),
                iteration=0, local_path=small)
    hd.add_frames(vid, out)
    hd.commit(vid, publish=True)
    counts = hd.stats(vid)["labels"]
    print("=" * 66, flush=True)
    print(note, flush=True)
    print("published %s: %s" % (name, counts), flush=True)
    print("=" * 66, flush=True)
    recorded = {}
    if task:
        logger = task.get_logger()
        for k, v in (("boards arrived", len(batch)), ("boards reviewed", len(pick)),
                     ("defects found in review", int(y[pick].sum())),
                     ("boards labelled so far", sum(counts.values())),
                     ("defective boards labelled so far", counts.get("defective", 0))):
            logger.report_single_value(k, v)
        if probs is not None:
            # What the current inspector made of the whole batch, and which
            # part of it went to review.
            edges = np.linspace(0, 1, 21)
            logger.report_histogram(
                title="inspector score on the arriving boards", series="all arrivals",
                values=np.histogram(probs, bins=edges)[0], iteration=0,
                xlabels=["%.2f" % e for e in edges[:-1]], xaxis="probability defective",
                yaxis="boards")
            logger.report_histogram(
                title="inspector score on the arriving boards", series="sent for review",
                values=np.histogram(probs_pick, bins=edges)[0], iteration=0,
                xlabels=["%.2f" % e for e in edges[:-1]], xaxis="probability defective",
                yaxis="boards")
        logger.report_table(
            title="dataset version", series=name, iteration=0,
            table_plot=[["label", "boards"]] + [[k, v] for k, v in sorted(counts.items())])
        task.flush(wait_for_uploads=True)
        recorded = {"dataview": hd.attach_dataview(
                        task.id, ds_id, vid, "%s %s" % (dataset, name),
                        "the dataset version this task published",
                        {"good": 0, "defective": 1}),
                    "debug_samples": min(16, len(pick)),
                    "plots": ["dataset version"] + (["inspector score on the arriving boards"]
                                                    if probs is not None else [])}
    return {"version": name, "recorded": recorded, "arrived": int(len(batch)), "reviewed": int(len(pick)),
            "defects_in_review": int(y[pick].sum()),
            "defects_in_batch": int(y[batch].sum()),
            "flagged_by_inspector": None if probs is None else int((probs > 0.5).sum()),
            "labelled": counts, "line_position": int(cursor_after),
            "boards_left_on_line": int(len(line) - cursor_after)}


# --------------------------------------------------------------------------
def line_train(steps=150, lr=1e-4, dataset=DATASET):
    """Fine-tune the inspector on the latest version, score it, register it."""
    import numpy as np
    import torch
    from clearml import OutputModel
    from clearml.backend_api import Session
    from sklearn.metrics import roc_auc_score
    import hyperdataset as hd

    task = bind_task()
    ds_id = hd.find_dataset(dataset)
    version, frames, _cursor, seed = _state(ds_id)
    if not version:
        raise SystemExit("nothing published yet -- run stage=line_arrive first")
    items, y, exam, _line = _world(seed or 0)
    index = {_key(p): i for i, (p, _, _) in enumerate(items)}
    labelled = np.array(sorted({index[f["meta"]["key"]] for f in frames
                                if f["meta"].get("key") in index}))
    if set(labelled) & set(exam):
        raise SystemExit("an exam board is in the training set -- refusing to score")
    print("%s: %d labelled boards, %d defective" % (version.get("name"), len(labelled),
                                                    int(y[labelled].sum())), flush=True)

    previous = latest_model(task.get_project_name()) if task else None
    name = "inspector-%s" % version.get("name")
    if previous and previous.name == name:
        raise SystemExit("%s is already trained on %s -- run stage=line_arrive for new boards"
                         % (name, version.get("name")))
    net = _net(previous.get_local_copy() if previous else None)
    print("starting from", previous.name if previous else BASE_MODEL + " (no inspector yet)",
          flush=True)

    Xe = _tensors(items, exam)

    def score(n):
        p = _prob(n, Xe)
        t = y[exam]
        pred = p > 0.5
        return {"exam_score": round(float(roc_auc_score(t, p)), 3),
                "defects_caught": round(float(pred[t == 1].mean()), 3),
                "false_alarms": round(float(pred[t == 0].mean()), 3)}

    before = score(net) if previous else None
    if task and before:
        for k, v in before.items():
            task.get_logger().report_scalar("exam during training", k.replace("_", " "), v, 0)

    Xt = _tensors(items, labelled)
    yt = torch.tensor(y[labelled])
    dev = next(net.parameters()).device
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    rng = np.random.default_rng(_version_no(version.get("name")))
    torch.manual_seed(_version_no(version.get("name")))
    good, bad = np.where(y[labelled] == 0)[0], np.where(y[labelled] == 1)[0]
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr))
    net.train()
    for step in range(int(steps)):
        # Good and defective boards in equal numbers, or the one-in-eleven
        # defect rate teaches the model to call everything good.
        b = (np.concatenate([rng.choice(good, 16), rng.choice(bad, 16)])
             if len(good) and len(bad) else rng.choice(len(labelled), 32))
        x = (Xt[b].to(dev).float() / 255.0 - mean) / std
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[3])
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[2])
        opt.zero_grad()
        loss = net(pixel_values=x, labels=yt[b].to(dev)).loss
        loss.backward()
        opt.step()
        if task and step % 10 == 0:
            task.get_logger().report_scalar("training", "loss", float(loss), step)
        if task and (step + 1) % 50 == 0 and step + 1 < int(steps):
            mid = score(net)
            for k, v in mid.items():
                task.get_logger().report_scalar("exam during training", k.replace("_", " "),
                                                v, step + 1)
            net.train()

    after = score(net)
    if task:
        for k, v in after.items():
            task.get_logger().report_scalar("exam during training", k.replace("_", " "),
                                            v, int(steps))
    result = dict(after, model=name, version=version.get("name"),
                  boards=int(len(labelled)), defects=int(y[labelled].sum()),
                  exam_boards=int(len(exam)), exam_defective=int(y[exam].sum()),
                  started_from=previous.name if previous else BASE_MODEL,
                  previous=before)
    print("=" * 66, flush=True)
    if before:
        print("before  %s  %s" % (previous.name, before), flush=True)
    print("after   %s  %s" % (name, after), flush=True)
    print("=" * 66, flush=True)

    out_dir = os.path.join(CACHE, "models", name)
    os.makedirs(out_dir, exist_ok=True)
    weights = os.path.join(out_dir, "model.pt")
    # Saved through a buffer: the SDK registers every file torch.save writes as
    # a model of its own, which would put two models per round in the list.
    import io
    buf = io.BytesIO()
    torch.save({"state_dict": net.state_dict(), "hf_model": BASE_MODEL,
                "id2label": {0: "good", 1: "defective"}}, buf)
    with open(weights, "wb") as fh:
        fh.write(buf.getvalue())
    if task:
        logger = task.get_logger()
        for k, v in after.items():
            logger.report_single_value(k.replace("_", " "), v)
        om = OutputModel(task=task, name=name, framework="PyTorch", tags=[MODEL_TAG])
        om.update_design(config_dict=result)
        om.update_weights(weights_filename=weights, auto_delete_file=False,
                          upload_uri=Session.get_files_server_host())
        result["model_id"] = om.id
        # The result travels with the model, so the project's model list shows
        # every inspector beside what it scored.
        for k in ("defects_caught", "false_alarms", "exam_score", "boards", "defects"):
            om.set_metadata(k, str(result[k]))
        om.set_metadata("dataset_version", str(result["version"]))

        # One chart of every inspector so far, by dataset version. Each earlier
        # result is read from its registered model, not from an earlier task.
        from clearml import Model
        rows = {}
        for m in Model.query_models(project_name=task.get_project_name(), tags=[MODEL_TAG]) or []:
            cfg = m.config_dict or {}
            if cfg.get("version") and cfg.get("defects_caught") is not None:
                rows[_version_no(cfg["version"])] = cfg
        rows[_version_no(result["version"])] = result
        history = [rows[k] for k in sorted(rows)]
        for h in history:
            n = _version_no(h["version"])
            logger.report_scalar("inspector by dataset version", "defects caught",
                                 float(h["defects_caught"]), n)
            logger.report_scalar("inspector by dataset version", "false alarms",
                                 float(h["false_alarms"]), n)
            logger.report_scalar("inspector by dataset version", "exam score",
                                 float(h["exam_score"]), n)
        logger.report_table(
            title="every inspector so far", series="exam results", iteration=0,
            table_plot=[["model", "dataset version", "boards", "defects",
                         "defects caught", "false alarms", "exam score"]]
            + [[h.get("model"), h["version"], h.get("boards"), h.get("defects"),
                "%.0f%%" % (100 * float(h["defects_caught"])),
                "%.1f%%" % (100 * float(h["false_alarms"])), h["exam_score"]] for h in history])
        # What the inspector got right and wrong on the exam, as pictures.
        from PIL import Image
        p_exam = _prob(net, Xe)
        t_exam = y[exam]
        show_dir = os.path.join(CACHE, "line", "exam-" + name)
        os.makedirs(show_dir, exist_ok=True)
        groups = (
            ("defects caught", [i for i in np.argsort(-p_exam) if t_exam[i] == 1 and p_exam[i] > 0.5]),
            ("defects missed", [i for i in np.argsort(p_exam) if t_exam[i] == 1 and p_exam[i] <= 0.5]),
            ("false alarms", [i for i in np.argsort(-p_exam) if t_exam[i] == 0 and p_exam[i] > 0.5]))
        shown = 0
        for title, idxs in groups:
            for j, i in enumerate(idxs[:6]):
                path, board, _l = items[exam[i]]
                small = os.path.join(show_dir, "%s_%02d.jpg" % (title.replace(" ", "_"), j))
                Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR).save(small, quality=88)
                logger.report_image(title=title, series="%02d %s  inspector said %.2f"
                                    % (j, board, p_exam[i]), iteration=0, local_path=small)
                shown += 1
        edges = np.linspace(0, 1, 21)
        for series, mask in (("good boards", t_exam == 0), ("defective boards", t_exam == 1)):
            logger.report_histogram(
                title="inspector score on the exam", series=series,
                values=np.histogram(p_exam[mask], bins=edges)[0], iteration=0,
                xlabels=["%.2f" % e for e in edges[:-1]], xaxis="probability defective",
                yaxis="boards")
        pred = p_exam > 0.5
        logger.report_confusion_matrix(
            title="exam confusion matrix", series=name, iteration=0,
            matrix=[[int(((t_exam == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)],
            xlabels=["called good", "called defective"], ylabels=["good", "defective"],
            xaxis="inspector", yaxis="truth")
        exam_board = np.array([items[i][1] for i in exam])
        logger.report_table(
            title="defects caught by board type", series=name, iteration=0,
            table_plot=[["board type", "defective on the exam", "caught", "share"]]
            + [[b, int(((exam_board == b) & (t_exam == 1)).sum()),
                int(((exam_board == b) & (t_exam == 1) & pred).sum()),
                "%.0f%%" % (100.0 * ((exam_board == b) & (t_exam == 1) & pred).sum()
                            / max(1, ((exam_board == b) & (t_exam == 1)).sum()))] for b in BOARDS])
        task.flush(wait_for_uploads=True)
        result["recorded"] = {
            "dataview": hd.attach_dataview(
                task.id, ds_id, version["id"], "%s %s" % (dataset, version.get("name")),
                "the dataset version this inspector was trained on",
                {"good": 0, "defective": 1}),
            "debug_samples": shown,
            "scalars": ["training", "exam during training", "inspector by dataset version"],
            "plots": ["every inspector so far", "inspector score on the exam",
                      "exam confusion matrix", "defects caught by board type"]}
        result["history"] = [{"version": h["version"], "defects_caught": h["defects_caught"],
                              "false_alarms": h["false_alarms"], "exam_score": h["exam_score"]}
                             for h in history]
    return result
