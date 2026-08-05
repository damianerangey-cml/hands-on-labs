"""HyperDatasets for the Physical AI Data Factory, over the stock clearml SDK.

WHY THIS EXISTS, AND WHY IT IS NOT `allegroai`
----------------------------------------------
HyperDatasets are an Enterprise feature and their Python SDK (`allegroai`,
`DatasetVersion` / `SingleFrame`) is not on public PyPI -- it comes from
ClearML's private index. Every stage of this lab runs inside one of NVIDIA's
containers with `CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1`, so anything we need
has to be pip-installable from inside a TAO or Cosmos image with no extra index
credentials. A private-index dependency is a bad trade for what we actually use,
which is six endpoints.

So this module talks to those endpoints directly. `Session.send_request` carries
the task's own credentials, so a stage authenticates as itself with no extra
configuration, and the only import is `clearml` -- already installed.

WHY HYPERDATASETS AT ALL, RATHER THAN `clearml.Dataset`
-------------------------------------------------------
A `Dataset` is a versioned bundle of files. To answer "how many solder-bridge
examples do I have?" you download it and count -- which for this lab means
pulling every image onto a GPU worker to learn a number.

A HyperDataset stores a FRAME per image, each carrying labels and a metadata
dict, and the server aggregates them. `datasets.get_stats` returns the per-label
counts for a version. So the agent asks a question and gets an answer, without
the pixels ever moving. That is what makes the "decide from the metadata alone"
step in this lab cheap enough to run in a loop, and what lets it run against a
customer's inspection data without their images leaving their storage.

THE SIX ENDPOINTS
-----------------
    datasets.create                  a dataset, once
    datasets.create_version          a DRAFT version to write into
    datasets.save_frames             frames, batched
    datasets.commit_version          seal it, compute stats, publish
    datasets.get_versions            find the latest PUBLISHED version
    datasets.get_stats               per-label counts for a version

Full schema: allegro-engine/server/schema/services/datasets.conf.
"""
from __future__ import annotations

import os
import uuid

_FRAME_BATCH = 500          # save_frames payload size; keeps requests well clear
                            # of the apiserver body limit on large ingests


class HyperDatasetError(RuntimeError):
    pass


def _session():
    from clearml.backend_api import Session
    return Session()


def _call(action: str, payload: dict, service: str = "datasets") -> dict:
    """POST one apiserver call and return its `data` block.

    Raises on transport failure AND on a non-200, because a partially-written
    dataset version is worse than a stage that stops: the frames that did land
    would be committed as if they were the whole thing, and every count taken
    from that version afterwards would be quietly wrong.
    """
    res = _session().send_request(service=service, action=action,
                                  json=payload, method="post")
    if res.status_code != 200:
        raise HyperDatasetError(
            "%s.%s -> HTTP %s: %s" % (service, action, res.status_code,
                                      res.text[:600]))
    body = res.json() or {}
    return body.get("data", body)


# ---------------------------------------------------------------- frames ----
def make_frame(uri: str,
               labels: list[str] | None = None,
               meta: dict | None = None,
               width: int | None = None,
               height: int | None = None,
               content_type: str = "image/png") -> dict:
    """One frame document: where the image is, what is in it, what we know.

    `labels` become an ROI. That matters more than it looks -- `get_stats`
    aggregates ROI labels, so a frame with no ROI is invisible to the very
    query this whole design rests on. A frame that is a clean board still gets
    a label ('clean'), because "how many clean boards" is a real question and
    an unlabelled frame cannot answer it.

    We attach ONE whole-image ROI rather than a bounding box: this dataset is
    classification-style (the label describes the board, not a located defect).
    When mask-conditioned generation lands, the generated region is known and
    the ROI can carry a real polygon without changing anything here.

    `meta` is free-form BUT effectively declares a schema -- frames in one
    dataset should agree on keys and types, or the UI's filters get ragged. The
    keys this lab commits to are in `frame_meta()`.
    """
    source = {"id": uuid.uuid4().hex[:16], "uri": uri,
              "content_type": content_type, "timestamp": 0}
    if width:
        source["width"] = width
    if height:
        source["height"] = height

    frame: dict = {"sources": [source], "meta": dict(meta or {})}
    if labels:
        frame["rois"] = [{"label": list(labels),
                          "sources": [source["id"]],
                          "meta": {"kind": "whole-image"}}]
    return frame


def frame_meta(*, origin: str, texture: str | None = None,
               defect: str | None = None, kind: str | None = None,
               generator: str | None = None,
               parent_version: str | None = None,
               verified: bool | None = None,
               **extra) -> dict:
    """The metadata contract for this lab, in one place.

    `origin` ('real' | 'synthetic') is the load-bearing one: it is what lets a
    later reader ask "what did this model actually train on?" and get a real
    answer. Keep it on every frame, including the real ones -- a field that is
    only present on synthetic frames cannot distinguish 'real' from 'nobody set
    it', which is the distinction the whole lineage story depends on.
    """
    meta = {"origin": origin}
    for k, v in (("texture", texture), ("defect", defect), ("kind", kind),
                 ("generator", generator), ("parent_version", parent_version)):
        if v is not None:
            meta[k] = v
    if verified is not None:
        meta["verified"] = bool(verified)
    meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


# --------------------------------------------------------------- dataset ----
def get_or_create_dataset(name: str, tags: list[str] | None = None) -> str:
    """Dataset id for `name`, creating it only if it is not already there.

    Idempotent on purpose: every pass of the loop calls this, and a second
    dataset with the same name would silently split the version history in two.
    """
    found = _call("get_all", {"name": "^%s$" % name, "page": 0,
                              "page_size": 10}).get("datasets") or []
    for d in found:
        if d.get("name") == name:
            return d["id"]
    created = _call("create", {"name": name, "tags": list(tags or [])})
    ds_id = created.get("id")
    if not ds_id:
        raise HyperDatasetError("datasets.create returned no id: %r" % created)
    return ds_id


def create_draft(dataset_id: str, version_name: str,
                 parent: str | None = None,
                 comment: str | None = None) -> str:
    """A writable version. `parent` inherits the parent's frames.

    Each enrichment pass should pass the previous published version as parent,
    so a version is 'everything so far plus what this pass added' rather than
    only the new frames -- otherwise training on the latest version would train
    on the last batch of synthetic images alone.
    """
    payload: dict = {"dataset": dataset_id, "name": version_name}
    if parent:
        payload["parent"] = parent
    if comment:
        payload["comment"] = comment
    out = _call("create_version", payload)
    vid = out.get("id")
    if not vid:
        raise HyperDatasetError("create_version returned no id: %r" % out)
    return vid


def add_frames(version_id: str, frames: list[dict]) -> int:
    """Write frames into a draft version, batched. Returns the number saved.

    Raises if the server reports ANY failure. A dataset that is 97% of what you
    think it is will not announce itself later; it will just make the counts
    slightly wrong, which is the one thing this design cannot tolerate.
    """
    saved = 0
    for i in range(0, len(frames), _FRAME_BATCH):
        batch = frames[i:i + _FRAME_BATCH]
        out = _call("save_frames", {"version": version_id, "frames": batch})
        failed = out.get("failed") or 0
        if failed:
            raise HyperDatasetError(
                "save_frames: %d of %d failed: %s"
                % (failed, len(batch), (out.get("errors") or [])[:3]))
        saved += out.get("saved") or 0
    return saved


def commit(version_id: str, publish: bool = True) -> dict:
    """Seal the draft, compute stats, and publish it.

    `calculate_stats` is not optional for this lab: it is what populates the
    per-label counts that `stats()` reads back, and therefore what the agent's
    next pass reasons over. Committing without it produces a version that looks
    fine in the UI and answers 'nothing' to the only question we ask of it.
    """
    return _call("commit_version", {"version": version_id,
                                    "calculate_stats": True,
                                    "publish": bool(publish)})


# ------------------------------------------------------ read side (cheap) ----
def latest_published(dataset_id: str) -> dict | None:
    """The newest PUBLISHED version, or None if the dataset has only drafts.

    Published-only is the point. A draft is a version someone is still writing
    into; training on it, or measuring a gap from it, means reading a moving
    target. `only_published` makes 'the latest version' mean something.
    """
    out = _call("get_versions", {"dataset": dataset_id, "only_published": True,
                                 "page": 0, "page_size": 50})
    versions = out.get("versions") or []
    if not versions:
        return None
    # The API does not promise an order here, so sort rather than assume.
    # 'created' is present on every version; fall back to name for old records.
    versions.sort(key=lambda v: (v.get("created") or "", v.get("name") or ""))
    return versions[-1]


def stats(version_id: str) -> dict:
    """Per-label counts for a version, computed server-side.

    THIS IS THE WHOLE POINT OF THE MODULE. No frames are downloaded, no images
    are read; the agent gets `{'solder_bridge': 8, 'missing_component': 412}`
    and can decide what to generate from that alone.

    Returns {'labels': {label: count}, 'frames': {...}, 'content_types': {...}}
    with the API's list-of-{name,count} flattened into plain dicts, because
    every caller here wants to look a label up by name.
    """
    raw = _call("get_stats", {"version": version_id})

    def flatten(rows):
        out = {}
        for r in rows or []:
            name = r.get("name")
            if name is not None:
                out[name] = r.get("count", 0)
        return out

    return {"labels": flatten(raw.get("labels")),
            "frames": flatten(raw.get("frames")),
            "content_types": flatten(raw.get("content_types"))}


def label_count(version_id: str, labels: list[str]) -> int:
    """Frames in `version_id` carrying at least one of `labels`.

    `stats()` answers 'how many of each' in one call and is what you usually
    want; this answers 'how many of THESE' for a specific check, e.g. verifying
    a pass actually closed the gap it set out to close.
    """
    out = _call("get_count", {"version": version_id, "labels": list(labels)},
                service="frames")
    return out.get("count", 0)


# ------------------------------------------------------------- uploading ----
def upload_image(local_path: str, dest_folder: str) -> str:
    """Put an image somewhere the server and the UI can both resolve.

    A frame is a POINTER; the bytes stay wherever you put them. On this lab
    that is the tenant's own fileserver, so a HyperDataset version is
    self-contained and previews render in the web UI. In a customer deployment
    this is the step you would NOT do -- their images already sit in their own
    bucket, and you would register those URIs directly, which is precisely the
    property that makes this approach viable on data that cannot be copied.
    """
    from clearml.storage import StorageManager
    name = os.path.basename(local_path)
    return StorageManager.upload_file(
        local_file=local_path, remote_url="%s/%s" % (dest_folder.rstrip("/"), name))
