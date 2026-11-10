"""Register NVIDIA's real PCB data as HyperDataset version `v1-real`.

Downloads `nvidia/Cosmos-AnomalyGen-PCB-Dataset` (86 anomaly images, their
defect masks and 5 clean references) and writes one frame per image, carrying
the defect label the directory tree encodes plus a metadata dictionary. After
this the layout stops mattering: the labels are queryable server-side, which is
what lets the loop read its own gap without downloading anything.
"""


def register_real(hyperdataset_name="PCB Inspection",
                  hf_repo="nvidia/Cosmos-AnomalyGen-PCB-Dataset",
                  version_name="v1-real"):
    """Real inspection data in, as frames with labels and metadata.

    The directory layout carries the labels -- `<texture>/anomaly_image/<defect>`
    and `<texture>/clean_image` -- so the ingest reads the tree once and turns
    it into per-frame labels. After this, nobody needs to know that layout
    again; the labels are queryable.

    IDEMPOTENT, BECAUSE THE DATA IS THE SAME DATA
    ----------------------------------------------
    `v1-real` is NVIDIA's 86 images. Registering them twice would be wrong even
    if the API allowed it -- there is one real dataset, not one per run. And the
    API does not allow it; a second call died on

        HTTP 400 ... A dataset version with the provided name already exists

    which is a rude way to greet somebody whose very first command this is. It
    happens more often than "fresh server" suggests: a HyperDataset can outlive
    the lab that created it, so the second claim of the same slot hits this.

    So if a published `v1-real` is already there, report it and hand back the
    dataset. Deliberately NOT falling through to `next_version_name()` -- a
    `v2-real` holding a second copy of the same 86 images would double every
    count the loop reasons over, which is the one failure this pipeline cannot
    tolerate. A name collision here means "already done", not "pick a new name".
    """
    import os
    import hyperdataset as hd
    from clearml import Task
    from huggingface_hub import snapshot_download

    from ag_common import bind_task
    task = bind_task()
    ds_id = hd.get_or_create_dataset(hyperdataset_name, tags=["physical-ai", "pcb"])
    already = [v for v in (hd._call(
                   "get_versions", {"dataset": ds_id, "only_published": True,
                                    "page": 0, "page_size": 100}
               ).get("versions") or []) if (v.get("name") or "") == version_name]
    if already:
        counts = hd.stats(already[0]["id"]).get("labels", {})
        print("=" * 66, flush=True)
        print("%s already published on this server -- reusing it" % version_name,
              flush=True)
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print("  %-24s %4d" % (k, v), flush=True)
        print("(the real data is registered once; a second copy would double "
              "every count the loop reads)", flush=True)
        print("=" * 66, flush=True)
        return ds_id

    local = snapshot_download(repo_id=hf_repo, repo_type="dataset",
                              token=os.environ.get("HF_TOKEN"))
    print("downloaded", hf_repo, "->", local)

    version_id = hd.create_draft(
        ds_id, version_name,
        comment="NVIDIA %s, ingested verbatim" % hf_repo)
    print("dataset", ds_id, "draft version", version_id)

    dest = hd.files_dest("pcb-real", version_name)

    frames, previewed = [], 0
    logger = task.get_logger() if task else None
    for root, _dirs, files in os.walk(local):
        parts = os.path.relpath(root, local).split(os.sep)
        if "anomaly_image" in parts:
            kind, texture, defect = "anomaly", parts[0], parts[-1]
        elif "clean_image" in parts:
            kind, texture, defect = "clean", parts[0], None
        elif "mask" in parts:
            kind, texture, defect = "mask", parts[0], parts[-1]
        else:
            continue

        for f in sorted(files):
            if not f.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(root, f)
            uri = hd.upload_image(path, dest)
            # Masks are not training examples -- they are the conditioning
            # input for mask-guided generation. Labelling them as defects
            # would double-count every anomaly in the stats.
            labels = ([defect] if kind == "anomaly" and defect else
                      ["clean"] if kind == "clean" else ["mask"])
            frames.append(hd.make_frame(
                uri, labels=labels,
                meta=hd.frame_meta(origin="real", texture=texture,
                                   defect=defect, kind=kind, verified=True),
                content_type="image/png" if f.lower().endswith(".png") else "image/jpeg"))

            if logger and kind in ("anomaly", "clean") and previewed < 12:
                # One grid, not twelve rows: same iteration, distinct series.
                logger.report_image(title="real data",
                                    series="%s_%02d" % (kind, previewed),
                                    iteration=0, local_path=path)
                previewed += 1

    saved = hd.add_frames(version_id, frames)
    hd.commit(version_id, publish=True)
    print("=" * 66)
    print("published %s / %s  --  %d frames" % (hyperdataset_name, version_name, saved))
    print("=" * 66)
    return ds_id
