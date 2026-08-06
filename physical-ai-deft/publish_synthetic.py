"""Publish AnomalyGen's output as the next HyperDataset version.

    read the gap -> [AnomalyGen generated] -> publish v2 -> read the gap again

This is the step that closes the loop, and the only one where the platform
rather than NVIDIA is doing the work. AnomalyGen leaves 24 images and a
SDG_result.csv on disk; on its own that is a folder. This turns it into a
published, immutable dataset version whose every frame can be traced back to the
clean board it came from, the mask that placed the defect, and the parameters
that generated it.

WHY THE CSV MATTERS MORE THAN THE IMAGES
-----------------------------------------
NVIDIA writes SDG_result.csv with a row per generated image: the clean source
image, the mask file, the anomaly type, guidance, steps, seed, PSNR and
guardrail_pass. That is provenance they emit for free, and it would be thrown
away by anyone who just globbed the output directory. Every column becomes frame
metadata here, so "what produced this training example?" is answerable from the
dataset itself rather than from a log nobody kept.

VERIFIED vs PENDING
--------------------
`guardrail_pass` is NVIDIA's own safety verdict, not a quality score -- it says
the image cleared the content filter, not that the defect is convincing. So a
frame that passes the guardrail is published under its real defect class only
when phases 4-7 (nn_score against the real examples) have also accepted it.
Without that, frames land as `pending-review` and do not count toward closing
any gap. Same rule as before: a label is a claim about what is in the picture.
"""
import csv
import os
import sys

CACHE = os.environ.get("DEFT_CACHE", "/cache")


def _nn_scores(results):
    """{basename: nn_score} from phase 4, or {} if it has not run.

    Per-frame rather than a blanket flag: phase 4's whole point is that some
    generated frames are good and some are not, and it says which.
    """
    import csv as _csv
    path = os.path.join(results, "per_sample.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in _csv.DictReader(fh):
            try:
                out[os.path.basename(r.get("path") or "")] = float(r["nn_score"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def publish_synthetic(hyperdataset_name="PCB Inspection",
                      dataset_name="pcb-uc1",
                      version_name=None,
                      target_per_class=60,
                      nn_threshold=None):
    """Publish the generated frames, parented on the latest published version."""
    import hyperdataset as hd
    from clearml import Task

    task = Task.current_task()
    results = os.path.join(CACHE, "results", dataset_name, "original")
    csv_path = os.path.join(results, "SDG_result.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("no SDG_result.csv under %s -- run generation first"
                         % results)

    ds_id = hd.get_or_create_dataset(hyperdataset_name,
                                     tags=["physical-ai", "pcb"])
    parent = hd.latest_published(ds_id)
    if not parent:
        raise SystemExit("no published version to parent on -- register the "
                         "real data first")

    before = hd.stats(parent["id"])["labels"]
    print("=" * 66, flush=True)
    print("BEFORE  (%s)" % parent.get("name"), flush=True)
    for k, v in sorted(before.items(), key=lambda kv: -kv[1]):
        print("  %-24s %4d" % (k, v), flush=True)

    # Derived, not constant: this runs once per enrichment round.
    if not version_name:
        version_name = "%s-anomalygen" % hd.next_version_name(ds_id)
    print("publishing as", version_name, flush=True)
    version_id = hd.create_draft(
        ds_id, version_name, parent=parent["id"],
        comment="NVIDIA AnomalyGen, mask-conditioned on the board's CAD")
    dest = hd.files_dest("pcb-synthetic", version_name)

    # DO NOT PUBLISH THE SAME GENERATION RUN TWICE.
    #
    # A version is parented on the previous one, so it inherits its frames. Run
    # publish twice against the same output directory and those images are in
    # the dataset twice -- observed live: v2 held 24 synthetic frames, v3
    # inherited them and added the same 24 again, so one generation run of 24
    # became 48 frames. Nothing errors; the counts the agent reasons over just
    # quietly inflate, which is the failure mode this whole design exists to
    # avoid.
    #
    # Generated filenames are deterministic (<anomaly_type>_<NNNNN>.png), so the
    # basenames already in the parent are enough to recognise a repeat.
    already = {os.path.basename(u)
               for u in hd.source_uris(ds_id, parent["id"])}

    scores = _nn_scores(results)
    if scores and nn_threshold is None:
        import statistics
        nn_threshold = statistics.median(scores.values())
    if scores:
        print("phase 4 scores present for %d frame(s); threshold %.3f"
              % (len(scores), nn_threshold), flush=True)
    else:
        print("no per_sample.csv -- phase 4 has not run, so every frame "
              "publishes as pending-review", flush=True)

    frames, skipped, accepted, duplicates = [], 0, 0, 0
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            path = row.get("output_filename") or ""
            if not path or not os.path.exists(path):
                skipped += 1
                continue
            if os.path.basename(path) in already:
                duplicates += 1
                continue
            # NVIDIA's guardrail is a SAFETY verdict, not a quality one. A frame
            # that fails it is not published at all.
            if str(row.get("guardrail_pass", "1")).strip() not in ("1", "True", "true"):
                skipped += 1
                continue

            defect = (row.get("anomaly_type") or "").split("+")[-1] or "unknown"
            texture = (row.get("anomaly_type") or "").split("+")[0] or None

            # THE LABEL IS EARNED, PER FRAME. A frame gets its real defect class
            # only if phase 4 scored it at or above the threshold; otherwise it
            # publishes as pending-review and counts toward closing nothing.
            nn = scores.get(os.path.basename(path))
            verified = nn is not None and nn >= nn_threshold
            accepted += 1 if verified else 0

            uri = hd.upload_image(path, dest)
            frames.append(hd.make_frame(
                uri,
                labels=[defect] if verified else ["pending-review"],
                meta=hd.frame_meta(
                    origin="synthetic", texture=texture, defect=defect,
                    kind="anomaly", generator="Cosmos-AnomalyGen-PCB-2B",
                    parent_version=parent["id"], verified=bool(verified),
                    nn_score=nn,
                    # Straight from NVIDIA's own record -- this is what makes a
                    # generated frame auditable rather than merely present.
                    clean_source=os.path.basename(row.get("image_filename") or ""),
                    mask=os.path.basename(row.get("mask_filename") or ""),
                    guidance=row.get("guidance"),
                    num_steps=row.get("num_steps"),
                    seed=row.get("seed"),
                    psnr=row.get("PSNR"),
                    guardrail_pass=row.get("guardrail_pass"))))

    saved = hd.add_frames(version_id, frames) if frames else 0
    hd.commit(version_id, publish=True)
    after = hd.stats(version_id)["labels"]

    print("=" * 66, flush=True)
    print("AFTER   (%s)  -- %d published (%d verified), %d skipped, "
          "%d already present" % (version_name, saved, accepted, skipped,
                                  duplicates), flush=True)
    for k, v in sorted(after.items(), key=lambda kv: -kv[1]):
        delta = v - before.get(k, 0)
        print("  %-24s %4d %s" % (k, v, ("(+%d)" % delta) if delta else ""), flush=True)

    housekeeping = {"clean", "mask", "pending-review"}
    gap = {k: target_per_class - v for k, v in after.items()
           if k not in housekeeping and v < target_per_class}
    print("=" * 66, flush=True)
    print("still short of %d:" % target_per_class,
          gap or "nothing -- every class is at target", flush=True)
    if after.get("pending-review"):
        print("%d frame(s) awaiting nn_score review -- they count toward "
              "nothing until phases 4-7 accept them" % after["pending-review"],
              flush=True)
    print("=" * 66, flush=True)

    if task:
        logger = task.get_logger()
        labels = sorted(set(before) | set(after))
        logger.report_table(
            title="the loop moved the numbers", series="frames per label",
            iteration=0,
            table_plot=[["label", parent.get("name"), version_name, "delta"]]
                       + [[k, before.get(k, 0), after.get(k, 0),
                           after.get(k, 0) - before.get(k, 0)] for k in labels])
        task.upload_artifact("gap", gap)

    return {"version_id": version_id, "version_name": version_name,
            "published": saved, "skipped": skipped, "gap": gap}


if __name__ == "__main__":
    thr = None
    if "--nn-threshold" in sys.argv:
        thr = float(sys.argv[sys.argv.index("--nn-threshold") + 1])
    publish_synthetic(nn_threshold=thr)
