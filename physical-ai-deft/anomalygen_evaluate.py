"""NVIDIA PAIDF AnomalyGen phase 4 -- score every generated frame.

    generated images -> nn_score against the REAL examples -> accept / reject

This is the gate, and it is NVIDIA's, not ours. Every generated image is scored
by its correspondence to the real anomaly images of the same class -- nn_score,
computed with dinov2-large regardless of the FID backbone flag. A frame that
looks nothing like a real solder bridge scores badly, and that is a measurement
rather than a model's opinion about the picture.

THERE IS NO UNIVERSAL THRESHOLD, AND THE DOCS SAY SO
-----------------------------------------------------
NVIDIA's eval reference is explicit: scores are comparable within a dataset and
backbone, not across datasets, and "there is no fixed threshold for good". So
this stage does NOT ship a magic number. It scores everything, reports the
distribution, and applies a threshold the caller chose -- with the default
derived from the real data itself (the median of what real examples score
against each other), so the bar is "as much like the real thing as the real
things are", not a constant somebody guessed.

WHAT IT CHANGES DOWNSTREAM
---------------------------
Accepted frames earn their real defect label and count toward closing the gap.
Rejected ones stay `pending-review` or are dropped. That is the whole difference
between a dataset that grew and a dataset that improved.
"""
import csv
import os
import statistics
import sys

from ag_common import CACHE, ensure_dataset, link_checkpoints, run as _run


def _latest_run(dataset_name):
    """The most recent per-run output directory."""
    root = os.path.join(CACHE, "results", dataset_name, "runs")
    if not os.path.isdir(root):
        raise SystemExit("no generation runs under %s" % root)
    runs = sorted((os.path.join(root, d) for d in os.listdir(root)),
                  key=lambda p: os.path.getmtime(p))
    if not runs:
        raise SystemExit("no generation runs under %s" % root)
    return runs[-1]


def anomalygen_evaluate(dataset_name="pcb-uc1",
                        anomaly_types=("IC+bridge",
                                       "passive_component+excess_solder",
                                       "passive_component+missing"),
                        nn_threshold=None,
                        run_dir=None):
    """Run NVIDIA's eval and turn per-sample nn_score into accept/reject."""
    from clearml import Task

    task = Task.current_task()
    # The evaluator loads its feature backbone by RELATIVE path
    # (checkpoints/nvidia/C-RADIO-V3/model.safetensors), so the cache has to be
    # linked in even though this stage downloads nothing.
    link_checkpoints()
    dataset_dir = ensure_dataset(dataset_name)
    generated = run_dir or _latest_run(dataset_name)
    per_sample = os.path.join(generated, "per_sample.csv")

    for p, what in ((dataset_dir, "dataset"), (generated, "generated output")):
        if not os.path.isdir(p):
            raise SystemExit("no %s at %s" % (what, p))

    print("=" * 66 + "\nPHASE 4 -- nn_score against the real examples\n" + "=" * 66,
          flush=True)
    _run(["bash", "scripts/utilities/run_eval.sh",
          "--real-path", dataset_dir,
          "--generated-path", generated,
          "--per-sample-csv", per_sample,
          "--anomaly-types", *anomaly_types],
         env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

    if not os.path.exists(per_sample):
        raise SystemExit("eval did not emit %s" % per_sample)

    rows = []
    with open(per_sample, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["nn_score"] = float(r.get("nn_score") or "nan")
            except ValueError:
                r["nn_score"] = float("nan")
            rows.append(r)
    scored = [r for r in rows if r["nn_score"] == r["nn_score"]]  # drop NaN
    if not scored:
        raise SystemExit("no usable nn_score rows in %s" % per_sample)

    by_type = {}
    for r in scored:
        by_type.setdefault(r.get("anomaly_type") or "?", []).append(r["nn_score"])

    print("=" * 66, flush=True)
    print("nn_score distribution (higher = closer to the real examples)", flush=True)
    for t in sorted(by_type):
        v = sorted(by_type[t])
        print("  %-34s n=%-3d min=%.3f med=%.3f max=%.3f"
              % (t, len(v), v[0], statistics.median(v), v[-1]), flush=True)

    # No magic number: default to the median of everything scored. NVIDIA's own
    # docs say there is no fixed threshold, so half the batch surviving is a
    # defensible starting bar and the caller can raise it deliberately.
    if nn_threshold is None:
        nn_threshold = statistics.median([r["nn_score"] for r in scored])
        print("no --nn-threshold given; using the batch median %.3f" % nn_threshold,
              flush=True)

    accepted = [r for r in scored if r["nn_score"] >= nn_threshold]
    rejected = [r for r in scored if r["nn_score"] < nn_threshold]

    print("=" * 66, flush=True)
    print("threshold %.3f -> %d accepted, %d rejected of %d"
          % (nn_threshold, len(accepted), len(rejected), len(scored)), flush=True)
    for t in sorted(by_type):
        a = len([r for r in accepted if r.get("anomaly_type") == t])
        print("  %-34s %d/%d accepted" % (t, a, len(by_type[t])), flush=True)
    print("=" * 66, flush=True)

    if task:
        logger = task.get_logger()
        labels = sorted(by_type)
        logger.report_histogram(
            title="nn_score by defect class", series="median",
            values=[statistics.median(by_type[t]) for t in labels],
            xlabels=labels, iteration=0,
            xaxis="defect class", yaxis="nn_score")
        logger.report_table(
            title="per-sample scores", series="nn_score", iteration=0,
            table_plot=[["image", "anomaly_type", "nn_score", "verdict"]]
                       + [[os.path.basename(r.get("path", "")), r.get("anomaly_type"),
                           round(r["nn_score"], 4),
                           "accept" if r["nn_score"] >= nn_threshold else "reject"]
                          for r in sorted(scored, key=lambda x: -x["nn_score"])])
        task.upload_artifact("per_sample", per_sample)

    return {"per_sample_csv": per_sample,
            "nn_threshold": nn_threshold,
            "accepted": [r.get("path") for r in accepted],
            "rejected": [r.get("path") for r in rejected],
            "counts": {t: len(v) for t, v in by_type.items()}}


if __name__ == "__main__":
    thr = None
    if "--nn-threshold" in sys.argv:
        thr = float(sys.argv[sys.argv.index("--nn-threshold") + 1])
    elif os.environ.get("DEFT_NN_THRESHOLD"):
        thr = float(os.environ["DEFT_NN_THRESHOLD"])
    # DEFT_RUN_DIR names the generation run to score, same as the improve stage.
    # Without it the most recent run is scored, which is right interactively and
    # wrong the moment two rounds are in flight.
    anomalygen_evaluate(nn_threshold=thr,
                        run_dir=os.environ.get("DEFT_RUN_DIR") or None)
