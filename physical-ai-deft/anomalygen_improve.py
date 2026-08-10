"""NVIDIA PAIDF AnomalyGen phases 5-7 -- the half that makes generation better.

    phase 5  search    re-generate each sample under different (guidance, crop_ratio)
    phase 6  assemble  keep, per sample, whichever attempt scored highest
    phase 7  filter    drop what is still below threshold and REGENERATE replacements

WHY THIS MATTERS MORE THAN IT SOUNDS
-------------------------------------
Without it the pipeline is one-shot: a frame is generated once, scored once, and
if it scores badly it is published as `pending-review` and left there forever.
The gate is honest but powerless -- it can decline to credit a bad frame, it
cannot get a better one. Measured on this dataset, that is not a hypothetical:
`excess_solder` scored a median 0.52 against 0.67 for `missing`, every round,
because a subtle change in solder volume is hard to synthesise at the default
guidance. One-shot, the answer is permanently "that class is bad".

Phases 5-7 are NVIDIA's answer, and they are already written -- `run_round.sh`,
`assemble_searched.py`, `filter_with_regen.py` ship in the container. Nothing
here reimplements them; this module chooses the draws, sequences the three
scripts, and reports what won.

THE DRAWS ARE THE DECISION
---------------------------
`run_round.sh` documents its `--draws` argument as "Claude-chosen hyperparameters
per sample" -- NVIDIA built the search expecting an agent to sit here. The shape
is `{"<sample_index>": {"guidance": <f>, "crop_ratio": <f>}}`, per SAMPLE, so a
policy can treat a stubborn excess_solder frame differently from a bridge that
already scored 0.69.

`draws_for_round` is that seat. Today it holds a deterministic grid, which is
enough to prove the mechanism and cheap to reason about. An agent replaces this
one function and nothing else changes -- which is the point of putting it behind
a named boundary now rather than after the fact.

COST, SAID OUT LOUD
-------------------
Each search round regenerates every sample: `rounds` x the original generation
time (~9 s/image on an A10G, so ~4 min per 24-image round). Phase 7 then makes
up to 5 regeneration attempts for whatever is still short. This stage is the
expensive one, and `rounds=0` skips phase 5 entirely while still running the
filter -- use it when you want the gate without the search.
"""
import csv
import json
import os
import statistics
import sys

from ag_common import CACHE, ensure_dataset, link_checkpoints, run as _run

# NVIDIA's defaults, and the original pass this lab already runs.
BASE_GUIDANCE = 7.0
BASE_CROP_RATIO = 2.0

# The grid the default policy walks. Chosen either side of NVIDIA's default
# rather than around it: a lower guidance lets the model drift further from the
# clean board (more defect, less fidelity), a higher one holds it closer. Which
# direction helps is exactly what is not knowable in advance per defect class --
# so try both and let the score decide.
DEFAULT_GRID = [
    {"guidance": 4.0, "crop_ratio": 1.5},
    {"guidance": 10.0, "crop_ratio": 3.0},
]


def read_scores(per_sample_csv):
    """{sample_index: nn_score} from a per_sample.csv, NaN rows dropped."""
    out = {}
    if not per_sample_csv or not os.path.exists(per_sample_csv):
        return out
    with open(per_sample_csv, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            key = r.get("sample_index") or r.get("index")
            try:
                idx = int(key) if key not in (None, "") else i
            except (TypeError, ValueError):
                idx = i
            try:
                v = float(r.get("nn_score"))
            except (TypeError, ValueError):
                continue
            if v == v:  # not NaN
                out[idx] = v
    return out


def draws_for_round(round_idx, n_samples, scores=None, grid=None):
    """Choose (guidance, crop_ratio) per sample for one search round.

    THIS IS THE SEAT AN AGENT TAKES. It gets the round number, how many samples
    there are, and every sample's score so far -- the same three things a person
    would want before deciding what to try next -- and returns NVIDIA's draws
    dict.

    The default policy is a grid: round N applies the Nth (guidance, crop_ratio)
    pair to every sample. Deliberately not adaptive. An adaptive policy that
    cannot be explained is worse than a grid that can, and phase 6 keeps the
    best attempt per sample regardless, so a round that helps only the stubborn
    frames costs nothing on the frames that were already good.
    """
    grid = grid or DEFAULT_GRID
    params = grid[(round_idx - 1) % len(grid)]
    return {str(i): dict(params) for i in range(n_samples)}


def _count_jsonl(path):
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())


def anomalygen_improve(dataset_name="pcb-uc1",
                       run_dir=None,
                       testcase=None,
                       checkpoint_dir=None,
                       step=14000,
                       model_size="2b",
                       rounds=2,
                       nn_threshold=None,
                       num_sdg=None,
                       allocation=None,
                       anomaly_types=None,
                       seed=0):
    """Run phases 5-7 over an existing generation run, in place under it."""
    from clearml import Task

    task = Task.current_task()
    link_checkpoints()

    dataset_dir = ensure_dataset(dataset_name)
    defect_spec = os.path.join(dataset_dir, "defect_spec.jsonl")
    if not run_dir:
        raise SystemExit("run_dir is required -- phases 5-7 improve a run that "
                         "phase 3 already produced")
    original_csv = os.path.join(run_dir, "per_sample.csv")
    if not os.path.exists(original_csv):
        raise SystemExit("no %s -- run phase 4 (evaluate) before improving"
                         % original_csv)

    testcase = testcase or os.path.join(CACHE, "ag_inference", dataset_name,
                                        "testcase.jsonl")
    if not os.path.exists(testcase):
        raise SystemExit("no base testcase at %s" % testcase)
    n_samples = _count_jsonl(testcase)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(CACHE, "checkpoints", "nvidia",
                                      "Cosmos-AnomalyGen-PCB-2B")
    if not anomaly_types:
        import anomalygen_generate as gen
        anomaly_types = gen.defect_types_from_spec(defect_spec)

    rounds_dir = os.path.join(run_dir, "search")
    searched_dir = os.path.join(run_dir, "searched")
    regens_dir = os.path.join(run_dir, "regens")
    os.makedirs(rounds_dir, exist_ok=True)

    before = read_scores(original_csv)
    if before:
        print("original pass: n=%d median nn_score %.3f"
              % (len(before), statistics.median(before.values())), flush=True)

    # ---- phase 5: search -------------------------------------------------
    for r in range(1, int(rounds) + 1):
        rd = os.path.join(rounds_dir, "round_%03d" % r)
        os.makedirs(rd, exist_ok=True)
        draws = draws_for_round(r, n_samples, scores=before)
        draws_path = os.path.join(rd, "draws.json")
        with open(draws_path, "w") as fh:
            json.dump(draws, fh, indent=1)

        sample = draws.get("0", {})
        print("=" * 66, flush=True)
        print("PHASE 5 -- search round %d of %d: guidance %s, crop_ratio %s"
              % (r, rounds, sample.get("guidance"), sample.get("crop_ratio")),
              flush=True)
        print("=" * 66, flush=True)
        _run(["bash", "scripts/utilities/run_round.sh",
              "--base-jsonl", testcase,
              "--draws", draws_path,
              "--output-dir", rd,
              "--real-path", dataset_dir,
              "--checkpoint-dir", checkpoint_dir,
              "--step", str(step),
              "--model-size", model_size,
              "--seed", str(seed + r),
              "--anomaly-types", *anomaly_types],
             env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

    # ---- phase 6: assemble ----------------------------------------------
    # Runs even with rounds=0: with no round directories it simply carries the
    # original pass through, which keeps the downstream path identical whether
    # or not a search happened.
    print("=" * 66 + "\nPHASE 6 -- keep the best attempt per sample\n" + "=" * 66,
          flush=True)
    _run([sys.executable, "scripts/utilities/assemble_searched.py",
          "--original-dir", run_dir,
          "--original-csv", original_csv,
          "--rounds-dir", rounds_dir,
          "--searched-dir", searched_dir])

    searched_csv = os.path.join(searched_dir, "per_sample.csv")
    after = read_scores(searched_csv)

    # ---- phase 7: filter + regenerate ------------------------------------
    # THE THRESHOLD IS NOT A CONSTANT. Same rule as phase 4: default to the
    # median of the original pass, so the bar is "at least as good as half of
    # what we already had" rather than a number somebody guessed. NVIDIA's own
    # eval docs say there is no fixed threshold for good.
    if nn_threshold is None and before:
        nn_threshold = statistics.median(before.values())
    if nn_threshold is None:
        nn_threshold = 0.0
    num_sdg = int(num_sdg or n_samples)

    print("=" * 66, flush=True)
    print("PHASE 7 -- filter at %.3f and regenerate replacements" % nn_threshold,
          flush=True)
    print("=" * 66, flush=True)
    cmd = [sys.executable, "scripts/utilities/filter_with_regen.py",
           "--searched-dir", searched_dir,
           "--per-sample-csv", searched_csv,
           "--threshold", "%.6f" % nn_threshold,
           "--num-sdg", str(num_sdg),
           "--rounds-dir", rounds_dir,
           "--regens-dir", regens_dir,
           "--real-path", dataset_dir,
           "--checkpoint-dir", checkpoint_dir,
           "--step", str(step),
           "--dataset-dir", dataset_dir,
           "--defect-spec", defect_spec,
           "--model-size", model_size,
           "--anomaly-types", *anomaly_types]
    if allocation:
        # Keep the per-class split the gap asked for. Without this, phase 7
        # backfills whatever is cheapest to regenerate and the round quietly
        # stops honouring the shortfall it was built around.
        cmd += ["--allocation", json.dumps(allocation)]
    _run(cmd, env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

    final_csv = os.path.join(searched_dir, "per_sample.csv")
    final = read_scores(final_csv)

    # ---- what the search actually bought ---------------------------------
    summary = os.path.join(rounds_dir, "search_summary.csv")
    won = {}
    if os.path.exists(summary):
        with open(summary, newline="") as fh:
            for r in csv.DictReader(fh):
                won[r.get("best_round") or "original"] = \
                    won.get(r.get("best_round") or "original", 0) + 1

    # EACH ROUND AS A WHOLE BATCH -- the comparison that is not a selection
    # effect. `after search` is the best of three attempts per sample, so its
    # median rises even if all three draws come from the same distribution;
    # quoting only that number would overstate what the search found. These
    # rows are every sample under one parameter setting, scored the same way,
    # so they say whether a setting is actually better. They are also the
    # finding worth keeping: "guidance 4.0 beats the 7.0 default on this
    # dataset" is something a customer can act on without running any of this.
    per_round = [("original (%.1f/%.1f)" % (BASE_GUIDANCE, BASE_CROP_RATIO),
                  before)]
    for r in range(1, int(rounds) + 1):
        rp = os.path.join(rounds_dir, "round_%03d" % r, "per_sample.csv")
        params = draws_for_round(r, 1).get("0", {})
        per_round.append(("round %d (%.1f/%.1f)"
                          % (r, params.get("guidance", 0),
                             params.get("crop_ratio", 0)), read_scores(rp)))

    print("=" * 66, flush=True)
    print("EACH SETTING, WHOLE BATCH (no best-of picking -- guidance/crop_ratio)",
          flush=True)
    for name, d in per_round:
        if d:
            v = sorted(d.values())
            print("  %-22s n=%-3d min=%.3f med=%.3f max=%.3f"
                  % (name, len(v), v[0], statistics.median(v), v[-1]), flush=True)

    print("=" * 66, flush=True)
    print("IMPROVEMENT", flush=True)
    for name, d in (("original", before), ("after search", after),
                    ("after regen", final)):
        if d:
            v = sorted(d.values())
            print("  %-14s n=%-3d min=%.3f med=%.3f max=%.3f"
                  % (name, len(v), v[0], statistics.median(v), v[-1]), flush=True)
    print("  ('after search' is the best of %d attempts per sample -- part of "
          "that lift is\n   selection, which is why the table above exists)"
          % (int(rounds) + 1), flush=True)
    if won:
        print("  winning attempt per sample: %s"
              % ", ".join("%s=%d" % kv for kv in sorted(won.items())), flush=True)
    if before and final:
        lifted = sum(1 for i, v in final.items() if v > before.get(i, -1))
        print("  %d of %d samples ended better than the original pass"
              % (lifted, len(final)), flush=True)
    print("=" * 66, flush=True)

    if task:
        logger = task.get_logger()
        for i, (name, d) in enumerate((("original", before), ("searched", after),
                                       ("final", final))):
            if d:
                logger.report_scalar("median nn_score", name,
                                     value=statistics.median(d.values()),
                                     iteration=i)
        if won:
            logger.report_table(
                title="which attempt won", series="per sample", iteration=0,
                table_plot=[["attempt", "samples"]]
                           + [[k, v] for k, v in sorted(won.items())])
        logger.report_table(
            title="each setting, whole batch", series="no best-of picking",
            iteration=0,
            table_plot=[["setting (guidance/crop_ratio)", "n", "min", "median", "max"]]
            + [[name, len(d), round(min(d.values()), 4),
                round(statistics.median(d.values()), 4), round(max(d.values()), 4)]
               for name, d in per_round if d])
        if os.path.exists(summary):
            task.upload_artifact("search_summary", summary)

    return {"searched_dir": searched_dir,
            "per_sample_csv": final_csv,
            "nn_threshold": nn_threshold,
            "rounds": int(rounds),
            "won": won,
            "median_before": statistics.median(before.values()) if before else None,
            "median_after": statistics.median(final.values()) if final else None}


if __name__ == "__main__":
    import anomalygen_evaluate as ev
    _rd = os.environ.get("DEFT_RUN_DIR") or ev._latest_run(
        os.environ.get("DEFT_DATASET") or "pcb-uc1")
    anomalygen_improve(run_dir=_rd,
                       rounds=int(os.environ.get("DEFT_SEARCH_ROUNDS") or 2))
