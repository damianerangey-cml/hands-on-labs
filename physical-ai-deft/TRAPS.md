# Traps

Failures that already happened once on NVIDIA's AnomalyGen pipeline. Each one
looked like a different problem than it was. Read the relevant one when a stage
fails; you do not need to read them all first.

### 1. The agent builds a venv and shadows NVIDIA's torch

NVIDIA's image already carries torch 2.10+cu128 and every AnomalyGen
dependency. Left to itself, the agent creates a fresh virtualenv for the task
and NVIDIA's packages are no longer importable. Symptoms range from a bare
`No module named torch` to CUDA version mismatches and missing custom ops.

`run.py` handles this: if the interpreter it was started with cannot import
torch, it finds the image's own Python, installs `clearml` there if it is
missing, and starts again under it. The log says `switching to the image's own
Python` when that happens. Nothing has to be set on the task.

### 2. `from __future__ import annotations` breaks entry points

clearml-agent patches the top of the executed script to inject its own
bootstrap. If the file starts with a `__future__` import, the patched file is a
`SyntaxError` — Python requires `__future__` imports to come first.

Remove it from **entry-point** files only. Modules that are merely imported are
fine.

### 3. Checkpoints are downloaded flat, and read nested

NVIDIA's downloader writes checkpoints flat into the target directory. Their
`run_sdg.sh` reads them from `<dir>/checkpoints/model/`. Nothing warns you; the
stage fails later looking like a missing-weights bug.

`ag_common.link_checkpoints()` creates the symlinks. **Call it in every stage
that touches checkpoints**, not just the first — this was rediscovered three
times because a later stage ran in a fresh pod with the same PVC.

### 4. `spatial_dependency` decides where the defect lands

Automatic Mask Placement routes on the defect spec:

- `cad` — region from the board's own CAD mask + `semantic_segmentation_labels.json`
- `text` — you describe the location; Qwen-VL + SAM2 segment it
- `free` — anywhere

NVIDIA's PCB dataset is `cad`, and the console should say `24 cad2roi`. If you
see `free`, the CAD mask or the segmentation labels did not reach the container
and the defects are landing in arbitrary places — the generated data is
worthless and nothing will tell you so.

### 5. Publishing the same run twice silently doubles the counts

A version inherits its parent's frames. Publish the same output directory twice
and those images are in the dataset twice. Nothing errors — the counts the loop
reasons over quietly inflate, which is the one thing a metadata-driven loop
cannot tolerate. Observed live: v2 held 24 synthetic frames, v3 inherited them
and added the same 24 again.

The guard is **run-scoped**, on the run id embedded in the upload path.
Do not "fix" it to match on basename: generated filenames are deterministic
(`<anomaly_type>_<NNNNN>.png`), so every round produces the same names for
different images. A basename guard blocks every legitimate round after the
first and stalls the loop silently.

### 5b. ...and the searched/ directory breaks that guard a second way

Once phases 5–7 run, the frames worth publishing are not in `runs/<run_id>/` but
in `runs/<run_id>/searched/`. The dedup guard keys on the directory's basename,
and that basename is `searched` for **every** round — so round 2 looks like a
republish of round 1 and is skipped, silently, with no error and no new frames.

`publish_synthetic(run_id=...)` exists for exactly this. Pass the real run id
whenever the directory no longer carries it. This is the fourth time a reused
name has broken this loop (version name, output dir, run id, now the searched
bucket) — if you add a stage that moves where frames live, check this first.

### 6. Three separate reasons rounds fail to accumulate

All three produce the same symptom — the training set never grows, accuracy is
flat, and it reads as "synthetic data changed nothing":

1. **Shared output dir** — every round wrote to the same path and overwrote the
   last. Fixed by a per-run output dir.
2. **Non-accumulating collect** — the trainer read only the latest run dir
   instead of all of them.
3. **Colliding run ids** — `round1/2/3` collides the moment the loop runs a
   *second* time, so this invocation's round 2 overwrites the previous
   invocation's. Run ids are scoped to the invocation (`<task-id-prefix>-roundN`).

If accuracy is suspiciously flat, check the *training set size per round* before
you believe anything about the data.

### 7. The holdout must be real-only, fixed, and it must have a control

Two bugs, both of which produce a meaningless number that looks fine:

- A holdout that **grows each round** and **contains synthetic frames** is
  measuring the generator against itself.
- Without a **round-0 baseline** trained on real images only, there is nothing
  to compare against and "the synthetic data helped" is untestable.

`train_inspector.py` takes a fixed real-only holdout, and `stage=train
baseline=1` is the control. Run the control before the first round.

### 8. Xet stalls

Hugging Face's Xet transport stalls on some networks; the download simply
never progresses. `run.py` sets `HF_HUB_DISABLE_XET=1` for every stage.

### 9. Phase 1 (few-shot fine-tune) needs more than 24 GB

Measured at batch size 1 and 2: the failing allocation is **identical**, so it
is model-resident memory, not activations. No batch-size knob reaches it.

On a 24 GB card, skip it and use NVIDIA's published adapter
`nvidia/Cosmos-AnomalyGen-PCB-2B` (ungated) — the same 2.9M parameters phase 1
would produce. `anomalygen_generate.py` runs `mode=inference_only` for exactly
this. On a bigger card (L40S / A100 / H100), `anomalygen_finetune.py` runs
phases 0+1 properly.

### 10. Version names collide

`next_version_name()` derives the next name from what already exists. Do not
hardcode `v2` — a re-run then fails on a name that is already taken, mid-loop,
after the GPU work is done.

### 11. "asked N, got fewer" is usually real — check before you call it a bug

Automatic Mask Placement can only put a defect where the CAD says that fault can
occur. Ask for 20 bridges on a board with 12 IC sites and you get 12. The
request is a ceiling, not a promise, and the next round's gap is computed from
what actually landed — so the shortfall must be visible.

Two ways to be misled here, both observed:

- **Counting the wrong thing.** NVIDIA writes each generated image into six
  subdirectories (`annotated_image`, `cropped_image`, `cropped_mask`,
  `original_image`, `original_mask`, `reconstructed_image`), so a directory walk
  reports 165 files for 24 images and a per-class tally by filename prefix
  multiplies accordingly. Count rows in `SDG_result.csv` — one per image, and the
  same file `publish_synthetic` reads.
- **AMP "samples" are not images.** `Total: 13 samples (0 text2roi, 13 cad2roi,
  0 free)` counts (clean board x defect type) work items, each of which yields
  several ROI x seed images. 13 samples produced 24 images. Not a shortfall.

A genuine shortfall shows up as `place_all ... ok < requested` or `fallback > 0`
in the AMP log.

---

## What to tell the user at the end

Report the measurement as measured. On NVIDIA's 86 sample images the honest
finding was: **baseline 0.964, rounds 0.929 flat across 48/60/72 synthetic
frames** — which is one image on a 28-image holdout, and the rare class the
whole lab is about (`bridge`, 8 real examples, ~2 in the holdout) cannot be
resolved in either direction.

That is not a failure of the pipeline and it is not evidence that synthetic data
hurts. It is an instrument without resolution. Say so. If the user has their own
dataset with hundreds of real examples per class, the same code answers the
question properly — and that is the actual recommendation.
