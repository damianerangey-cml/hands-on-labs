# Kickoff prompt

Copy this into a fresh Claude Code session on the machine that can reach the
target ClearML server. Fill in the four bracketed values first — the session
will ask for them otherwise, and asking costs a round trip.

---

I want to run NVIDIA's Physical AI Data Factory (Cosmos AnomalyGen) as a bounded
enrichment loop on our ClearML server.

Clone https://github.com/damianerangey-cml/hands-on-labs and read
`physical-ai-deft/SKILL.md` **completely** before running anything — it is
mostly failure modes that already happened once, and each one looks like a
different problem than it is.

Our setup:

- ClearML server: `[https://app.example.com]` (credentials are in
  `[~/clearml.conf | the CLEARML_* env vars | wherever]`)
- Whole-GPU queue: `[1XGPU]`, backed by `[an A10G, 24 GB, driver 580]`
- CPU queue: `[1XCPU]`
- `NGC_API_KEY` and `HF_TOKEN` are already projected into task pods from a
  Kubernetes Secret — do **not** pass them as task parameters or docker args
- Shared model cache mounted at `[/cache]` on every task pod
- HyperDatasets: `[enabled]`

Do this, verifying at each step rather than at the end:

1. `python probe_parity.py` — confirm the pod gives NVIDIA's container what its
   `docker run` line implies: whole GPU, `/dev/shm` ≥ 16 G, writable cache,
   both credentials present. **Stop and tell me if any of the four is missing**
   — the rest will fail in ways that look like model bugs.
2. `python launch.py register` — expect 177 frames, 100% annotated, and label
   counts `missing 62 / excess_solder 16 / bridge 8 / clean 5 / mask 86`.
3. **Do one round first**: `python launch.py rounds --rounds 1 --search-rounds 1`.
   It exercises every stage for about a quarter of the cost, and if something is
   wrong with our setup you find out in ~25 minutes rather than two hours.
   Three things to check in the console, in order:
   - `ALLOCATION -- N image(s), by shortfall` — the scarcest class should get
     the most, and a class already at target should be **absent**, not zero.
   - `0 free` in the mask-placement line (`... cad2roi, 0 free`). If you see
     `free`, **stop**: the CAD mask did not reach the container, the defects are
     landing in arbitrary places, and the generated data is worthless. Nothing
     else will tell you this.
   - `asked N got N` per class. Fewer is usually legitimate — AMP can only place
     a defect where the CAD says that fault can occur — but check the AMP log
     for `ok < requested` before you accept it.
4. Then scale up: `python launch.py rounds --rounds 3 --search-rounds 2`.
   **Know what this costs**: each search round regenerates every sample, so this
   is nine batches of generation, not three. `--search-rounds 0` runs the
   quality gate without the parameter search.
5. Report the per-round table: dataset version, synthetic frames published,
   frames that cleared the nn_score gate, and the accuracy of the model trained
   on that version — against the round-0 real-only baseline. Also report the
   `EACH SETTING, WHOLE BATCH` table from the improve stage: that one is not a
   selection effect, so it is the only place the search's own finding is
   trustworthy.

Assume the cluster is already set up (PVCs, secrets, queues). If something on
the cluster is actually missing, tell me what and stop — do not work around it
by putting credentials on the task record.

On the numbers: report them as measured. If the holdout is too small to resolve
the difference between rounds, say that plainly rather than reading a trend into
it. With NVIDIA's own 86 sample images it is — 28 held-out images means accuracy
moves in steps of 3.6%, and `bridge` has about two examples in there.

Two specific ways to be fooled, both of which already fooled me:

- **"after search" medians are best-of-N per sample**, so they rise even when
  every attempt is equally good. Quote the whole-batch table instead.
- **Counting files instead of rows.** NVIDIA writes each generated image into
  six subdirectories, so a directory walk reports 165 files for 24 images.
  `SDG_result.csv` has one row per image and is what publishing reads.

One finding from our run that may or may not hold on your data, worth checking
early because it is cheap: NVIDIA's default `guidance` of 7.0 was **too high**
for this dataset. At 4.0 the whole batch scored better — `excess_solder` went
0.5888 → 0.6158 — and that class had looked unsynthesisable until then.
