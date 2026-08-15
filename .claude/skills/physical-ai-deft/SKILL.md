---
name: physical-ai-deft
description: Drive NVIDIA Cosmos AnomalyGen on a ClearML server to synthesise the rare manufacturing defects a dataset is short of. Use when asked to generate synthetic defects, close a rare-class gap, run AnomalyGen or the Physical AI Data Factory, enrich an inspection dataset, or work with the PCB HyperDataset. Covers reading the gap from metadata, allocating generation by shortfall, gating on nn_score, and publishing versions with lineage.
---

# Cosmos AnomalyGen, driven from a control plane

You close the gap between the defect examples an inspection dataset **has** and
the ones it **needs**, using NVIDIA's Cosmos AnomalyGen — and you are honest
about how far you got.

## Read these before acting

The repository is `physical-ai-deft/`. In a lab session it is usually already
cloned at `~/environment/task_repository/hands-on-labs.git` — the app's Git
field does that before you start; work from it in place. Only fetch it yourself
if it is genuinely absent, and note the session container may ship **no git
binary**: that is not a problem to fix — curl the GitHub tarball
(`archive/refs/heads/main.tar.gz`) instead of installing anything.

- **`physical-ai-deft/AGENT.md`** — the operations contract. Read it in full.
- **`physical-ai-deft/SKILL.md`** — the traps, eleven of them, each one a real
  failure that cost a GPU hour.

## The five things that matter most

**1. You have no GPU and never will.** This session is a control plane: read
state over the API, enqueue work onto queues. `deft.gap()`, `deft.history()`,
`deft.queues()` run locally. Everything else is `launch.py <stage>`.

**2. Find the interpreter that owns the packages.** Containers ship more than
one Python and `PATH` is often on the wrong one. On the ClearML app image it is
`/usr/local/bin/python3`, not `python3`. **Never repair or delete a system
Python** — an agent that ran `rm -rf /usr/local/lib/python3.11` deleted the only
copy of `clearml` and destroyed its own session.

```bash
for p in /usr/local/bin/python3 /usr/bin/python3; do
  echo "$p -> $($p -c 'import clearml; print(clearml.__version__)' 2>&1 | tail -1)"
done
```

**3. Do not guess queue names.** `deft.pick_queue("gpu")` resolves or refuses.
A queue nobody serves accepts your task and holds it forever — identical, from
outside, to a slow start. Ask; record the answer in `DEFT_QUEUE_GPU` etc.

**4. Say what you verified.** VERIFIED = you called it and saw the response.
Reading this repository's source is not evidence about the server. If you cannot
execute code, that is the headline, not a footnote.

**5. A frame earns its label from the gate.** Never publish a generated frame
under its defect class because you expect it to be good. What the gate rejects
becomes `pending-review` and counts toward nothing. You read these counts next
round; inflating them is deceiving yourself.

## The loop

```bash
python launch.py register                     # once — the real data
python launch.py train --baseline             # the control, BEFORE generating
# then per round, deciding from what the last step reported:
python launch.py generate --gap bridge:52,excess_solder:44 --run-id r1
python launch.py evaluate --run-dir /cache/results/pcb-uc1/runs/r1
python launch.py improve  --run-dir /cache/results/pcb-uc1/runs/r1 --search-rounds 1
python launch.py publish  --run-dir /cache/results/pcb-uc1/runs/r1/searched --run-id r1
python launch.py train    --name inspector-round1
```

`run_id` must be unique per invocation or publish silently skips the round.

## What you decide, and what you do not

Arithmetic — how many of each class — falls out of `deft.gap()`. The judgement
is elsewhere: a class scoring badly (is it the generator, the defect spec, or a
class that will not synthesise on this board?), whether a parameter search is
worth a full regeneration, and when to stop. Stop at target, out of budget, or
when a round adds nothing.

## Report honestly

Per round: the version, what you asked for, what the gate accepted, the model's
accuracy against the baseline — and what you decided and why, including the
calls that did not pay off. If the holdout is too small to resolve a difference,
say so rather than reading a trend into it. On NVIDIA's 86 sample images it is:
28 held-out images means accuracy moves in steps of 3.6%.
