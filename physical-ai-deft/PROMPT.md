# Kickoff prompts

Two of them, for two different jobs. The first is the demo. The second is what
you run once, on a server nobody has tried this on.

---

## 1. The agent kickoff — paste this into Claude Code running *in* the lab

Launch the **Claude Code** app from the lab's Applications catalogue, with this
repository as its Git source. Then paste:

> You own the PCB inspection dataset in this ClearML lab. Read
> `physical-ai-deft/AGENT.md` first — it is the operations contract, not a
> script, and it lists the rules you do not get to break.
>
> Goal: get every defect class to **60 examples**, using NVIDIA's Cosmos
> AnomalyGen to generate what was never photographed. Budget: **3 rounds**, and
> stop earlier if a round adds nothing.
>
> Before you generate anything, train the control — a model on real images only
> — or you will have nothing to compare against.
>
> Each round: read the gap, decide what to ask for and why, generate, score
> against the real examples, decide whether the result is worth improving, then
> publish only what the gate accepted and train a model against that version.
>
> Two things I care about more than the numbers. **Do not let the dataset claim
> more than it has** — a frame gets its defect label from the gate, never from
> your expectation. And **tell me what you decided and why**, including the
> calls that did not pay off. The platform records what ran; it cannot record
> your reasoning.
>
> Report at the end: per round, the version, what you asked for, what the gate
> accepted, and the model's accuracy against the baseline. If the holdout is too
> small to resolve the difference, say so rather than reading a trend into it.

That is the whole demo. The agent decides the allocation, the parameters, the
retries and the stopping point; the scaffold stops it getting the mechanics
silently wrong.

**A fair thing to try, live:** ask it something the grid cannot answer —
*"`excess_solder` keeps scoring worst. Work out whether that is fixable, and if
it is not, stop spending GPU on it."*

---

## 2. The bring-up prompt — a server nobody has run this on

Fill in the four bracketed values first; the session will ask otherwise.

> I want to run NVIDIA's Physical AI Data Factory (Cosmos AnomalyGen) on our
> ClearML server.
>
> Clone https://github.com/damianerangey-cml/hands-on-labs and read
> `physical-ai-deft/SKILL.md` **completely** before running anything — it is
> mostly failure modes that already happened, and each one looks like a
> different problem than it is.
>
> Our setup:
> - ClearML server: `[https://app.example.com]`, credentials in `[~/clearml.conf]`
> - Whole-GPU queue: `[1XGPU]`, backed by `[an A10G, 24 GB, driver 580]`
> - CPU queue: `[1XCPU]`
> - `NGC_API_KEY` and `HF_TOKEN` are projected into task pods from a Kubernetes
>   Secret — do **not** pass them as task parameters or docker args
> - Shared model cache at `[/cache]` on every task pod
> - HyperDatasets: `[enabled]`
>
> Do this, verifying at each step rather than at the end:
>
> 1. `python probe_parity.py` — confirm the pod gives NVIDIA's container what
>    its `docker run` line implies: whole GPU, `/dev/shm` ≥ 16 G, writable
>    cache, both credentials. **Stop and tell me if any of the four is missing**
>    — the rest will fail in ways that look like model bugs.
> 2. `python launch.py register` — expect 177 frames, 100% annotated, counts
>    `missing 62 / excess_solder 16 / bridge 8 / clean 5 / mask 86`. If the
>    dataset already exists it will say so and reuse it; that is correct.
> 3. `python launch.py rounds --rounds 1 --search-rounds 1` — the scripted
>    reference implementation, one round, as a smoke test of the whole chain.
>    Check three things in the console: the `ALLOCATION` block puts the most
>    images on the scarcest class and omits any class at target; the
>    mask-placement line says `0 free`; and `asked N got N` per class.
> 4. Only once that passes, hand the loop to an agent — prompt 1 above.
>
> Assume the cluster is set up (PVCs, secrets, queues). If something is actually
> missing, tell me what and stop — do not work around it by putting credentials
> on the task record.
>
> Two ways to be fooled by the output, both of which fooled me:
> **"after search" medians are best-of-N per sample**, so they rise even when
> every attempt is equally good — quote the whole-batch table instead. And
> **counting files instead of CSV rows** reports 165 images for 24, because
> NVIDIA writes each one into six subdirectories.
