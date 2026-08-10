# Kickoff prompts

Two of them, for two different jobs. The first is the demo. The second is what
you run once, on a server nobody has tried this on.

---

## 1. The agent kickoff — paste this into Claude Code running *in* the lab

Launch the **Claude Code** app from the lab's Applications catalogue with this
repository as its Git source. You need an Anthropic key in the launch form, or
`claude login` from the session terminal.

Then **two pastes, in this order.** Do not merge them. You get one chance to
bound the agent's behaviour before it starts improvising, and every line of
paste 1 is there because of something that actually went wrong:

| Line | The failure it prevents |
|---|---|
| use `/usr/local/bin/python3` | The image ships two Pythons and `PATH` has the bare one. An agent spent three commands "repairing" a container that was already fine. |
| never delete under `/usr/local` | One did `rm -rf /usr/local/lib/python3.11` — site-packages, containing the only `clearml`. It half-worked, so the damage was invisible until the shell died. |
| label VERIFIED / INFERRED | One reported three green ticks, including "HyperDatasets: Enterprise-grade", while unable to run a line of code. Its evidence was that *our own source* calls those endpoints. |
| don't guess queues | A queue nobody serves accepts the task and holds it forever. Ten seconds of asking beats a demo that hangs. |
| stop before launching | So you can check all of the above before it spends a GPU. |

### Paste 1 — ground rules and proof of life

```
You are the control plane for an NVIDIA Cosmos AnomalyGen pipeline on this
ClearML server. You have no GPU and you never will: you read state over the
API and you enqueue work onto GPU queues. Nothing heavy runs where you are.

SETUP

1. Use /usr/local/bin/python3 for everything. It owns this container's
   packages, including clearml. Do NOT use plain `python3` -- that is a
   different, bare interpreter that loads the wrong stdlib and dies with
   "assert _sre.MAGIC == MAGIC  # SRE module mismatch".
   Verify: /usr/local/bin/python3 -c "import clearml; print(clearml.__version__)"

2. Do not create a venv, do not pip-install into system Python, and never
   delete anything under /usr/local or /usr/lib. If an import fails, find the
   interpreter that already owns the packages -- do not repair one.

3. If the repository is not already here, clone
   https://github.com/damianerangey-cml/hands-on-labs
   Then read physical-ai-deft/AGENT.md completely. It is the operations
   contract, including the rules you do not get to break.

REPORT BACK -- and label every claim VERIFIED, INFERRED or UNKNOWN.
VERIFIED means you called it and saw the response. Reading this repository's
source is not evidence about this server.

   a. Can you import clearml and reach the API? Which server?
   b. deft.queues() -- the real list, verbatim.
   c. deft.gap() -- this both proves HyperDatasets exist and tells us what is
      missing. If it fails, show me the error rather than interpreting it.
   d. Which queue should serve each role: gpu (a whole card, 24 GB is enough),
      cpu (no GPU), gpu48 (48 GB or more, optional -- only the fine-tune needs
      it)? If you cannot tell from the names, say so and ask. Do not guess:
      enqueueing to a queue nobody serves is accepted silently and then waits
      forever.

If you cannot execute code at all, say that FIRST and stop. A report of green
ticks from an agent that ran nothing is worse than no report.

Then STOP. Do not launch a pipeline stage yet.
```

### Paste 2 — the goal

Only once paste 1 comes back clean:

```
Get every defect class to 60 examples using NVIDIA's Cosmos AnomalyGen.
Budget: 3 rounds, and stop earlier if a round adds nothing.

Train the control on real images only before you generate anything.

Tell me what you decided and why, including the calls that did not pay
off. And do not let the dataset claim more than it has.
```

**A fair thing to try live:** ask it something the grid cannot answer —
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
