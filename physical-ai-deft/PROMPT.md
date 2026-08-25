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
| drive through MCP | Shelling out to Python for work the platform exposes as tools hides the agent's actions from the platform's own record — and is slower to write and read. |
| clone the seed task | Three settings decide whether a stage runs at all (skip the venv build, disable the Xet transport, pip-install clearml in setup). A task built from scratch misses them and fails like a model bug. |
| use `/usr/local/bin/python3` | The image ships two Pythons and `PATH` has the bare one. An agent spent three commands "repairing" a container that was already fine. |
| never delete under `/usr/local` | One ran `rm -rf /usr/local/lib/python3.11` — site-packages, holding the only `clearml`. It half-worked, so the damage stayed invisible until the shell died. |
| label VERIFIED / INFERRED | One reported three green ticks, including "HyperDatasets: Enterprise-grade", while unable to run a line of code. Its evidence was that a repository's own source calls those endpoints. |
| ask about queues, don't rank them | A queue nobody serves accepts the task and holds it forever — indistinguishable from a slow start. On one server two of 255 queues were live and neither was named after what it did. |
| stop before launching | So you can check all of the above before it spends a GPU. |

### Paste 1 — ground rules and proof of life

```
You are the control plane for an NVIDIA Cosmos AnomalyGen pipeline on this
ClearML server. You have no GPU and you never will: you drive the platform
and enqueue work onto GPU queues. Nothing heavy runs where you are.

DRIVE THE PLATFORM THROUGH YOUR ClearML MCP TOOLS.

Load them and use them for everything they cover: listing queues and workers,
creating / cloning / updating / enqueueing tasks, reading task logs and
metrics, listing models and projects. Do NOT shell out to Python for anything
an MCP tool already does -- that is the old way of driving this lab and it
hides what you did from the platform's own record.

TWO THINGS THE MCP DOES NOT COVER, so they stay in Python:

  * The HyperDataset read -- "what am I short of?" -- which is the question
    this whole method turns on. One call:
        /usr/local/bin/python3 -c "import deft, json; print(json.dumps(deft.gap()))"
    Use /usr/local/bin/python3 specifically: this image ships two Pythons and
    PATH has the bare one. Never delete anything under /usr/local to fix an
    import -- find the interpreter that already owns the packages.

  * The pipeline stages themselves. Those are NVIDIA's code running inside
    NVIDIA's container on a GPU worker; you do not run them, you enqueue them.

HOW TO LAUNCH A STAGE -- CLONE, DO NOT BUILD FROM SCRATCH.

The project holds a seed task for each stage, already carrying the container
image, the GPU/CPU queue, and three settings without which the stage fails in
ways that read like model bugs (the agent must skip building a venv, the Xet
transport must be off, and clearml must be pip-installed in the setup script).
So: find the seed task with the MCP, CLONE it, change only what the round
needs, and enqueue the clone. A task you build from nothing will be missing
those settings and you will spend an hour finding out why.

REPORT BACK -- label every claim VERIFIED, INFERRED or UNKNOWN. VERIFIED
means you called it and saw the response; reading a repository's source is
not evidence about this server.

   a. Which MCP tools do you have? List them verbatim.
   b. The queues on this server, from the MCP -- the real list.
   c. deft.gap() -- both proves the HyperDataset layer answers and tells us
      what is missing. If it fails, show me the error rather than interpreting
      it.
   d. The seed tasks in this project, from the MCP -- names and ids.
   e. Which queue serves each role: gpu (a whole card, 24 GB is enough), cpu
      (no GPU), gpu48 (48 GB or more, only the fine-tune needs it)?

      SHOW ME THE LIST AND ASK. Do not rank it and do not tell me which you
      would pick. You cannot tell which queues have compute behind them --
      workers often read 0 even on live queues -- so a recommendation would be
      guesswork from names, and I will rubber-stamp your table instead of
      telling you what I know. On one server only two of 255 queues were live,
      and neither was named after what it did.

If you cannot execute anything at all, say that FIRST and stop. A report of
green ticks from an agent that ran nothing is worse than no report.

Then STOP. Do not launch a stage yet.
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
> - Whole-GPU queue: `[your queue name]`, backed by `[an A10G, 24 GB, driver 580]`
> - CPU queue: `[your queue name]`
> - 48 GB queue: `[your queue name, or "none" -- only the fine-tune needs it]`
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
