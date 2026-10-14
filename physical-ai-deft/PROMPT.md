# Kickoff prompts

Two of them, for two different jobs. The first starts the agent-driven loop on
any Klique environment. The second is a scripted smoke test of the whole chain,
for a server nobody has tried this on.

---

## 1. The agent kickoff — any Klique environment

Launch the platform's **Claude Code** app from the app catalogue on your own
Klique (ClearML Enterprise) environment:

| Field | Setting | Why |
|---|---|---|
| Compute Resource | a CPU compute resource | The agent holds no GPU. Everything heavy is enqueued. |
| Session Project | the project the tasks should land in | The session sits beside every task it launches. |
| Model | a gateway profile, or your own Anthropic key | Your own key goes in your Configuration Vault as `ANTHROPIC_API_KEY`, with *I already have an API key in my Vault* ticked. |
| Skip permission prompts | on | Lets the loop run unattended inside its sandbox. Paste 1 still ends with a stop. |
| Git Access | empty | This repository is public. The agent clones it itself, in paste 1. |

The app wires its MCP server into the session. Nothing about this repository
assumes a particular server, queue name, GPU size or image: the agent finds
out, and asks you what it cannot find out.

Then **two pastes, in this order.** Do not merge them. You get one chance to
bound the agent's behaviour before it starts improvising, and every line of
paste 1 is there because of something that actually went wrong:

| Line | The failure it prevents |
|---|---|
| drive through MCP | Shelling out to Python for work the platform exposes as tools hides the agent's actions from the platform's own record, and is slower to write and read. |
| report a broken MCP, do not fix it | The tools come with the app. An agent that rewires them, or quietly falls back to scripts, is running a different system from the one you approved. |
| find the interpreter, never repair one | An image can ship two Pythons with the bare one on `PATH`. One agent ran `rm -rf /usr/local/lib/python3.11`, deleting the only `clearml`; the damage stayed invisible until the shell died. |
| a draft per run, enqueued through MCP | Three settings decide whether a stage runs at all (skip the venv build, disable the Xet transport, pip-install clearml in setup), and each run needs its own run id and gap. `launch.py --dry-run` writes all of that into a draft; the MCP cannot edit a cloned task's container settings, so a cloned seed reruns the same run id and publish silently adds nothing. |
| label VERIFIED / INFERRED | One agent reported three green ticks, including "HyperDatasets: Enterprise-grade", while unable to run a line of code. Its evidence was that a repository's own source calls those endpoints. |
| ask about queues, do not rank them | A queue nobody serves accepts the task and holds it forever, indistinguishable from a slow start. On one server two of 255 queues were live and neither was named after what it did. |
| stop before launching | So you can check all of the above before it spends a GPU. |

### Paste 1 — ground rules and proof of life

```
First, clone https://github.com/damianerangey-cml/hands-on-labs.git into
~/hands-on-labs and read physical-ai-deft/AGENT.md completely before running
anything. Work from ~/hands-on-labs/physical-ai-deft from then on.

You are the control plane for an NVIDIA Cosmos AnomalyGen pipeline on this
Klique (ClearML) server. You have no GPU and you never will: you drive the
platform and enqueue work onto GPU queues. Nothing heavy runs where you are.

DRIVE THE PLATFORM THROUGH YOUR MCP TOOLS. This session's Claude Code app
configured them. Use them for everything they cover: listing queues and
workers, creating / cloning / updating / enqueueing tasks, reading task logs
and metrics, listing models and projects. If the MCP server fails to connect,
report its name, its URL and the error, and STOP. Do not repoint it and do not
fall back to Python for what it would have done.

PYTHON, FOR THE TWO THINGS THE MCP DOES NOT COVER: the HyperDataset read
(deft.gap()) and creating run drafts (launch.py --dry-run). Find the
interpreter that already imports clearml, as AGENT.md shows. If none does,
install requirements.txt for your user only. Never delete or repair anything
under a system path.

LAUNCHING A STAGE: create a draft for that run with
`launch.py <stage> --dry-run` and the run's own arguments (--run-id, --gap,
--num-sdg, --baseline), then enqueue the draft with the MCP and watch it
there. Never build a stage task by hand, and do not clone a draft to reuse it:
the MCP cannot change a clone's run settings.

REPORT BACK -- label every claim VERIFIED, INFERRED or UNKNOWN. VERIFIED
means you called it and saw the response; reading a repository's source is
not evidence about this server.

   a. Which MCP tools do you have? List them.
   b. The queues on this server, from the MCP -- the real list.
   c. deft.gap() -- proves the HyperDataset layer answers and shows what is
      missing. If it fails, show me the error rather than interpreting it.
   d. The tasks already in this project, if any, from the MCP.
   e. Which queue serves each role: gpu (a whole GPU, 24 GB is enough),
      cpu (no GPU), gpu_large (more than 24 GB, only the fine-tune needs it;
      it may be the same queue as gpu, or "none").
   f. Whether task pods share a storage path for the stages, and where.

      For e and f, SHOW ME WHAT YOU FOUND AND ASK. Do not rank queues and do
      not tell me which you would pick: you cannot see which have compute
      behind them, so a recommendation would be guesswork from names.

If you cannot execute anything at all, say that FIRST and stop. A report of
green ticks from an agent that ran nothing is worse than no report.

Then STOP. Do not launch a stage yet. Once I answer, record my answers with
deft.set_queues(...), run probe_parity.py once, report what it found, and
stop again.
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
> - Large-GPU queue (more than 24 GB): `[your queue name, the same as the GPU queue, or "none" -- only the fine-tune needs it]`
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
