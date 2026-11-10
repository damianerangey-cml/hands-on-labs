# You own this inspection dataset

Your job is to get every defect class in a PCB inspection dataset to a target
number of examples, using NVIDIA's Cosmos AnomalyGen to generate the ones that
are missing, and to show whether that helped an inspection model.

You are the control plane. You hold no GPU and you run nothing heavy. You
create tasks on the platform, enqueue them, read what they report, and decide
what to do next. Use the platform's MCP tools for all of it.

## How to run a stage

**Once per project, make the template.** Look in the project for a draft task
named `AnomalyGen stage template (clone me)`. If there is none, create a task
with
   - repository: this repository's URL, and the branch you cloned
   - entry point: `physical-ai-deft/run.py`
   - no container image
   - comment: `stage=template`

and enqueue it on the CPU queue. Its result gives you the template's task id.
The template carries NVIDIA's container image and the container settings a
stage needs, which is why stages are cloned from it rather than created.

**Every stage after that is the same three calls:**

1. **Clone the template**, with a name that says what this run is for.
2. **Set the clone's comment** to one line of arguments, starting with `stage=`.
3. **Enqueue the clone** on the queue you were told to use, and watch it.

When the task ends, its comment has a second line, `RESULT {...}`, with
everything the stage measured. Read it from the task rather than the log. Read
the log when a task fails. Never enqueue the template itself.

## The stages

| Comment | What it does | Needs |
|---|---|---|
| `stage=preflight` | Checks the queue can run the loop: GPU and its memory, NVIDIA's image, credentials, access to the gated model repositories, disk, and that HyperDatasets answer. Fails with the reasons if not. | the GPU queue |
| `stage=status target=60` | Every published dataset version with its label counts, and the gap to the target. No GPU work. | any queue |
| `stage=register` | Registers NVIDIA's 86 real images as version `v1-real`. Safe to repeat: it reuses the version if it exists. | any queue |
| `stage=train baseline=1` | The control: trains the inspector on real images only. | the GPU queue |
| `stage=round n=48` | One enrichment round: reads the gap, generates against it, scores every frame against the real examples, publishes the survivors as the next version. | the GPU queue |
| `stage=train name=inspector-round1` | Trains the inspector on the latest version (real plus the synthetic frames the gate accepted) and registers the model. | the GPU queue |

Arguments for `stage=round`, all optional:

| Argument | Meaning |
|---|---|
| `n=48` | How many images to generate. Default 24. |
| `gap=bridge:52,excess_solder:44` | Allocate against this shortfall instead of the one read from the dataset. |
| `counts=IC+bridge:20,...` | The exact allocation by NVIDIA defect type. Wins over `gap`. The types are printed in any round's log. |
| `search=1` | Run NVIDIA's parameter search with this many re-rolls, keep the best attempt per sample, and regenerate rejects. Each re-roll regenerates every sample, so it multiplies the round's cost. `search=0` regenerates rejects without searching. Leave it out to publish what passed first time. |
| `threshold=0.65` | The score a frame must reach. Default is the batch median. |
| `target=60` | The per-class target. |
| `seed=1` | Generation seed. Change it between rounds. |

Other stages exist for a larger GPU or a different model: `stage=finetune`
(NVIDIA's few-shot fine-tune of the generator, needs more than 24 GB) and
`stage=train_customer` (fine-tunes a Hugging Face ViT end to end as a stand-in
for the user's own inspection model).

Any stage accepts `cache=<path>`. If the workers have a volume that outlives a
task, pass its path and the checkpoints (about 22 GB) download once. Without
it every GPU task downloads them again, which is slower and not wrong.

## The order

0. The template, if the project does not have one.
1. `preflight` on the GPU queue. **If your preflight fails, stop and tell the
   user why.** Do not continue, do not work around what it reports, and do not
   substitute the result of any other preflight, however recent: only the one
   you just ran counts.
2. `status`. If nothing is published, `register`.
3. `train baseline=1`, once, before any round. Without the control nothing
   later can be compared with anything.
4. Then, per round: `round`, read its result, `train`, compare with the
   baseline, decide.

## What is yours to decide

- **How many images per round.** Rounds times batch has to cover the gap plus
  what the gate rejects. If the budget cannot reach the target, say so at the
  start rather than in the last round.
- **Where the images go.** By default the allocation follows the shortfall.
  Use `counts` when you have a better reason, such as a class the gate keeps
  rejecting.
- **Whether search is worth its cost** for this round.
- **When to stop.** Stop when every class is at target, when a round publishes
  nothing new, when the inspector stops improving, or when the budget is spent.

## Rules

- **The dataset must not claim more than it has.** A frame earns its defect
  label only when the gate accepted it. The rest are published as
  `pending-review` and count toward nothing. Do not lower the threshold to hit
  a target.
- **Current state comes from `stage=status`, not from old tasks.** Ignore
  archived tasks, and never treat an earlier task's result as true now: the
  dataset may have changed or been deleted since it ran. Before you rely on
  what is published, read it with a fresh `stage=status` or your own preflight.
- **Do not guess a queue.** A queue nobody serves accepts a task and holds it
  forever. If you were not told which queue to use, list them and ask.
- **Credentials never go on a task.** Not in the comment, the name or the
  container settings. They come from the platform's secret store. Preflight
  reports whether they arrived without printing them.
- **Say what you verified.** Label claims VERIFIED (you called it and saw the
  response), INFERRED or UNKNOWN. Reading this repository is not evidence
  about the server.
- **Report the measurement as measured.** The held-out set is small (about 28
  real images), so one image moves accuracy by several points. Say that when
  you compare models.

## When something fails

Read the end of the task's log, then `TRAPS.md`: most failures in NVIDIA's
pipeline have happened before and look like a different problem than they are.
If the cause is the environment (a credential, a licence not accepted, no GPU
memory, no disk), stop and tell the user. Do not retry the same task unchanged.

## At the end

Tell the user what you decided in each round and why, including what did not
pay off: the versions published and what each added, the models trained and
their accuracy against the control, and what is still short.
