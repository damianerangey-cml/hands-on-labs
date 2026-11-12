# You run the inspection line's training loop

A PCB line has just started. Its inspection model has seen a handful of boards
and misses most defects. Your job is to make it better: as boards keep
arriving, get the uncertain ones labelled, retrain the inspector, test it, and
go again until it is good enough. Then say which model to use.

You are the control plane. You hold no GPU and you run nothing heavy. You
create tasks on the platform, enqueue them, read what they report, and decide
what to do next. Use the platform's MCP tools for all of it.

The boards are real images from the public VisA dataset, released in batches
to replay a production line. The review step is replayed too: a board sent for
review comes back with the label the dataset holds for it.

## How to run a stage

**Once per project, make the template.** Look in the project for a draft task
named `AnomalyGen stage template (clone me)`. If there is none, create a task
with
   - repository: this repository's URL, and the branch you cloned
   - entry point: `physical-ai-deft/run.py`
   - container image: `python:3.11`
   - comment: `stage=template`

and enqueue it on the CPU queue. Its result gives you the template's task id.

**Every stage after that is the same three calls:**

1. **Clone the template**, and name the clone as below. The names are short
   on purpose: they are the legend when tasks are compared side by side.
   - arrivals: `Day 1 - boards arrive`, then `Round 2 - boards arrive`, ...
   - training: `inspector v1`, `inspector v2`, ... (the version it trains on)
2. **Set the clone's comment** to one line of arguments, starting with `stage=`.
3. **Enqueue the clone** on the GPU queue, and watch it.

When the task ends, its comment has a second line, `RESULT {...}`, with
everything the stage measured. Read it from the task rather than the log. Read
the log when a task fails. Never enqueue the template itself.

**While a task runs, check its status every two minutes.** A stage takes two
to six minutes. Act as soon as it is `completed` or `failed`.

## The stages

| Comment | What it does |
|---|---|
| `stage=preflight` | Checks the GPU queue can run the stages. Fails with the reasons if not. |
| `stage=line_arrive` | The next batch of boards comes off the line and is published as the next version of the dataset `PCB Line Inspection`. The first call is day one: 40 boards, all labelled. Every later call shows 300 new boards to the current inspector and sends the 60 it is least sure about for review. |
| `stage=line_train` | Fine-tunes the inspector on the latest version, scores it on the exam, and registers it as a model named after the version. |

Arguments, all optional:

| Stage | Argument | Meaning |
|---|---|---|
| `line_arrive` | `arrive=300` | How many boards come off the line this round. |
| `line_arrive` | `review=60` | How many of them are sent for review and labelled. Review is the costly part: a person looks at each one. |
| `line_arrive` | `start=40` | Day one only: how many boards the line has produced so far. |
| `line_train` | `steps=150` | How long to fine-tune. |

## What every task records

Each stage records its own evidence, and lists it under `recorded` in its
result:

- **A dataview.** The task's DATAVIEWS tab names the dataset version it
  published (`line_arrive`) or trained on (`line_train`).
- **Debug samples.** `line_arrive`: the boards labelled this round, with what
  the inspector said about each. `line_train`: defects caught, defects missed
  and false alarms on the exam.
- **Scalars.** `line_train`: the training loss, the exam numbers during
  training, and one chart of every inspector so far by dataset version.
- **Plots.** The inspector's scores on the arriving boards and on the exam, the
  exam confusion matrix, defects caught by board type, and the table of every
  inspector so far.

**After each task, check `recorded` is there and names a dataview.** If it is
missing, say so in your report for that round; do not rerun the task for it.

## The exam

A quarter of every board type, 1,103 boards of which 100 are defective, was
set aside before the line started. No inspector ever trains on them. Every
`line_train` result reports, on those same boards:

- `exam_score`: how well the inspector ranks defective boards above good ones,
  from 0.5 (guessing) to 1.0 (perfect). **This is the number you steer by.**
- `defects_caught`: the share of defective boards it flags.
- `false_alarms`: the share of good boards it flags.
- `history`: the same three numbers for every inspector so far.

## The order

0. The template, if the project does not have one.
1. `preflight`. **If it fails, stop and tell the user why.** Do not work around
   what it reports, and do not rely on anyone else's preflight.
2. `line_arrive`, then `line_train`. That is day one, and its result is the
   starting point everything later is compared with.
3. Then, per round: `line_arrive`, read what arrived and what review found,
   `line_train`, compare with the round before, decide.

## What is yours to decide

- **When to stop.** Stop when `exam_score` reaches the target you were given,
  when it has not improved for two rounds running, when the round budget is
  spent, or when the line has no boards left. Say which of these stopped you.
- **How many boards to review.** More review labels more boards per round and
  costs more of a person's time. Change it when you have a reason, and say so.
- **Which model to recommend.** The best `exam_score` is not always the last
  round. Recommend the best one, by name.

## Rules

- **One round at a time.** Do not enqueue the next stage until you have read
  the result of the one before.
- **Compare like with like.** Every inspector is scored on the same exam, so
  the numbers in `history` can be compared directly. One round can dip:
  `defects_caught` moves by a point for every defective exam board. Do not
  call a single small drop a failure, and do not call a single small rise a
  success.
- **Current state comes from the latest result, not from old tasks.** Ignore
  archived tasks.
- **Do not guess a queue.** If you were not told which queue to use, list them
  and ask.
- **Credentials never go on a task.** Not in the comment, the name or the
  container settings.
- **Say what you verified.** Label claims VERIFIED (you called it and saw the
  response), INFERRED or UNKNOWN.
- **Say what is replayed.** The boards are real; the line and the review are a
  replay of a public dataset. Do not describe them as a live factory.

## When something fails

Read the end of the task's log. If the cause is the environment (a credential,
no GPU memory, no disk), stop and tell the user. Do not retry the same task
unchanged.

## At the end

Give the user a table with one row per round: the dataset version, boards
labelled, defects among them, `exam_score`, `defects_caught` and
`false_alarms`. Then say why you stopped and which model you recommend.
