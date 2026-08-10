"""The surface an agent drives. Verbs, not a loop.

    import deft
    deft.gap()                      # what am I short of?
    deft.history()                  # what did I already do?
    deft.generate(gap=..., n=24)    # ask for images
    deft.score(run_dir)             # judge them
    deft.improve(run_dir, ...)      # search, keep best, regenerate
    deft.publish(run_dir, run_id)   # commit the survivors
    deft.train(round_name)          # train + register against that version

WHY THIS EXISTS RATHER THAN A run_rounds()
-------------------------------------------
`run_rounds.py` works, and it owns every decision worth making: how many
images, of which class, which generation parameters, whether a bad frame is
worth retrying, when to stop. A `for` loop makes all of those. That is a
hardcoded pipeline with an agent-shaped hole in the documentation.

This module is the other half of that trade. It exposes the operations and
keeps the decisions out. An agent composes them, and because it is a coding
agent it can also ignore this file and call the stage modules directly -- which
is the point. Scaffolding you can step outside of is scaffolding; a driver you
must go through is a script.

WHAT THE SCAFFOLD IS FOR
------------------------
Not convenience. Every helper here exists because getting it wrong is silent:

  * run ids that must be unique per invocation, or a round republishes the
    previous round's frames and the counts inflate;
  * a label that must be earned from the gate rather than assumed, or the loop
    starts believing its own output;
  * an upload destination that has to carry the run, or the duplicate guard has
    nothing to match on.

An agent should not have to know any of that, and should not be able to get it
wrong by accident. Everything the agent SHOULD decide is a parameter with no
default worth trusting.

YOUR STATE IS THE DATASET, NOT A FILE
--------------------------------------
`gap()` and `history()` read the server. There is no local state file to lose,
so an agent that crashes mid-round can restart, ask what is true now, and carry
on. NVIDIA's DEFT keeps its whole memory in three files in one directory on one
machine; this keeps it in the platform, which is the difference between a loop
that survives its own driver and one that does not.
"""
import os

# These three ARE defaults, and unlike queue names they should be: they name
# this lab's own content, not your cluster. "PCB Inspection" is what the lab
# calls the thing it builds -- portable, because it does not have to match
# anything that already exists on your server. A queue name has to.
DATASET = os.environ.get("DEFT_HYPERDATASET", "PCB Inspection")
UC = os.environ.get("DEFT_DATASET", "pcb-uc1")
PROJECT = os.environ.get("DEFT_PROJECT", "Physical AI Inspection")

# Housekeeping labels are not defect classes -- they must never look like a gap.
HOUSEKEEPING = {"clean", "mask", "pending-review"}


def _ds():
    import hyperdataset as hd
    return hd, hd.get_or_create_dataset(DATASET)


def _ds_ro():
    """(module, dataset_id_or_None) -- for the READ verbs. Creates nothing."""
    import hyperdataset as hd
    return hd, hd.find_dataset(DATASET)


# ---- queues: DISCOVER, do not assume -------------------------------------
# THERE ARE NO DEFAULT QUEUE NAMES HERE, AND THAT IS DELIBERATE.
#
# An earlier version carried one deployment's own queue names as fallbacks.
# On a server that spelled one of them differently it refused to resolve;
# the fix looked like case-insensitive matching, and it is in fact the whole
# idea of a default that was wrong. A public repo cannot know what anyone calls
# their queues, and a name that ALMOST matches somebody else's cluster is worse
# than no name at all -- it turns "I need to ask" into "I found it", and the
# thing it found may be eight fractional slices of one card.
#
# So the answer comes from the operator, once, and is then RECORDED:
#
#     env DEFT_QUEUE_{GPU,CPU,GPU48}   ->  set_queues() config file  ->  ask
#
# The config file matters more than it looks. An agent's `export FOO=bar` dies
# with the shell that ran it, so an agent told to "set an env var" re-asks on
# every command and eventually starts guessing again. Writing the answer down
# is what makes asking a one-time cost.
_QUEUE_ENV = {
    "gpu":   "DEFT_QUEUE_GPU",
    "cpu":   "DEFT_QUEUE_CPU",
    "gpu48": "DEFT_QUEUE_GPU48",
}
_QUEUE_WANT = {
    "gpu":   "a whole GPU (24 GB is enough) -- generation, scoring, training",
    "cpu":   "no GPU -- coordination work only",
    "gpu48": "a GPU with at least 48 GB, for the few-shot fine-tune (optional)",
}
CONF = os.path.expanduser(os.environ.get("DEFT_CONF", "~/.deft/config.json"))


class NoSuchQueue(LookupError):
    """This server has no queue for that role, and somebody has confirmed it.

    Distinct from the plain LookupError meaning "nobody has told me yet". The
    caller can catch this one and skip the stage; catching the other and
    skipping would silently drop work because a question went unanswered.
    """


def config():
    """Whatever has been recorded about this server. {} if nothing has."""
    import json
    try:
        with open(CONF) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def set_queues(gpu=None, cpu=None, gpu48=None):
    """Record which queue serves which role on THIS server. Ask once, keep it.

    Every name is checked against the server before it is written -- recording
    a typo would reproduce the exact failure this whole mechanism exists to
    prevent, except now it would be durable.

    Three ways to answer a role, and the middle one is the point:

        gpu48="1xGPU-48GB"   there is one, and this is it
        gpu48=False          THERE IS NONE HERE -- recorded, so nobody re-asks
        gpu48=None           no answer yet (leaves any previous answer alone)

    `False` matters because "we don't have that" is a real answer and used to be
    unrecordable: the agent would be told there was no 48 GB queue, store
    nothing, and ask again on the next call. An answer you cannot write down is
    a question you ask forever, and an agent that keeps asking eventually
    guesses. Only `gpu48` is genuinely optional -- without it the fine-tune is
    skipped and the other six phases run on a 24 GB card, which is what this
    lab has been demonstrating.
    """
    import difflib
    import json
    have = {q["name"] for q in queues()}
    want = {"gpu": gpu, "cpu": cpu, "gpu48": gpu48}
    bad = {k: v for k, v in want.items() if v and v not in have}
    if bad:
        # A near-miss beats an inventory. Measured on a server with 255 queues:
        # dumping all of them is technically the answer to "what exists" and
        # useless as an answer to "what did I get wrong", which is the question
        # actually being asked. The close matches are usually the typo.
        lines = []
        for role, name in sorted(bad.items()):
            near = difflib.get_close_matches(name, have, n=4, cutoff=0.5)
            lines.append("  %s=%r -- %s" % (
                role, name,
                ("did you mean: " + ", ".join(near)) if near
                else "no close match"))
        raise LookupError(
            "No such queue on this server:\n%s\n\n"
            "%d queues here; deft.queues() lists them all."
            % ("\n".join(lines), len(have)))
    conf = config()
    # `False` -> stored as JSON null, meaning "asked and answered: none here".
    # `None`  -> not mentioned in this call, so leave whatever was there.
    conf.setdefault("queues", {}).update(
        {k: (None if v is False else v)
         for k, v in want.items() if v is not None})
    os.makedirs(os.path.dirname(CONF) or ".", exist_ok=True)
    with open(CONF, "w") as fh:
        json.dump(conf, fh, indent=2, sort_keys=True)
    return conf["queues"]


def _recent_task_counts(limit=1000):
    """{queue_id: how many of the last `limit` tasks ran there}.

    The only evidence available that a queue is CONNECTED TO ANYTHING. A name
    is chosen by whoever created the queue and need not describe what serves it,
    or whether anything does; `workers` says nothing on enterprise (see below);
    and the autoscaler or pool mapping that actually decides is not exposed over
    the API. But a queue that has run work has, demonstrably, run work.

    Best-effort: returns {} rather than raising, because a missing shortlist
    should degrade the question, not break it.
    """
    from clearml.backend_api.session.client import APIClient
    try:
        tasks = APIClient().tasks.get_all(
            only_fields=["execution.queue"], page=0, page_size=limit,
            order_by=["-last_update"])
    except Exception:
        return {}
    counts = {}
    for t in tasks:
        ex = getattr(t, "execution", None)
        qid = (ex or {}).get("queue") if isinstance(ex, dict) \
            else getattr(ex, "queue", None)
        if qid:
            counts[qid] = counts.get(qid, 0) + 1
    return counts


def queues(activity=True):
    """Every queue on this server, with whatever tells them apart.

    Returns {name, id, workers, queued, recent_tasks}.

    DO NOT READ `workers == 0` AS "NOTHING SERVES THIS". It is a useful signal
    on a stock ClearML deployment and a LIE on some enterprise ones: where
    queues are fronted by a resource pool, the agent listens on an internal
    `resource_link_<uuid>` queue and the friendly name shows zero workers while
    being perfectly well served. Measured on the deployment this was written
    against -- every queue reported 0 workers while tasks ran on all of them.

    `recent_tasks` is the signal that survives that. Measured on a server with
    255 queues: two of them had ever run anything, and those two were the only
    two with autoscalers behind them. It is evidence, not proof -- a brand new
    queue has run nothing and may still be perfectly alive -- so it belongs in
    the QUESTION you put to a human, not in a decision you make without one.
    """
    from clearml.backend_api.session.client import APIClient
    c = APIClient()
    seen = _recent_task_counts() if activity else {}
    out = []
    for q in c.queues.get_all():
        workers = list(getattr(q, "workers", None) or [])
        entries = list(getattr(q, "entries", None) or [])
        out.append({
            "name": q.name,
            "id": q.id,
            "workers": len(workers),
            "queued": len(entries),
            "worker_names": [getattr(w, "key", None) or getattr(w, "name", "?")
                             for w in workers][:4],
            "recent_tasks": seen.get(q.id, 0),
        })
    return sorted(out, key=lambda d: d["name"])


def pick_queue(kind="gpu"):
    """Resolve the queue for `kind` ("gpu" | "cpu" | "gpu48"), or refuse.

    Order: the DEFT_QUEUE_* env var, then whatever `set_queues()` recorded.
    There is no third source -- no name is built in, and nothing is inferred
    from a queue being called something GPU-ish. A queue named `gpu-shared`
    might be eight fractional slices of one card, which is exactly the wrong
    place to send a fine-tune, and the API does not expose enough to tell them
    apart. The operator knows their cluster; ten seconds of asking beats a task
    that sits in `queued` forever looking like a slow start.
    """
    env = os.environ.get(_QUEUE_ENV[kind])
    if env:
        return env
    known = config().get("queues", {})
    if known.get(kind):
        return known[kind]
    if kind in known:
        # Recorded as null: somebody was asked and said this server has none.
        # A different exception on purpose -- "no such queue here, skip the
        # stage" is an answer, and must not read like "nobody has told me yet".
        raise NoSuchQueue(
            "This server has no queue for %s (%s) -- recorded by set_queues.\n"
            "%s"
            % (kind, _QUEUE_WANT[kind],
               "Skip the fine-tune; the other phases run on the gpu queue."
               if kind == "gpu48" else
               "Nothing can run until somebody records one."))
    # This message is the QUESTION an agent will put to a person, so it is
    # written as one. It leads with what is actually known (which queues have
    # run work), states plainly what cannot be known from here, and offers no
    # ranking -- because a recommendation drawn from evidence you have just
    # called insufficient is not a recommendation, it is a rubber stamp waiting
    # to be signed. Observed: an agent refused all three roles for want of
    # evidence and then, in the same reply, ranked them by name. The human
    # approved the ranking instead of supplying the answer they actually had.
    all_q = queues()
    live = sorted((q for q in all_q if q["recent_tasks"]),
                  key=lambda q: -q["recent_tasks"])
    idle = sorted(q["name"] for q in all_q if not q["recent_tasks"])

    if live:
        seen = "\n".join("  %-32s %d recent task%s"
                         % (q["name"], q["recent_tasks"],
                            "" if q["recent_tasks"] == 1 else "s")
                         for q in live[:15])
        evidence = (
            "Queues that have actually run something recently:\n%s\n\n"
            "%d other queue%s exist and have run nothing in that window. That is\n"
            "NOT proof they are dead -- a new queue has no history either -- and a\n"
            "queue above is not proof it suits this role.\n"
            % (seen, len(idle), "" if len(idle) == 1 else "s"))
    else:
        evidence = (
            "No queue here has run anything recently, so I have no evidence at\n"
            "all about which are live. %d queues exist; deft.queues() lists them.\n"
            % len(all_q))

    raise LookupError(
        "I do not know which queue to use for %s (%s).\n\n"
        "%s\n"
        "I cannot tell which queues have compute behind them. `workers` reads 0\n"
        "for all of them on this kind of deployment, and the autoscaler or pool\n"
        "mapping that decides is not exposed to me. A queue's NAME is not\n"
        "evidence either -- whoever created it chose it, and it need not say\n"
        "anything about what serves it, or whether anything does.\n\n"
        "So: ask which queue serves this role. Show this list, do not rank it,\n"
        "and do not offer a favourite -- you would be guessing from the same\n"
        "names you have just been told mean nothing.\n\n"
        "Record the reply:\n"
        "  deft.set_queues(%s=\"<name>\")\n"
        "  deft.set_queues(%s=False)      # if there is none here -- also an answer\n"
        % (kind, _QUEUE_WANT[kind], evidence, kind, kind)
    )


def gap(target=60):
    """What each defect class is short of, and what is there now.

    Returns {"version", "version_id", "counts", "gap", "at_target"}.

    This is the read the whole method turns on: it is server-side label counts,
    so it costs about a second, downloads nothing, and can therefore be asked
    on every pass without thinking about it. `gap` excludes classes already at
    target -- generating more of what you have enough of is the waste the
    method exists to avoid.
    """
    hd, ds_id = _ds_ro()
    v = hd.latest_published(ds_id) if ds_id else None
    if not v:
        # Two different nothings, and an agent must not conflate them: no
        # dataset at all means `register` has never run here, whereas a dataset
        # with no published version means it ran and nothing was committed.
        return {"version": None, "version_id": None, "counts": {}, "gap": {},
                "at_target": [],
                "dataset": ds_id,
                "note": ("no HyperDataset %r on this server -- run register first"
                         % DATASET) if not ds_id else
                        "dataset exists but has no published version yet"}
    counts = hd.stats(v["id"])["labels"]
    short = {k: target - n for k, n in counts.items()
             if k not in HOUSEKEEPING and n < target}
    at = [k for k, n in counts.items() if k not in HOUSEKEEPING and n >= target]
    return {"version": v.get("name"), "version_id": v["id"], "counts": counts,
            "gap": short, "at_target": at}


def history():
    """Every published version and registered model so far, oldest first.

    For a restarted agent: this is how you find out what a previous you already
    did, without a local file and without trusting your own notes.
    """
    hd, ds_id = _ds_ro()
    from clearml import Model
    versions = hd._call("get_versions", {"dataset": ds_id, "only_published": True,
                                         "page": 0, "page_size": 200}
                        ).get("versions") if ds_id else None
    versions = versions or []
    versions.sort(key=lambda v: v.get("created") or "")
    out_v = [{"name": v.get("name"), "id": v["id"], "created": v.get("created")}
             for v in versions]
    try:
        models = [{"name": m.name, "id": m.id,
                   "version": (m.get_metadata() or {}).get("version_id")}
                  for m in Model.query_models(project_name=PROJECT)]
    except Exception as e:
        models = [{"error": str(e)}]
    return {"versions": out_v, "models": models}


# ---- the stage verbs -------------------------------------------------------
# Thin on purpose. Each one is the stage module's own function; this module
# supplies only the wiring that is easy to get silently wrong.

def generate(gap=None, n=24, run_id=None, per_defect_counts=None, seed=0):
    """Place masks from the board's CAD and generate `n` images.

    Pass `gap` to have the budget split by shortfall, or `per_defect_counts` to
    decide the split yourself -- an agent with a reason to want six bridges and
    nothing else should say so directly rather than reverse-engineering a gap
    that produces it.

    `run_id` MUST be unique per invocation. Reuse one and the next publish
    treats the round as already published and silently adds nothing.
    """
    import anomalygen_generate as g
    if not run_id:
        raise ValueError(
            "run_id is required and must be unique per invocation -- reusing "
            "one makes publish() skip the round silently. Scope it to your "
            "session, e.g. '<task-id-prefix>-round2'.")
    return g.anomalygen_generate(dataset_name=UC, num_sdg=n, gap=gap,
                                 per_defect_counts=per_defect_counts,
                                 run_id=run_id, seed=seed)


def score(run_dir, threshold=None):
    """NVIDIA's nn_score for every generated frame, against the REAL examples.

    Leave `threshold` unset and it defaults to the batch median, which is a
    defensible starting bar rather than a number somebody guessed -- NVIDIA's
    own docs say there is no fixed threshold for good.
    """
    import anomalygen_evaluate as e
    return e.anomalygen_evaluate(dataset_name=UC, run_dir=run_dir,
                                 nn_threshold=threshold)


def improve(run_dir, search_rounds=1, allocation=None, threshold=None, n=None):
    """Phases 5-7: search generation parameters, keep the best per sample,
    then filter and regenerate what is still below the bar.

    EXPENSIVE: each search round regenerates every sample. `search_rounds=0`
    runs the filter without the search, which is the cheap path when you only
    want the rejects retried.
    """
    import anomalygen_improve as i
    return i.anomalygen_improve(dataset_name=UC, run_dir=run_dir,
                                rounds=search_rounds, allocation=allocation,
                                nn_threshold=threshold, num_sdg=n)


def publish(run_dir, run_id, target=60):
    """Publish the survivors as the next immutable version, parented on the last.

    A frame earns its defect label only if the gate accepted it; the rest land
    as `pending-review` and count toward nothing. That rule is not negotiable --
    it is the only reason the counts you read next round mean anything.
    """
    import publish_synthetic as p
    return p.publish_synthetic(hyperdataset_name=DATASET, dataset_name=UC,
                               run_dir=run_dir, run_id=run_id,
                               target_per_class=target)


def train(round_name, use_synthetic=True):
    """Train an inspector on the latest published version and register it.

    Call once with `use_synthetic=False` before you generate anything. Without
    that control there is nothing to compare against and "the synthetic data
    helped" is untestable.
    """
    import train_inspector as t
    return t.train_inspector(hyperdataset_name=DATASET, dataset_name=UC,
                             round_name=round_name, use_synthetic=use_synthetic)


if __name__ == "__main__":
    import json
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "gap"
    print(json.dumps({"gap": gap, "history": history}[what](), indent=2,
                     default=str))
