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

DATASET = os.environ.get("DEFT_HYPERDATASET", "PCB Inspection")
UC = os.environ.get("DEFT_DATASET", "pcb-uc1")

# Housekeeping labels are not defect classes -- they must never look like a gap.
HOUSEKEEPING = {"clean", "mask", "pending-review"}


def _ds():
    import hyperdataset as hd
    return hd, hd.get_or_create_dataset(DATASET)


def gap(target=60):
    """What each defect class is short of, and what is there now.

    Returns {"version", "version_id", "counts", "gap", "at_target"}.

    This is the read the whole method turns on: it is server-side label counts,
    so it costs about a second, downloads nothing, and can therefore be asked
    on every pass without thinking about it. `gap` excludes classes already at
    target -- generating more of what you have enough of is the waste the
    method exists to avoid.
    """
    hd, ds_id = _ds()
    v = hd.latest_published(ds_id)
    if not v:
        return {"version": None, "version_id": None, "counts": {}, "gap": {},
                "at_target": []}
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
    hd, ds_id = _ds()
    from clearml import Model
    versions = hd._call("get_versions", {"dataset": ds_id, "only_published": True,
                                         "page": 0, "page_size": 200}
                        ).get("versions") or []
    versions.sort(key=lambda v: v.get("created") or "")
    out_v = [{"name": v.get("name"), "id": v["id"], "created": v.get("created")}
             for v in versions]
    try:
        models = [{"name": m.name, "id": m.id,
                   "version": (m.get_metadata() or {}).get("version_id")}
                  for m in Model.query_models(project_name="Physical AI Inspection")]
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
