"""Drive N enrichment rounds, each ending in a trained model.

    for round in 1..N:
        read the gap from the latest published version   (metadata only)
        generate against it                              (NVIDIA AnomalyGen)
        score every frame                                (NVIDIA nn_score)
        publish the survivors as the next version
        train an inspector on it, and register the lineage

BOUNDED, ON PURPOSE
-------------------
`rounds` is a hard count, not a "keep going until it is good enough". An
unbounded loop on a GPU is a bill nobody approved, and the honest demo is three
rounds you can watch rather than a process that runs until someone notices. The
diagram's `go again` arrow is a decision, and a decision needs a stopping rule.

It also stops EARLY if a round adds nothing -- if the gap does not move, running
the same generation again will not move it either, and the loop should say so
rather than burn two more rounds proving it.

WHAT COMES OUT
--------------
N dataset versions and N models, each model carrying the version id it trained
on. Laid side by side that is the DEFT claim, or its refutation: accuracy per
round against synthetic-frames-accepted per round. If the line is flat, the
generated data did not help and the lab should say so.
"""
import json
import os
import sys

from ag_common import CACHE


def run_rounds(rounds=3,
               hyperdataset_name="PCB Inspection",
               dataset_name="pcb-uc1",
               num_sdg=24,
               target_per_class=60,
               seed_base=0):
    """Run `rounds` enrichment rounds end to end, in-process."""
    from clearml import Task

    import anomalygen_generate as gen
    import anomalygen_evaluate as ev
    import publish_synthetic as pub
    import train_inspector as tr
    import hyperdataset as hd

    task = Task.current_task()
    invocation = (task.id[:8] if task else "local")
    print("invocation id: %s" % invocation, flush=True)
    ds_id = hd.get_or_create_dataset(hyperdataset_name)
    history = []

    # ROUND 0: the control. Same fixed real-only holdout, same real training
    # images, no synthetic at all. Every later round is measured against this
    # one number, and without it "the generated data helped" cannot be tested.
    print("\n" + "#" * 66, flush=True)
    print("# ROUND 0 -- BASELINE (no synthetic data)", flush=True)
    print("#" * 66, flush=True)
    base = tr.train_inspector(hyperdataset_name=hyperdataset_name,
                              dataset_name=dataset_name,
                              round_name="inspector-baseline",
                              use_synthetic=False)
    print("baseline accuracy %.3f on %d real held-out images"
          % (base.get("accuracy") or 0, base.get("held_out_real") or 0), flush=True)
    history.append({"round": 0, "version": "baseline (real only)",
                    "published": 0, "verified": 0,
                    "accuracy": base.get("accuracy"), "gap_after": {}})

    for r in range(1, int(rounds) + 1):
        print("\n" + "#" * 66, flush=True)
        print("# ROUND %d of %d" % (r, rounds), flush=True)
        print("#" * 66, flush=True)

        before_v = hd.latest_published(ds_id)
        before = hd.stats(before_v["id"])["labels"] if before_v else {}
        housekeeping = {"clean", "mask", "pending-review"}
        gap = {k: target_per_class - v for k, v in before.items()
               if k not in housekeeping and v < target_per_class}
        print("gap before round %d: %s" % (r, gap or "none"), flush=True)
        if not gap:
            print("every class is at target -- stopping early", flush=True)
            break

        # RUN ID MUST BE UNIQUE ACROSS INVOCATIONS, not just across rounds.
        #
        # "round1/2/3" collides the moment the loop is run a second time: this
        # invocation's round 2 overwrites the previous invocation's round2
        # directory, so the accumulator sees the same number of directories all
        # the way through and the training set never grows. Observed -- rounds 1
        # and 2 of a re-run both trained on exactly 99 images and both scored
        # 0.929, which reads as "more data changed nothing" when in fact there
        # was no more data.
        #
        # Third time a reused name has caused this (version name, output dir,
        # now run id), so scope it to the invocation and be done.
        run_id = "%s-round%d" % (invocation, r)
        g = gen.anomalygen_generate(dataset_name=dataset_name,
                                    num_sdg=num_sdg,
                                    seed=seed_base + r,
                                    run_id=run_id)
        run_dir = g["output_dir"]
        ev.anomalygen_evaluate(dataset_name=dataset_name, run_dir=run_dir)
        p = pub.publish_synthetic(hyperdataset_name=hyperdataset_name,
                                  dataset_name=dataset_name,
                                  target_per_class=target_per_class,
                                  run_dir=run_dir)
        if not p.get("published"):
            print("round %d published nothing -- stopping early" % r, flush=True)
            break

        m = tr.train_inspector(hyperdataset_name=hyperdataset_name,
                               dataset_name=dataset_name,
                               round_name="inspector-round%d" % r)

        history.append({"round": r,
                        "version": p.get("version_name"),
                        "published": p.get("published"),
                        "verified": m.get("synthetic_accepted"),
                        "accuracy": m.get("accuracy"),
                        "gap_after": p.get("gap")})
        print("round %d done: %s acc=%.3f" % (r, p.get("version_name"),
                                              m.get("accuracy") or 0), flush=True)

    print("\n" + "=" * 66, flush=True)
    print("ROUNDS COMPLETE", flush=True)
    for h in history:
        print("  round %d  %-16s published=%-3d verified=%-3d acc=%.3f"
              % (h["round"], h["version"], h["published"],
                 h["verified"] or 0, h["accuracy"] or 0), flush=True)
    print("=" * 66, flush=True)

    if task and history:
        logger = task.get_logger()
        for h in history:
            logger.report_scalar("accuracy", "per round",
                                 value=h["accuracy"] or 0, iteration=h["round"])
            logger.report_scalar("synthetic accepted", "per round",
                                 value=h["verified"] or 0, iteration=h["round"])
        logger.report_table(
            title="the loop, round by round", series="summary", iteration=0,
            table_plot=[["round", "version", "published", "verified", "accuracy"]]
                       + [[h["round"], h["version"], h["published"],
                           h["verified"] or 0, round(h["accuracy"] or 0, 4)]
                          for h in history])
        task.upload_artifact("rounds", history)
    return history


if __name__ == "__main__":
    # DEFT_ROUNDS is how `launch.py` passes the count -- an env var lands on the
    # task record where a reader can see it, and needs no private API to set.
    n = int(os.environ.get("DEFT_ROUNDS") or 3)
    if "--rounds" in sys.argv:
        n = int(sys.argv[sys.argv.index("--rounds") + 1])
    run_rounds(rounds=n)
