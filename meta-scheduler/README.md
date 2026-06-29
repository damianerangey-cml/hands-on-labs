# Meta Scheduler — example code

*ClearML Meta Scheduler (Slurm + Kubernetes).*

### `finetune_slurm.py`
Fine-tunes on a **Slurm** EC2 node, then registers a ClearML **`OutputModel`**
whose UUID is served on the **Kubernetes** plane. Same ClearML SDK throughout —
the point is one tenant scheduling work across two compute planes, with the model
registry as the hand-off between them.

Pre-seeded by the HOL orchestrator as a ClearML task referencing this file by
commit; ordinary Python you can read and run.
