"""Entry point: register NVIDIA's real PCB data as HyperDataset `v1-real`.

The work is `cosmos_pipeline.register_real` -- this file exists so the step can
be a task in its own right (`python launch.py register`) rather than only a
node inside the pipeline. Same function either way, so the two cannot drift.

It downloads `nvidia/Cosmos-AnomalyGen-PCB-Dataset` (86 anomaly images + their
defect masks + 5 clean references) and writes one frame per image, carrying the
defect label the directory tree encodes plus a metadata dictionary. After this
the layout stops mattering: the labels are queryable server-side, which is what
lets the loop read its own gap without downloading anything.
"""
import os


def main():
    from clearml import Task

    Task.init(project_name="Physical AI Inspection",
              task_name="Register the real data (HyperDataset v1-real)",
              task_type=Task.TaskTypes.data_processing,
              auto_connect_frameworks=False, output_uri=True)

    import cosmos_pipeline as cp
    ds_id = cp.register_real(
        hyperdataset_name=os.environ.get("DEFT_HYPERDATASET", "PCB Inspection"))
    print("HYPERDATASET", ds_id, flush=True)
    return ds_id


if __name__ == "__main__":
    main()
