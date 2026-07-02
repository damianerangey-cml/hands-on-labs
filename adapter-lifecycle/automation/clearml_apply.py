#!/usr/bin/env python3
"""GitOps deploy for the adapter-lifecycle lab -- deploy the multi-model vLLM
endpoint from a Git-tracked config, no UI.

This is the "ArgoCD / CI instead of the UI" answer. The desired endpoint state
lives in `multimodel-serve.json` (next to this script, in Git). This script:
  1. resolves the latest SupportBot / SalesBot models from the ClearML registry
     BY TAG (so you never hand-paste a UUID),
  2. injects their UUIDs into the config, and
  3. launches / updates the deployment via the ClearML apps API.

Run it from a shell, a CI job, or an ArgoCD PreSync/Sync hook with ClearML
credentials configured (CLEARML_API_* env or ~/clearml.conf). Idempotent: edit
the JSON in Git and re-run to roll the endpoint forward.

    python clearml_apply.py [path/to/multimodel-serve.json]

NOTE: the apps.launch_instance binding below uses the clearml backend Session;
verify the call shape against your SDK version on first run.
"""
import json
import sys
from pathlib import Path

from clearml import Model
from clearml.backend_api import Session

CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "multimodel-serve.json"
REGISTRY_PROJECT = "Fine-Tuned Chatbots"
APP_ID = "model-deployment"


def latest_model_id_by_tag(tag: str) -> str:
    """Newest registered model carrying `tag`, searched across ALL projects so it
    resolves models trained by hand (in Fine-Tuned Chatbots) OR by the pipeline (which
    registers its models in its own pipeline project, not Fine-Tuned Chatbots)."""
    models = Model.query_models(tags=[tag], max_results=1)
    if not models:
        raise SystemExit(
            "no model tagged '%s' in the registry -- train that project first" % tag)
    print("resolved tag '%s' -> model %s" % (tag, models[0].id))
    return models[0].id


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # Each instance's served_model_name IS the project tag (SupportBot/SalesBot);
    # resolve it to the latest registered model UUID.
    for inst in cfg.get("config_per_instance", []):
        tag = inst.get("served_model_name")
        if not tag:
            raise SystemExit("each config_per_instance entry needs a served_model_name (== project tag)")
        inst["model"] = latest_model_id_by_tag(tag)

    session = Session()
    res = session.send_request(
        service="apps", action="launch_instance", method="post",
        json={"app": APP_ID, "launch_params": cfg,
              "task_name": cfg.get("task_name", "project-models-serve")})
    ok = getattr(res, "ok", False)
    body = getattr(res, "text", str(res))
    print("apps.launch_instance ->", getattr(res, "status_code", "?"), body[:400])
    if not ok:
        raise SystemExit("deploy failed")
    print("deployed. endpoint will be live once the serving pod is healthy.")


if __name__ == "__main__":
    main()
