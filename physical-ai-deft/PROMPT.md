# The kickoff prompt

Launch the platform's **Claude Code** app on a CPU compute resource, in the
project the tasks should land in. The app wires the platform's MCP server into
the session. Then paste this, with your two queue names filled in:

```
Clone https://github.com/damianerangey-cml/hands-on-labs.git and read
physical-ai-deft/AGENT.md before doing anything. You drive this platform
through your MCP tools only: you have no GPU and run nothing heavy yourself.

GPU queue: <a queue with a whole GPU of 24 GB or more>
CPU queue: <a queue with no GPU>
Project: this session's project.

1. Run the preflight on the GPU queue and show me what it found. Stop if it
   is not ready.
2. Train the control on real images only.
3. Get every defect class to 60 examples with NVIDIA's Cosmos AnomalyGen.
   Budget: 3 rounds. Stop early if a round adds nothing or the inspector
   stops improving.

Label each claim VERIFIED or INFERRED. At the end tell me what you decided and
why, including what did not pay off, and do not let the dataset claim more
than it has.
```

That is the whole handover. Everything the agent needs to know about running a
stage is in `AGENT.md`; everything it needs to know about this server is the
two queue names.

To steer it afterwards, use plain words:

```
Round 1 looks good. Keep going: generate for every class still under 60,
then retrain the inspector and compare it with the control.
```

## Before the first run on a new server

| What | Where it comes from |
|---|---|
| `HF_TOKEN` in the task environment | Your Configuration Vault (an `environment { ... }` block), or a secret the platform projects into tasks. |
| Licences accepted for `nvidia/Cosmos-Predict2-2B-Text2Image` and `nvidia/Cosmos-Guardrail1` | On huggingface.co, with the account that owns the token. |
| Pull access to `nvcr.io` on the workers | An image pull secret, set up by whoever runs the cluster. |
| HyperDatasets | A Klique (ClearML Enterprise) feature; the loop keeps its state there. |

The preflight stage checks all four and says which one is missing.
