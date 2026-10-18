"""Call the vLLM endpoint serving the fine-tuned model.

From a laptop, with a Klique access token (Settings > Workspace configuration):

    export ENDPOINT="<endpoint URL shown on the vLLM app instance, ending in /v1>"
    export TOKEN="<a Klique access token>"
    python call_endpoint.py --prompt "Explain LoRA in two sentences."

As a Klique task (clone it, set Args/endpoint, enqueue to a CPU compute resource): with no TOKEN
set, the script records itself as a task and calls the endpoint with that task's own access token,
so no token is copied anywhere.
"""
import argparse
import os

from openai import OpenAI   # the endpoint is OpenAI-compatible

p = argparse.ArgumentParser()
p.add_argument("--endpoint", default=os.environ.get("ENDPOINT", ""))
p.add_argument("--model", default=os.environ.get("SERVED_MODEL", "chat-finetune"))   # ADAPT: the endpoint name
p.add_argument("--prompt", default="Explain LoRA in two sentences.")
p.add_argument("--max_tokens", type=int, default=200)

token = os.environ.get("TOKEN")
if not token:   # no token given: run as a Klique task and use its own access token
    from clearml import Task
    task = Task.init(project_name="Builders - Train on Slurm, Serve on Kubernetes",   # ADAPT
                     task_name="Call the endpoint", task_type=Task.TaskTypes.inference)
    token = task.session.token
args = p.parse_args()

client = OpenAI(base_url=args.endpoint, api_key=token)
reply = client.chat.completions.create(
    model=args.model, max_tokens=args.max_tokens,
    messages=[{"role": "user", "content": args.prompt}])
print("Prompt:", args.prompt)
print("Answer:", reply.choices[0].message.content)
