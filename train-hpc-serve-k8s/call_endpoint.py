"""Call the vLLM endpoint serving the fine-tuned model.

    export ENDPOINT="<endpoint URL shown on the vLLM app instance, ending in /v1>"
    export TOKEN="<a Klique access token>"
    python call_endpoint.py "Explain LoRA in two sentences."
"""
import os
import sys

from openai import OpenAI   # the endpoint is OpenAI-compatible

client = OpenAI(base_url=os.environ["ENDPOINT"], api_key=os.environ["TOKEN"])
served = os.environ.get("SERVED_MODEL", "chat-finetune")   # ADAPT: the app's endpoint name
reply = client.chat.completions.create(
    model=served, max_tokens=200,
    messages=[{"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "Hello"}])
print(reply.choices[0].message.content)
