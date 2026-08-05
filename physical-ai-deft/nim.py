"""Launch, use and put away an NVIDIA NIM from inside the loop.

WHY THIS IS NOT "START THE NIM FIRST, THEN RUN THE LAB"
-------------------------------------------------------
An autonomous loop that requires a human to have started a model server before
it runs is not autonomous; it is a script with a prerequisite. Worse, on the
economics: a served NIM holds a whole GPU for as long as it exists, and the loop
only needs it during the seconds it spends scoring a batch. Leaving it up
between passes is the difference between a card that is busy and a card that is
billed.

So the scoring stage brings its own evaluator up, uses it, and stops it.

    inst = launch(container=COSMOS_REASON, queue="1XGPU", max_idle_hours=1)
    url  = wait_ready(inst)
    ...  score frames ...
    stop(inst)

THE FAILSAFE MATTERS MORE THAN THE TEARDOWN
--------------------------------------------
`stop()` is the happy path. The case that actually costs money is the loop dying
between launch and stop -- a crash, an abort, a pod eviction -- leaving a GPU
held by a model server nobody is talking to and nobody remembers starting.

`max_idle_hours` is passed at LAUNCH, not set afterwards, so the instance is
born knowing how to die. If this module never gets to run another line, the
server still goes away on its own. Set it even when you intend to stop
explicitly; the two are not alternatives.

KNOWN ISSUE, READ BEFORE RELYING ON THIS IN A TIGHT LOOP
---------------------------------------------------------
On this stack, stopping an app has been observed to abort the parent controller
task while leaking the child pod -- which is why a pod-reaper CronJob exists.
A loop that launches and stops a NIM on every pass would hit that on every
pass, and one leaked GPU pod per iteration is worse than never tearing down at
all. `stop()` therefore verifies the instance actually reached a terminal state
and says so loudly when it does not, rather than returning quietly and letting
the leak accumulate unremarked.

API: allegro-engine/server/schema/services/apps.conf
"""
from __future__ import annotations

import base64
import json
import os
import time

# Cosmos Reason 2 (2B): NVIDIA's reasoning VLM, small enough to share an A10G
# with a container pull and still answer quickly. Entitlement is on the lab's
# NGC key; the image pull uses the namespace's registry secret, NOT this key --
# see the recipe's `lab-credentials` note.
COSMOS_REASON = "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.7.0"

_TERMINAL = {"completed", "stopped", "failed", "closed", "aborted"}


class NimError(RuntimeError):
    pass


def _call(action: str, payload: dict, service: str = "apps") -> dict:
    from clearml.backend_api import Session
    res = Session().send_request(service=service, action=action,
                                 json=payload, method="post")
    if res.status_code != 200:
        raise NimError("%s.%s -> HTTP %s: %s"
                       % (service, action, res.status_code, res.text[:500]))
    body = res.json() or {}
    return body.get("data", body)


# ----------------------------------------------------------------- launch ---
def launch_fields(app: str = "nim") -> list[str]:
    """The field names this app's launch form accepts, straight from the server.

    Worth calling rather than hardcoding: the app owns its own form, and asking
    it what it takes is what keeps this working when NVIDIA ships a NIM with a
    field we have never heard of.
    """
    tpl = _call("get_launch_template", {"app": app}).get("launch_template") or {}
    return [f.get("name") for g in tpl.get("formFieldGroups") or []
            for f in g.get("fields") or []]


def launch(container: str = COSMOS_REASON,
           queue_name: str = "1XGPU",
           project: str = "Physical AI Inspection",
           session_name: str = "Cosmos Reason (evaluator)",
           max_idle_hours: float = 1,
           env: dict | None = None,
           app: str = "nim",
           tags: list[str] | None = None) -> str:
    """Start a NIM and return its instance id.

    Returns as soon as the launch is accepted -- the container still has to be
    pulled and the model loaded. Call `wait_ready`.
    """
    # Each value must match the TYPE ITS FORM FIELD DECLARES -- the apiserver
    # validates launch_params against the launch template, and is strict in
    # both directions. `max_idle_time_hour` is declared `string`, so 1 is
    # rejected with "'string' expected"; `run_as_root` is declared `boolean`,
    # so "true" is rejected with "'boolean' expected". There is no blanket
    # rule to apply here; when adding a field, read its `type` in
    # `launch_fields`' underlying template and match it.
    params: dict = {
        "session_name": str(session_name),
        "project": str(project),
        "container": str(container),
        "queue_name": str(queue_name),
        "max_idle_time_hour": str(max_idle_hours),   # declared string
        "run_as_root": True,                          # declared enumeration/bool
        "session_tags": ",".join(str(t) for t in (tags or ["deft", "evaluator"])),
    }
    # Environment variables. `env_key`/`env_val` are a single text PAIR in the
    # wizard (one row the UI repeats), while `environment_vars_list` is the
    # collapsed list the launcher actually consumes -- and it is REQUIRED even
    # with no env vars to set, failing the launch with "Configuration parameter
    # is missing" rather than defaulting to empty. It does not appear in
    # get_launch_template's field list at all, because the form derives it:
    # the one place where asking the app what it takes does not tell you
    # everything.
    env = dict(env or {})
    params["env_key"] = ""
    params["env_val"] = ""
    params["environment_vars_list"] = ["%s=%s" % (k, v) for k, v in env.items()]

    out = _call("launch_instance", {"app": app, "launch_params": params})
    instance = out.get("instance")
    if not instance:
        raise NimError("launch_instance returned no instance id: %r" % out)
    print("launched NIM instance", instance, "->", container)
    return instance


def info(instance: str) -> dict:
    return _call("get_instance_info", {"instance": instance}).get("info") or {}


def _find_endpoint(blob) -> str | None:
    """Pull the first http(s) URL out of whatever shape the info comes back in.

    Deliberately structural rather than keyed on a field name: the dashboard
    template decides how an app surfaces its endpoint, and NIM's is not
    guaranteed to use the same key as VSCode's. A scan that finds "the URL" is
    more durable than a guess at "info['endpoint']" that silently returns None.
    """
    if isinstance(blob, str):
        return blob if blob.startswith(("http://", "https://")) else None
    if isinstance(blob, dict):
        # Prefer obviously-named keys, then fall back to any URL present.
        for key in ("endpoint", "url", "app_url", "external_url", "address"):
            found = _find_endpoint(blob.get(key))
            if found:
                return found
        for v in blob.values():
            found = _find_endpoint(v)
            if found:
                return found
    elif isinstance(blob, (list, tuple)):
        for v in blob:
            found = _find_endpoint(v)
            if found:
                return found
    return None


def wait_ready(instance: str, timeout_s: int = 25 * 60,
               poll_s: int = 15) -> str:
    """Block until the instance serves, and return its base URL.

    The timeout is generous because the first launch on a cold node pays for a
    multi-gigabyte container pull before the model even starts loading. It is
    not unbounded: a NIM that never comes up should fail the stage, not hold the
    pipeline open until the lab expires.
    """
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        i = info(instance)
        status = (i.get("status") or "").lower()
        url = _find_endpoint(i)
        if status != last:
            print("  NIM %s: status=%s%s" % (instance, status or "?",
                                             (" url=%s" % url) if url else ""))
            last = status
        if status in _TERMINAL:
            raise NimError("NIM reached terminal status %r before serving; "
                           "check the instance task's console" % status)
        if url:
            return url.rstrip("/")
        time.sleep(poll_s)
    raise NimError("NIM %s did not serve within %ds" % (instance, timeout_s))


def stop(instance: str, wait_s: int = 120) -> bool:
    """Stop the instance, and verify it actually stopped.

    Returns True on a confirmed terminal status. Returns False -- loudly, not
    silently -- if the stop was accepted but the instance never got there,
    because that is the shape of the known pod-leak issue and a loop that
    ignores it accumulates a held GPU per pass.
    """
    try:
        _call("stop", {"tasks": [instance]}, service="tasks")
    except NimError as e:
        print("  stop request failed:", str(e)[:200])
        return False

    deadline = time.time() + wait_s
    while time.time() < deadline:
        status = (info(instance).get("status") or "").lower()
        if status in _TERMINAL:
            print("  NIM %s stopped (%s)" % (instance, status))
            return True
        time.sleep(5)

    print("!" * 66)
    print("NIM %s did NOT reach a terminal status within %ds." % (instance, wait_s))
    print("It may still be holding a GPU. This is the known app-stop pod leak --")
    print("check for an orphaned pod before running another pass.")
    print("!" * 66)
    return False


# ------------------------------------------------------------------ score ---
def _data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(path, "rb") as fh:
        return "data:%s;base64,%s" % (mime, base64.b64encode(fh.read()).decode())


def score_image(base_url: str, image_path: str, defect_class: str,
                model: str = "nvidia/cosmos-reason2-2b",
                timeout_s: int = 120) -> dict:
    """Ask the evaluator whether this image really shows `defect_class`.

    The prompt asks for a bare JSON verdict rather than prose, and the parse
    treats anything it cannot read as a REJECTION. That asymmetry is deliberate:
    an unparseable answer is not evidence the defect is there, and the whole
    reason this stage exists is to keep unverified frames out of the counts.
    """
    import requests

    readable = defect_class.replace("_", " ")
    question = (
        "You are inspecting a photograph of a printed circuit board for a "
        "quality control dataset. Does this image clearly show a '%s' defect? "
        "Answer with JSON only, no other text, in exactly this form: "
        '{"present": true or false, "confidence": 0.0 to 1.0, "reason": "one short sentence"}'
        % readable)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
        ]}],
        "max_tokens": 200,
        "temperature": 0,
    }
    r = requests.post("%s/v1/chat/completions" % base_url,
                      json=payload, timeout=timeout_s)
    if r.status_code != 200:
        return {"present": False, "confidence": 0.0,
                "reason": "evaluator HTTP %s" % r.status_code, "error": True}

    text = ""
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return {"present": False, "confidence": 0.0,
                "reason": "unreadable evaluator response", "error": True}

    # Models like to wrap JSON in prose or fences; take the first {...} block.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"present": False, "confidence": 0.0,
                "reason": "no JSON in evaluator answer: %s" % text[:120].strip(),
                "error": True}
    try:
        v = json.loads(text[start:end + 1])
    except ValueError:
        return {"present": False, "confidence": 0.0,
                "reason": "malformed JSON: %s" % text[start:start + 120],
                "error": True}

    return {"present": bool(v.get("present")),
            "confidence": float(v.get("confidence") or 0.0),
            "reason": str(v.get("reason") or "")[:200],
            "error": False}
