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

    inst = launch(container=COSMOS_REASON, queue_name=deft.pick_queue("gpu"),
                  max_idle_hours=1)
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

PROJECT = os.environ.get("DEFT_PROJECT", "Physical AI Inspection")

# Cosmos Reason 2 (2B): NVIDIA's reasoning VLM, small enough to share an A10G
# with a container pull and still answer quickly. Entitlement is on the lab's
# NGC key; the image pull uses the namespace's registry secret, NOT this key --
# see the recipe's `lab-credentials` note.
COSMOS_REASON = "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.7.0"

# NIM images ship with NVIDIA's INTERNAL pip index baked into their pip config
# (urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi). It is not reachable from
# outside NVIDIA, so the app's default setup script -- which bootstraps pip and
# then installs clearml-agent -- dies on DNS:
#
#   Failed to establish a new connection: [Errno -5] No address associated
#   with hostname ... /simple/clearml-agent/
#   ERROR: No matching distribution found for clearml-agent
#   /usr/bin/python3.12: No module named clearml_agent
#
# and the pod sits Running with nothing serving. Forcing the public index for
# the bootstrap is enough; it does not affect the model server, which is
# already baked into the image and installs nothing.
#
# Comments are NOT supported in this field (the app's own hint says so) -- keep
# it to plain commands.
_SETUP_SCRIPT = (
    "export PIP_INDEX_URL=https://pypi.org/simple\n"
    "export PIP_EXTRA_INDEX_URL=\n"
    "export PIP_TRUSTED_HOST=pypi.org\n"
    'python3 -c "import urllib.request, ssl; ctx = ssl.create_default_context(); '
    "ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE; "
    "code = urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py', "
    'context=ctx).read(); exec(code)"\n'
    "python3 -m pip install --index-url https://pypi.org/simple clearml-agent\n"
    "export LOCAL_PYTHON=python3\n"
)

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
           queue_name: str | None = None,   # None -> deft.pick_queue("gpu")
           project: str = PROJECT,
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
        "queue_name": str(queue_name or __import__("deft").pick_queue("gpu")),
        "max_idle_time_hour": str(max_idle_hours),   # declared string
        "run_as_root": True,                          # declared enumeration/bool
        "session_tags": ",".join(str(t) for t in (tags or ["deft", "evaluator"])),
        "setup_shell_script": _SETUP_SCRIPT,
    }
    # Environment variables go over as `environment_vars_list`: a list of
    # {env_key, env_val} OBJECTS.
    #
    # `env_key` and `env_val` show up as top-level names in the flattened
    # launch template, which is misleading -- they are the list's item_template
    # (see container_launcher.app.conf), not fields in their own right. Sending
    # them at the top level, or sending the list as "KEY=VALUE" strings, gets
    # you a 500 with "'str' object has no attribute 'get'" once the server
    # iterates it. An EMPTY list passes either way, which is exactly why this
    # only surfaces the first time you actually set a variable.
    #
    # The key itself is required even when empty: omit it and the launch fails
    # with "Configuration parameter is missing".
    # SKIP_PYTHON_ENV_INSTALL is not optional here, which is why it is a
    # default rather than something a caller remembers. Without it the agent
    # builds a fresh venv inside the image and runs the session out of THAT --
    # so the session starts in a Python that has no vllm, no NIM runtime,
    # nothing the image was built for, and dies with "Process failed, exit code
    # 1" after a long and completely misleading pip log. Same trap the Cosmos
    # stages hit; see run_stage.py.
    env = dict(env or {})
    env.setdefault("CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL", "1")
    params["environment_vars_list"] = [
        {"env_key": str(k), "env_val": str(v)} for k, v in env.items()]

    out = _call("launch_instance", {"app": app, "launch_params": params})
    instance = out.get("instance")
    if not instance:
        raise NimError("launch_instance returned no instance id: %r" % out)
    print("launched NIM instance", instance, "->", container)
    return instance


def launch_via_container(container: str = COSMOS_REASON,
                         command_line: str = "bash /opt/nim/start_server.sh",
                         internal_port: int = 8000,
                         queue_name: str | None = None,   # None -> pick_queue
                         project: str = PROJECT,
                         session_name: str = "Cosmos Reason (evaluator)",
                         max_idle_hours: float = 1,
                         env: dict | None = None,
                         tags: list[str] | None = None) -> str:
    """Same lifecycle, launched through `container_launcher` instead of `nim`.

    WHY THIS EXISTS. The `nim` app's session manager execs the image's start
    script directly, and on cosmos-reason2-2b:1.7.0 that fails:

        OSError: [Errno 8] Exec format error: '/opt/nim/start_server.sh'

    The script is not the problem -- on the same image, on the same node, in a
    plain pod, `/opt/nim/start_server.sh --help` exits 0. The app and this image
    disagree about something, and we do not need to win that argument to get an
    evaluator: `container_launcher` takes a `command_line` and explicitly
    ignores the image's entrypoint, so running the script THROUGH bash sidesteps
    the exec path entirely.

    Everything else is unchanged -- `wait_ready`, `score_image` and `stop` do not
    care which app launched the instance, because an instance is an instance.

    router_type=http gets the same JWT-authenticated gateway endpoint the nim
    app produces; tcp would expose the port unauthenticated.
    """
    env = dict(env or {})
    env.setdefault("CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL", "1")
    # CAP THE CONTEXT WINDOW OR THE ENGINE WILL NOT START ON AN A10G.
    #
    # cosmos-reason2-2b defaults to max_model_len=262144. vLLM sizes its KV
    # cache to serve one request at the full context, and refuses to start if
    # it cannot:
    #
    #   ValueError: To serve at least one request with the model's max seq len
    #   (262144), 29.0 GiB KV cache is needed, which is larger than the
    #   available KV cache memory (12.33 GiB)
    #
    # The card has 22.1 GiB; weights take 4.74 and the rest is not enough for a
    # 256K context. Nothing platform-specific about this -- the same container
    # fails the same way under plain `docker run` on the same GPU.
    #
    # 32768 is far more than this stage needs: one image plus a two-sentence
    # question is a few thousand tokens. vLLM's own estimate for this card was
    # 111440, so there is a wide margin.
    env.setdefault("NIM_MAX_MODEL_LEN", "32768")

    params = {
        "session_name": str(session_name),
        "project": str(project),
        "container": str(container),
        "command_line": str(command_line),
        "working_dir": "/opt/nim",
        "queue_name": str(queue_name or __import__("deft").pick_queue("gpu")),
        "router_type": "http",
        "router_internal_port": int(internal_port),
        "max_idle_time_hour": str(max_idle_hours),
        "run_as_root": True,
        # The image ships its own Python with the NIM runtime in it. Letting
        # the agent build a venv is the same failure the nim app hits.
        "disable_clearml_venv": True,
        "continuous_console_logging": True,
        "monitor_endpoint": True,
        "session_tags": ",".join(str(t) for t in (tags or ["deft", "evaluator"])),
        "setup_shell_script": _SETUP_SCRIPT,
        "environment_vars_list": [
            {"env_key": str(k), "env_val": str(v)} for k, v in env.items()],
        # `config_files` is the second derived list on this form, with
        # target_file/config_content as its item_template -- exactly the same
        # shape as environment_vars_list, and required even when empty.
        # If a launch 400s with "Configuration parameter is missing", look for
        # a list whose item fields you are sending at the top level instead.
        "config_files": [],
        "storage_session": "",
        "container_args": "",
    }
    out = _call("launch_instance", {"app": "containerlaunch",
                                    "launch_params": params})
    instance = out.get("instance")
    if not instance:
        raise NimError("launch_instance returned no instance id: %r" % out)
    print("launched", container, "via container_launcher ->", instance)
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
        # `task` singular, and `force` -- tasks.stop rejects a `tasks` LIST with
        # a validation error, and without force it refuses an instance that is
        # still starting up, which is exactly when a failed launch needs killing.
        _call("stop", {"task": instance, "force": True}, service="tasks")
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
