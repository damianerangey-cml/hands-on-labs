"""The one entry point. Every stage is this file; the task's comment says which.

    stage=template
    stage=preflight
    stage=status   target=60
    stage=register
    stage=train    baseline=1
    stage=round    n=48 gap=bridge:52,excess_solder:44 search=1
    stage=train    name=inspector-round1

HOW AN AGENT RUNS A STAGE
-------------------------
Once per project, run `stage=template`: it creates a draft task that carries
NVIDIA's image and the container settings a stage needs. After that, every
stage is the same three steps: clone the template, put one line of `key=value`
arguments in the clone's comment, enqueue it. Any tool that can clone a task,
edit its comment and enqueue it can drive the whole loop.

When a stage ends it prints one line, `RESULT {...}`, and appends the same line
to the task's comment, so the outcome can be read from the task list without
downloading a log.

WHAT THIS FILE DOES BEFORE THE STAGE STARTS
-------------------------------------------
NVIDIA's image carries torch and every AnomalyGen dependency in its own Python.
An agent that builds a virtual environment for the task hides them. So this
file checks whether the interpreter it was started with can import torch, and
if it cannot, finds the image's own Python and starts again under that.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Set before anything imports clearml or huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # Xet stalls on some networks
os.environ.setdefault("CLEARML_CACHE_DIR", "/tmp/clearml-cache")  # writable by any uid

DATASET = os.environ.get("DEFT_HYPERDATASET", "PCB Inspection")
UC = os.environ.get("DEFT_DATASET", "pcb-uc1")
HOUSEKEEPING = {"clean", "mask", "pending-review"}

# Stages that run NVIDIA's code and so need the image's own Python.
HEAVY = {"register", "round", "train", "train_customer", "finetune", "visa_try", "seg_check", "line_check"}
STAGES = ("template", "preflight", "status") + tuple(sorted(HEAVY))

IMAGE = os.environ.get("DEFT_IMAGE", "nvcr.io/nvidia/paidf-anomalygen:1.0.1")
TEMPLATE_NAME = "AnomalyGen stage template (clone me)"
# NVIDIA's image runs as a non-root user. On a worker whose agent keeps its
# caches under /root, that user cannot write them and the task dies before any
# of this code runs: "Repository cloning failed: Permission denied:
# /root/.clearml/vcs-cache". These move the caches somewhere any user can write.
TEMPLATE_ARGS = ("-e CLEARML_AGENT__AGENT__VCS_CACHE__PATH=/tmp/vcs-cache "
                 "-e CLEARML_AGENT__AGENT__PIP_DOWNLOAD_CACHE__PATH=/tmp/pip-cache "
                 "-e CLEARML_CACHE_DIR=/tmp/clearml-cache")

# The gated Hugging Face repositories generation downloads from. A token whose
# account has not accepted their licences gets a 403, and the stage then fails
# on "4 artifact(s) missing", which says nothing about licences.
GATED = ("nvidia/Cosmos-Predict2-2B-Text2Image", "nvidia/Cosmos-Guardrail1")


_HF_PROBE = r"""
import json, os, sys, tempfile
import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata

def why(e):
    chain = []
    while e is not None and len(chain) < 4:
        chain.append("%s: %s" % (type(e).__name__, " ".join(str(e).split())[:260]))
        e = e.__cause__ or e.__context__
    return " <- ".join(chain)

out = {}
for repo in sys.argv[1:]:
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        files = sorted(info.siblings, key=lambda s: s.size or 0)
        small = next((f.rfilename for f in files if f.rfilename.lower().endswith((".md", ".txt", ".json"))), files[0].rfilename)
        hf_hub_download(repo, small, force_download=True, local_dir=tempfile.mkdtemp())
        # The weights are served from a different host than the small files,
        # so check the largest file can be reached as well (headers only).
        big = files[-1].rfilename
        get_hf_file_metadata(hf_hub_url(repo, big))
        out[repo] = "ok"
    except Exception as e:
        out[repo] = why(e)
try:
    account = HfApi().whoami().get("name")
except Exception as e:
    account = "unknown (%s)" % why(e)
print("HFENV " + json.dumps({"account": account, "huggingface_hub": huggingface_hub.__version__,
      **{k: v for k, v in os.environ.items()
         if k.startswith(("HF_", "HUGGINGFACE_")) and "TOKEN" not in k}}))
print("HFCHECK " + json.dumps(out))
"""


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------
def parse_args(text):
    """`stage=round n=48 gap=bridge:52` -> {'stage': 'round', 'n': '48', ...}.

    Only the first line that contains `stage=` is read, so a RESULT line
    appended by an earlier run of a cloned task is ignored.
    """
    for line in (text or "").splitlines():
        if "stage=" not in line:
            continue
        out = {}
        for tok in shlex.split(line):
            key, sep, val = tok.partition("=")
            if sep and key:
                out[key.strip().lower().replace("-", "_")] = val.strip()
        return out
    return {}


def read_args():
    """Command line first (`run.py stage=status`), then the task's comment."""
    args = parse_args(" ".join(shlex.quote(a) for a in sys.argv[1:]))
    if args:
        return args
    task_id = os.environ.get("CLEARML_TASK_ID")
    if task_id:
        try:
            import clearml  # noqa: F401
        except ImportError:
            # A worker set up to skip the virtual environment runs the image's
            # Python directly, and NVIDIA's image does not ship clearml.
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "--no-input", "clearml", "scikit-learn"], check=True)
        from clearml import Task
        return parse_args(Task.get_task(task_id=task_id).comment)
    return {}


def parse_counts(text):
    """`bridge:52,excess_solder:44` -> {'bridge': 52, 'excess_solder': 44}."""
    out = {}
    for part in (text or "").split(","):
        key, _, val = part.strip().partition(":")
        if key and val:
            out[key.strip()] = int(val)
    return out


def truthy(v):
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# the image's own Python
# --------------------------------------------------------------------------
def _clean_env():
    """The environment without the task's virtual environment in it."""
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    if venv:
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep)
            if not p.startswith(venv))
    return env


def _imports(python, modules, env=None):
    try:
        return subprocess.run([python, "-c", "import " + ", ".join(modules)],
                              env=env, capture_output=True, timeout=180
                              ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def image_python():
    """An interpreter that can import torch, or None. The current one wins."""
    if _imports(sys.executable, ["torch"]):
        return sys.executable
    env = _clean_env()
    seen = []
    for cand in ([os.environ.get("DEFT_PYTHON")]
                 + [shutil.which(n, path=env.get("PATH")) for n in ("python3", "python")]
                 + ["/usr/bin/python3", "/usr/local/bin/python3",
                    "/opt/conda/bin/python3", "/usr/bin/python"]):
        if cand and cand not in seen and os.path.exists(cand):
            seen.append(cand)
            if _imports(cand, ["torch"], env):
                return cand
    return None


def reexec_under_image_python(args):
    """Start this file again under the interpreter that has NVIDIA's packages."""
    if _imports(sys.executable, ["torch"]):
        return
    if os.environ.get("DEFT_REEXEC"):
        raise SystemExit("still no torch after switching interpreter -- is this "
                         "NVIDIA's AnomalyGen image?")
    python = image_python()
    if not python:
        raise SystemExit(
            "no interpreter in this container can import torch. The task's "
            "container image must be NVIDIA's AnomalyGen image "
            "(nvcr.io/nvidia/paidf-anomalygen:1.0.1 unless you rebuilt it).")
    env = _clean_env()
    if not _imports(python, ["clearml", "sklearn"], env):
        print("installing clearml and scikit-learn for %s" % python, flush=True)
        subprocess.run([python, "-m", "pip", "install", "-q", "--no-input",
                        "clearml", "scikit-learn"], env=env, check=True)
    env["DEFT_REEXEC"] = "1"
    print("switching to the image's own Python: %s" % python, flush=True)
    sys.stdout.flush()
    os.execve(python, [python, os.path.abspath(__file__)]
              + ["%s=%s" % kv for kv in args.items()], env)


# --------------------------------------------------------------------------
# reading the dataset
# --------------------------------------------------------------------------
def read_status(target=60):
    """Every published version with its label counts, and the current gap.

    Server-side label counts: about a second, no pixels moved. This is the read
    the whole method turns on.
    """
    import hyperdataset as hd
    ds_id = hd.find_dataset(DATASET)
    if not ds_id:
        return {"dataset": None, "versions": [], "gap": {}, "at_target": [],
                "note": "no HyperDataset %r here yet -- run stage=register" % DATASET}
    versions = hd._call("get_versions", {"dataset": ds_id, "only_published": True,
                                         "page": 0, "page_size": 200}
                        ).get("versions") or []
    versions.sort(key=lambda v: (v.get("created") or "", v.get("name") or ""))
    out = [{"name": v.get("name"), "id": v["id"],
            "counts": hd.stats(v["id"])["labels"]} for v in versions]
    counts = out[-1]["counts"] if out else {}
    return {
        "dataset": ds_id, "target": target, "versions": out,
        "latest": out[-1]["name"] if out else None,
        "counts": counts,
        "gap": {k: target - n for k, n in counts.items()
                if k not in HOUSEKEEPING and n < target},
        "at_target": sorted(k for k, n in counts.items()
                            if k not in HOUSEKEEPING and n >= target),
        "pending_review": counts.get("pending-review", 0),
    }


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
GATED_STAGES = {"round", "train", "train_customer", "finetune", "visa_try", "seg_check", "line_check"}
PREFLIGHT_HOURS = 24


def require_preflight():
    """Refuse to run a GPU stage on a queue no preflight has passed on.

    AGENT.md says to stop on a failed preflight, and an agent once read that,
    watched its preflight fail, found an older passing result from a different
    worker and carried on. A rule in a document is advice. This is the rule in
    code: a GPU stage looks for a preflight that completed with ready=true on
    THE SAME QUEUE in the last 24 hours, and exits if there is none.
    """
    task_id = os.environ.get("CLEARML_TASK_ID")
    if not task_id:
        return                      # not running as a task: nothing to gate
    import datetime
    from clearml.backend_api.session.client import APIClient
    client = APIClient()
    me = client.tasks.get_by_id(task=task_id)
    queue = getattr(getattr(me, "execution", None), "queue", None)
    if not queue:
        return                      # started outside a queue; nothing to compare
    since = (datetime.datetime.utcnow()
             - datetime.timedelta(hours=PREFLIGHT_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
    done = client.tasks.get_all(
        project=[me.project], status=["completed"],
        status_changed=[">=" + since], order_by=["-status_changed"],
        only_fields=["id", "comment", "execution.queue"], page_size=500)
    for t in done:
        comment = t.comment or ""
        if (getattr(getattr(t, "execution", None), "queue", None) == queue
                and comment.lstrip().startswith("stage=preflight")
                and '"ready": true' in comment):
            print("preflight %s passed on this queue" % t.id, flush=True)
            return
    raise SystemExit(
        "NO PASSING PREFLIGHT ON THIS QUEUE in the last %d hours, so this stage "
        "will not run.\nRun stage=preflight on the same queue first. If it "
        "fails, fix what it reports; a preflight that passed on another queue "
        "says nothing about this one." % PREFLIGHT_HOURS)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_template(args):
    """Create the task every other stage is cloned from. Run once per project.

    A stage needs container arguments that a plain "create task" call cannot
    always set (see TEMPLATE_ARGS). This stage writes them once, onto a draft
    task; cloning that draft copies them, so every stage after this one is
    clone, set the comment, enqueue. It needs no GPU and not NVIDIA's image.
    """
    from clearml import Task
    from ag_common import bind_task
    task = bind_task()
    script = getattr(getattr(task, "data", None), "script", None) if task else None
    project = args.get("project") or (task.get_project_name() if task else None)
    repo = args.get("repo") or getattr(script, "repository", None)
    branch = args.get("branch") or getattr(script, "branch", None)
    if not (project and repo and branch):
        raise SystemExit("template needs project=, repo= and branch= when it "
                         "is not run as a task")
    for old in Task.get_tasks(project_name=project, task_name="^%s$"
                              % TEMPLATE_NAME.replace("(", r"\(").replace(")", r"\)"),
                              task_filter={"status": ["created"],
                                           "system_tags": ["-archived"]}):
        return {"template": old.id, "name": TEMPLATE_NAME, "reused": True}
    t = Task.create(project_name=project, task_name=TEMPLATE_NAME,
                    task_type=Task.TaskTypes.data_processing,
                    repo=repo, branch=branch,
                    script="physical-ai-deft/run.py", working_directory=".",
                    docker=IMAGE, docker_args=TEMPLATE_ARGS)
    t.set_comment("Do not enqueue this task. Clone it, set the clone's comment "
                  "to the stage arguments, and enqueue the clone.")
    return {"template": t.id, "name": TEMPLATE_NAME, "reused": False,
            "repo": repo, "branch": branch}


def stage_preflight(args):
    """Can this queue run the loop? Checks everything that fails late and
    obscurely otherwise, and reports it in one place. Prints no secret."""
    blocking, notes, facts = [], [], {}

    facts["python"] = sys.executable
    py = image_python()
    facts["image_python"] = py
    if not py:
        blocking.append("no Python in this container imports torch -- the task "
                        "is not running in NVIDIA's AnomalyGen image")
    else:
        probe = subprocess.run(
            [py, "-c",
             "import torch, json; ok = torch.cuda.is_available(); "
             "print(json.dumps({'torch': torch.__version__, 'cuda': ok, "
             "'gpu': torch.cuda.get_device_name(0) if ok else None, "
             "'gpu_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if ok else 0, "
             "'gpus': torch.cuda.device_count() if ok else 0}))"],
            env=_clean_env(), capture_output=True, text=True)
        try:
            facts.update(json.loads(probe.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            facts["torch_error"] = (probe.stderr or probe.stdout)[-300:]
        if not facts.get("cuda"):
            blocking.append("no GPU is visible to torch on this queue")
        elif facts.get("gpu_gb", 0) < 22:
            blocking.append("the GPU has %.1f GB; generation needs a whole GPU "
                            "of about 24 GB" % facts["gpu_gb"])
        elif facts.get("gpu_gb", 0) < 30:
            notes.append("enough for every stage except finetune, which needs "
                         "more than 24 GB; NVIDIA's published adapter is used instead")

    import ag_common
    facts["nvidia_code"] = os.path.isdir(ag_common.REPO_ROOT)
    if not facts["nvidia_code"]:
        blocking.append("NVIDIA's code is not at %s (set DEFT_AG_REPO_ROOT if "
                        "your image keeps it elsewhere)" % ag_common.REPO_ROOT)

    facts["uid"] = os.getuid()
    facts["shm_gb"] = round(shutil.disk_usage("/dev/shm").total / 2**30, 1) \
        if os.path.isdir("/dev/shm") else 0
    if facts["shm_gb"] < 8:
        notes.append("/dev/shm is %.1f GB; NVIDIA's docker run line uses 16 GB "
                     "and data loaders may fail below that" % facts["shm_gb"])
    facts["cache"] = ag_common.CACHE
    facts["cache_persists"] = bool(os.environ.get("DEFT_CACHE")) \
        and ag_common.CACHE == os.environ.get("DEFT_CACHE")
    facts["cache_free_gb"] = round(shutil.disk_usage(ag_common.CACHE).free / 2**30, 1)
    if facts["cache_free_gb"] < 45:
        blocking.append("%.0f GB free at %s; the checkpoints alone are about "
                        "22 GB" % (facts["cache_free_gb"], ag_common.CACHE))
    if not facts["cache_persists"]:
        notes.append("no persistent cache: each task downloads about 22 GB of "
                     "checkpoints again. Pass cache=<path> if the workers have "
                     "a volume that outlives a task")

    creds = {k: bool(os.environ.get(k)) for k in ("HF_TOKEN", "NGC_KEY", "NGC_API_KEY")}
    facts["credentials_present"] = creds
    if not creds["HF_TOKEN"]:
        blocking.append("HF_TOKEN is not in the task environment (it belongs in "
                        "the platform's secret store or your Configuration "
                        "Vault, never on the task)")
    elif py:
        # Download one small file from each gated repository, with the same
        # library and interpreter NVIDIA's downloader uses. Asking the API
        # whether the token "has access" is not the same thing: it said yes on
        # a worker where the real download then failed.
        probe = subprocess.run([py, "-c", _HF_PROBE] + list(GATED),
                               env=_clean_env(), capture_output=True, text=True)
        access = {}
        for line in probe.stdout.splitlines():
            if line.startswith("HFCHECK "):
                access = json.loads(line[8:])
            elif line.startswith("HFENV "):
                facts["hf_env"] = json.loads(line[6:])
        if not access:
            access = {r: "probe failed: %s" % (probe.stderr or probe.stdout)[-300:]
                      for r in GATED}
        facts["hf_access"] = access
        for repo, verdict in access.items():
            if verdict != "ok":
                blocking.append("cannot download from %s with this HF_TOKEN: %s"
                                % (repo, verdict))
    if not (creds["NGC_KEY"] or creds["NGC_API_KEY"]):
        notes.append("NGC_KEY / NGC_API_KEY are not set; the image pulled, so "
                     "this only matters if a stage fetches from NGC")

    try:
        status = read_status(int(args.get("target") or 60))
        facts["hyperdatasets"] = "ok"
        facts["dataset"] = status.get("latest") or status.get("note")
        facts["gap"] = status.get("gap")
    except Exception as e:
        facts["hyperdatasets"] = "%s: %s" % (e.__class__.__name__, str(e)[:200])
        blocking.append("the HyperDataset API did not answer -- the loop keeps "
                        "its state there")

    return {"ready": not blocking, "blocking": blocking, "notes": notes,
            "facts": facts}


def stage_status(args):
    return read_status(int(args.get("target") or 60))


def stage_register(args):
    import register_real
    ds_id = register_real.register_real(hyperdataset_name=DATASET)
    return dict(read_status(int(args.get("target") or 60)), dataset=ds_id)


def stage_round(args):
    """One enrichment round in one task: read the gap, generate against it,
    score every frame against the real examples, optionally search, publish
    the survivors as the next version.

    Arguments, all optional:
      n          how many images to generate (default 24)
      gap        label:count pairs to allocate against (default: read from the
                 dataset at `target`)
      counts     NVIDIA defect type:count pairs, the explicit allocation; wins
                 over gap. Types are in the log as `IC+bridge` and so on.
      search     N runs NVIDIA's phases 5-7 with N parameter re-rolls (0 runs
                 the filter and regeneration without the search). Leave it out
                 to publish what the gate accepted from the first batch.
                 EXPENSIVE: each re-roll regenerates every sample.
      threshold  the nn_score bar (default: the batch median)
      target     per-class target (default 60)
      seed       generation seed (default 0)
      run_id     defaults to this task's id, which is unique; reusing one makes
                 publish skip the round
    """
    import anomalygen_evaluate as ev
    import anomalygen_generate as gen
    import publish_synthetic as pub
    from ag_common import bind_task

    task = bind_task()
    target = int(args.get("target") or 60)
    n = int(args.get("n") or 24)
    seed = int(args.get("seed") or 0)
    thr = float(args["threshold"]) if args.get("threshold") else None
    run_id = args.get("run_id") or (task.id[:8] if task else "local")

    before = read_status(target)
    if not before.get("latest"):
        raise SystemExit("nothing published yet -- run stage=register first")
    counts = parse_counts(args.get("counts")) or None
    gap = parse_counts(args.get("gap")) or before["gap"]
    if not counts and not gap:
        return {"published": 0, "note": "every class is at target; nothing to do",
                "gap": {}, "version": before["latest"]}

    g = gen.anomalygen_generate(dataset_name=UC, num_sdg=n, seed=seed,
                                run_id=run_id, gap=gap, per_defect_counts=counts)
    run_dir = g["output_dir"]
    ev.anomalygen_evaluate(dataset_name=UC, run_dir=run_dir, nn_threshold=thr)

    publish_dir = run_dir
    if args.get("search") not in (None, ""):
        import anomalygen_improve as improve
        imp = improve.anomalygen_improve(
            dataset_name=UC, run_dir=run_dir, rounds=int(args["search"]),
            num_sdg=n, allocation=g.get("allocation"),
            step=g.get("step") or 14000, checkpoint_dir=g.get("checkpoint"),
            seed=seed, nn_threshold=thr)
        publish_dir = imp["searched_dir"]

    p = pub.publish_synthetic(hyperdataset_name=DATASET, dataset_name=UC,
                              target_per_class=target, run_dir=publish_dir,
                              run_id=run_id, nn_threshold=thr)
    after = read_status(target)
    added = {k: after["counts"].get(k, 0) - before["counts"].get(k, 0)
             for k in after["counts"]
             if after["counts"].get(k, 0) != before["counts"].get(k, 0)}
    return {"run_id": run_id, "version": p.get("version_name"),
            "asked": g.get("allocation"), "generated": g.get("delivered"),
            "published": p.get("published"), "added": added,
            "counts": after["counts"], "gap": after["gap"],
            "pending_review": after["pending_review"]}


def stage_train(args):
    """Train the inspector on the latest published version and register it.

    baseline=1 is the control: real images only. Run it once before any round,
    or there is nothing to compare later models with.
    """
    import train_inspector
    baseline = truthy(args.get("baseline"))
    return train_inspector.train_inspector(
        hyperdataset_name=DATASET, dataset_name=UC,
        round_name=args.get("name") or ("inspector-baseline" if baseline else None),
        use_synthetic=not baseline)


def stage_train_customer(args):
    """Fine-tune the customer's inspection model end to end and score it.

    baseline=1 trains on real images only (the control); oversample=1 repeats
    the rare real images instead of adding synthetic ones; otherwise it trains
    on real plus the accepted synthetic frames.
    """
    import train_customer
    mode = ("real" if truthy(args.get("baseline"))
            else "oversample" if truthy(args.get("oversample")) else "synthetic")
    return train_customer.train_customer(
        hyperdataset_name=DATASET, dataset_name=UC, model_name=args.get("name"),
        mode=mode, hf_model=args.get("model"),
        folds=int(args.get("folds", 4)), seeds=int(args.get("repeats", 3)),
        steps=int(args.get("steps", 150)), lr=args.get("lr"))


def stage_visa_try(args):
    """Experiment: paint NVIDIA's defect types onto good VisA boards."""
    import visa_try
    return visa_try.visa_try(n=int(args.get("n", 16)), seed=int(args.get("seed", 1)))


def stage_seg_check(args):
    """Experiment: an inspector that outlines the defect, with and without
    synthetic frames."""
    import seg_check
    return seg_check.seg_check(
        dataset_name=UC, n=int(args.get("n", 48)), seed=int(args.get("seed", 0)),
        folds=int(args.get("folds", 4)), repeats=int(args.get("repeats", 3)),
        steps=int(args.get("steps", 300)))


def stage_line_check(args):
    """Experiment: replay a production line and score the inspector each round."""
    import line_check
    return line_check.line_check(
        seed=int(args.get("seed", 0)), repeats=int(args.get("repeats", 2)),
        rounds=int(args.get("rounds", 6)), start=int(args.get("start", 40)),
        steps=int(args.get("steps", 150)), hf_model=args.get("model") or "microsoft/resnet-18")


def stage_finetune(args):
    import anomalygen_finetune
    return anomalygen_finetune.anomalygen_finetune(dataset_name=UC)


# --------------------------------------------------------------------------
def finish(args, result):
    line = "RESULT " + json.dumps(result, default=str, sort_keys=True)
    print("=" * 70, flush=True)
    print(line, flush=True)
    print("=" * 70, flush=True)
    if os.environ.get("CLEARML_TASK_ID"):
        from ag_common import bind_task
        task = bind_task()
        if task:
            first = " ".join("%s=%s" % kv for kv in args.items())
            task.set_comment(first + "\n" + line[:6000])


def main():
    args = read_args()
    stage = (args.get("stage") or "").replace("-", "_")
    if stage not in STAGES:
        raise SystemExit(
            "No stage to run. Put one line of arguments in the task's comment, "
            "for example:  stage=preflight\nStages: %s\n(got: %r)"
            % (", ".join(STAGES), args or None))
    if args.get("cache"):
        os.environ["DEFT_CACHE"] = args["cache"]
    if stage in GATED_STAGES and not os.environ.get("DEFT_REEXEC"):
        require_preflight()
    if stage in HEAVY:
        reexec_under_image_python(args)

    print("stage %s  %s" % (stage, {k: v for k, v in args.items() if k != "stage"}),
          flush=True)
    result = globals()["stage_" + stage](args)
    finish(args, result)
    if stage == "preflight" and not result["ready"]:
        raise SystemExit("PREFLIGHT FAILED: " + "; ".join(result["blocking"]))


if __name__ == "__main__":
    main()
