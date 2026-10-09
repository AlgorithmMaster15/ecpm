"""Agentic cost check: real token use and cost of single runs, per arm. Stdlib only. Keeps everything local.

    python3 run_agentic_cost_check.py                  # all 5 seeds x 3 arms, silent break
    python3 run_agentic_cost_check.py --seeds 8        # quick check on one seed (3 runs)
    python3 run_agentic_cost_check.py --mode sto       # stochastic instead of deterministic
    python3 run_agentic_cost_check.py --dry            # same flow with no API calls
    python3 run_agentic_cost_check.py --condition redirect      # another scenario (default silent_break)
    python3 run_agentic_cost_check.py --openrouter MODEL_ID --price-in 0.3 --price-out 1.2   # via OpenRouter
    python3 run_agentic_cost_check.py --reasoning off   # or on; omitted = model default (not controlled)
    python3 run_agentic_cost_check.py --help            # this text

All options (combine freely):
  --seeds 8,13        graphs (default 8,13,25,0,1)
  --mode det|sto      deterministic or stochastic world (default det)
  --condition NAME    no_change | irrelevant | silent_break | hard_removal | redirect | degradation (sto only)
  --history NAME      none | retained_reports_v2 | separate_reports_post_task_v1 (default none)
  --matched-prep      preparation turn for every arm, as in ICL
  --reasoning off|on  sends an explicit control: Azure reasoning_effort none/medium,
                      OpenRouter reasoning {enabled: false} / {effort: medium}
  --reasoning-control JSON   override that control, e.g. '{"reasoning_effort": "low"}'
  --openrouter MODEL_ID      run through OpenRouter instead of Azure
  --price-in X --price-out Y USD per million tokens, used only when the provider reports
                      no billed cost (OpenRouter reports it, cache discounts included)
  --max-tokens N      output cap per call, reasoning included (default 16384)
  --dry               no API calls
  --yes               skip the confirmation after the first run (for unattended loops)
Every option is recorded in the run folder name and in each run's artifact.

What it does:
  - runs every line of RUN_DIR/queue.txt ("condition seed repeat"), creating the default
    queue (3 arms x the chosen seeds, one run each) on first start
  - re-reads queue.txt after every run, so you can add lines while it is running
  - skips runs that already finished, so you can stop and restart at any time
  - stops cleanly after the current run if you create a file called STOP in RUN_DIR
  - after the first live run, shows its real token count and the projected total,
    and asks once whether to continue
  - after every run, rewrites RUN_DIR/exposure.csv (tokens, cost inputs, exposure)
    and prints progress; at the end zips RUN_DIR next to it
The API key is read from AZURE_OPENAI_API_KEY or asked once and stored in
~/.ecpm_azure.json (outside the repository, readable only by you).
"""
import datetime, getpass, json, os, shutil, subprocess, sys

# ---- settings (edit here) -------------------------------------------------------
SEEDS, REPEATS = [8, 13, 25, 0, 1], [0]          # the 5 graphs from the Results-tab matrix, one run each
CONDITIONS = ["task_only", "model_first", "graph_given"]
MODE = "det"                                      # the Results-tab plan starts with the deterministic case
SETUP = ["--pilot-type", "active", "--mode", MODE, "--scenario", "seed7_silent_break",
         "--m0-episodes", "4", "--m1-episodes", "4", "--max-steps-per-episode", "20"]
PRICE_IN, PRICE_OUT = 2.50, 10.00            # USD per million tokens (GPT-4o list price; check yours)
OPENROUTER_URL = "https://openrouter.ai/api/v1"
RUN_TIMEOUT = 3600                            # seconds per run
# ----------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry" in sys.argv
YES = "--yes" in sys.argv          # skip the one confirmation after the first run (unattended batches)
if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__); sys.exit(0)
VALUE_FLAGS = {"--seeds", "--mode", "--condition", "--history", "--reasoning", "--reasoning-control", "--max-tokens",
               "--openrouter", "--price-in", "--price-out"}
SWITCH_FLAGS = {"--dry", "--yes", "--matched-prep", "--help", "-h"}
_rest = sys.argv[1:]
while _rest:   # a mistyped option would otherwise be ignored silently, e.g. a run without its reasoning control
    _flag = _rest.pop(0)
    if _flag in VALUE_FLAGS:
        if not _rest or _rest[0].startswith("--"):
            sys.exit(f"{_flag} needs a value (see --help)")
        _rest.pop(0)
    elif _flag not in SWITCH_FLAGS:
        sys.exit(f"unknown option {_flag!r} (see --help)")
def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
SEEDS = [int(x) for x in _arg("--seeds", ",".join(map(str, SEEDS))).split(",")]
MODE = _arg("--mode", MODE); SETUP[SETUP.index("--mode") + 1] = MODE
SCENARIO = _arg("--condition", "silent_break")    # no_change | irrelevant | silent_break | hard_removal | redirect | degradation (sto only)
if SCENARIO != "silent_break":
    SETUP += ["--condition", SCENARIO]
if SCENARIO == "no_change":     # nothing changed, so there is nothing to localize
    SETUP += ["--probes", "detection", "preservation", "adaptation"]
OPENROUTER = _arg("--openrouter", "")             # e.g. --openrouter deepseek/deepseek-chat
_price = lambda v: float(str(v).replace(",", "."))   # accept 0,0173 as typed on comma-decimal systems
PRICE_IN = _price(_arg("--price-in", PRICE_IN)); PRICE_OUT = _price(_arg("--price-out", PRICE_OUT))
MAX_TOKENS = _arg("--max-tokens", "16384")          # output cap per call, reasoning included
SETUP += ["--max-tokens", MAX_TOKENS]
HISTORY = _arg("--history", "none")                 # ablation: retained_reports_v2 | separate_reports_post_task_v1
MATCHED_PREP = "--matched-prep" in sys.argv
REASONING = _arg("--reasoning", "")                 # off | on; empty = model default, not controlled
REASONING_CONTROL = _arg("--reasoning-control", "")  # JSON override of the default control
if REASONING not in ("", "off", "on"):
    sys.exit("--reasoning must be off or on")          # ablation: preparation turn for every arm, as in ICL
SWITCHES = ["--mode-prompts", "--history", HISTORY] + (["--matched-prep"] if MATCHED_PREP else [])
RUN_DIR = os.path.join(ROOT, "runs", ("DRY_" if DRY else "") + datetime.date.today().isoformat()
                       + f"_agentic_cost_check_{MODE}_seeds{'-'.join(map(str, SEEDS))}"
                       + ("" if HISTORY == "none" else f"_{HISTORY.split('_')[0]}") + ("_prep" if MATCHED_PREP else "")
                       + ("" if SCENARIO == "silent_break" else f"_{SCENARIO}")
                       + (f"_{OPENROUTER.replace('/', '-')}" if OPENROUTER else "")
                       + (f"_reasoning-{REASONING}" if REASONING else ""))
CRED = os.path.expanduser("~/.ecpm_azure.json")

def openrouter_creds():
    k = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key: ").strip()
    return {"key": k}

def drain_typeahead():
    """Discard keys already waiting in the console, so only a fresh answer counts."""
    try:
        import msvcrt                      # Windows console
        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        try:
            import termios                 # macOS / Linux terminal
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass

def creds():
    c = json.load(open(CRED)) if os.path.exists(CRED) else {}
    c.setdefault("key", os.environ.get("AZURE_OPENAI_API_KEY") or getpass.getpass("Azure API key: ").strip())
    c.setdefault("endpoint", os.environ.get("AZURE_ENDPOINT") or input("Azure endpoint URL: ").strip())
    c.setdefault("deployment", os.environ.get("AZURE_DEPLOYMENT") or input("Deployment name [gpt-4o]: ").strip() or "gpt-4o")
    with open(CRED, "w") as fh: json.dump(c, fh)
    os.chmod(CRED, 0o600)
    return c

def queue():
    q = os.path.join(RUN_DIR, "queue.txt")
    if not os.path.exists(q):
        with open(q, "w") as fh:
            fh.write("# condition seed repeat  (add lines any time, even during the runs)\n")
            for c in CONDITIONS:
                for s in SEEDS:
                    for r in REPEATS: fh.write(f"{c} {s} {r}\n")
    items = []
    for ln in open(q):
        p = ln.split("#")[0].split()
        if len(p) == 3: items.append((p[0], int(p[1]), int(p[2])))
    return list(dict.fromkeys(items))   # a line added twice is one run, not two (progress and totals)

def tag(c, s, r): return f"s{s}_{c}_r{r}"
def _artifact(t):
    d = os.path.join(RUN_DIR, t)
    fs = [f for f in os.listdir(d) if f.startswith("pilot_") and f.endswith(".json")] if os.path.isdir(d) else []
    return os.path.join(d, fs[0]) if fs else None

def done(t): return os.path.exists(os.path.join(RUN_DIR, t, "condition.json")) and _artifact(t) is not None

def tokens(t):
    """From run_pilot's token_usage_total in the artifact (the project's one token log)."""
    u = json.load(open(_artifact(t))).get("token_usage_total") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)

def run_cost(t):
    """(USD, source) for one run: the provider's billed cost when every call reports one
    (OpenRouter does, cache discounts included), else an estimate from the list prices."""
    a = json.load(open(_artifact(t)))
    calls = a.get("provider_usage_calls") or []
    if calls and all(isinstance((u or {}).get("cost"), (int, float)) for u in calls):
        return sum(u["cost"] for u in calls), "billed"
    i, o = tokens(t)
    return i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT, "est."

def cut_off(t):
    """Calls in a run that stopped at the output cap (finish_reason "length")."""
    return sum(1 for u in json.load(open(_artifact(t))).get("provider_usage_calls") or []
               if (u or {}).get("finish_reason") == "length")

def usd(v):
    return f"${v:.4f}" if 0 < v < 0.01 else f"${v:.2f}"   # sub-cent runs (cheap models) stay visible

def log(msg):
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(os.path.join(RUN_DIR, "log.txt"), "a") as fh: fh.write(line + "\n")

def reasoning_args(cr):
    """run_pilot flags for an explicit reasoning control (validated again by run_pilot)."""
    if REASONING_CONTROL:
        control, source = json.loads(REASONING_CONTROL), "operator override via --reasoning-control"
    elif OPENROUTER:
        control = {"reasoning": {"enabled": False}} if REASONING == "off" else {"reasoning": {"effort": "medium"}}
        source = "OpenRouter unified reasoning parameter; check reasoning_tokens_total in the artifact"
    else:
        if not cr["deployment"].lower().startswith(("gpt-5", "o1", "o3", "o4")):
            sys.exit(f"{cr['deployment']} has no reasoning control; drop --reasoning or use a reasoning model")
        control = {"reasoning_effort": "none" if REASONING == "off" else "medium"}
        source = "Azure reasoning_effort; none gave 0 reasoning tokens on gpt-5.6-sol (ICL checks, 9 Oct)"
    return ["--reasoning-mode", REASONING, "--reasoning-control-json", json.dumps(control),
            "--reasoning-control-source", source]

def run_one(c, s, r, cr):
    t = tag(c, s, r)
    if DRY:
        prov = ["--provider", "dry-run"]
    elif OPENROUTER:   # OpenAI-compatible endpoint, model id like "deepseek/deepseek-chat"
        prov = ["--provider", "openai", "--base-url", OPENROUTER_URL, "--model", OPENROUTER]
    else:
        prov = ["--provider", "azure", "--model", cr["deployment"], "--azure-endpoint", cr["endpoint"]]
    if not DRY and not OPENROUTER and cr["deployment"].lower().startswith(("gpt-5", "o1", "o3", "o4")):
        prov.append("--azure-reasoning-model")   # reasoning deployments need different request fields
    if REASONING and not DRY:   # dry-run sends nothing, so no control is attached
        prov += reasoning_args(cr)
    # -u: unbuffered, so stdout.txt shows each call as it happens (progress is visible mid-run)
    cmd = [sys.executable, "-B", "-u", "agentic_conditions.py", "run", c, "--repeat", str(r)] + SWITCHES + ["--"] + SETUP + \
          ["--seed", str(s), "--tag", t, "--out", RUN_DIR] + prov
    env = dict(os.environ, AZURE_OPENAI_API_KEY=cr.get("key", ""))
    if OPENROUTER:
        env["OPENAI_API_KEY"] = cr.get("key", "")
    with open(os.path.join(RUN_DIR, "stdout.txt"), "a") as out:
        try:
            ok = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                timeout=RUN_TIMEOUT).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    return ok and done(t)

def summary():
    subprocess.run([sys.executable, "-B", "agentic_conditions.py", "exposure", RUN_DIR], cwd=ROOT, stdout=subprocess.DEVNULL)

def main():
    if not os.path.exists(os.path.join(ROOT, "agentic_conditions.py")):
        sys.exit("agentic_conditions.py is not in this folder. Run git pull (or use the latest main) first.")
    os.makedirs(RUN_DIR, exist_ok=True)
    cr = {} if DRY else (openrouter_creds() if OPENROUTER else creds())
    log(f"start {'DRY ' if DRY else ''}batch in {RUN_DIR}")
    asked = DRY or YES
    streak = 0   # consecutive failures: three in a row means a setup problem, not a bad run
    while True:
        if os.path.exists(os.path.join(RUN_DIR, "STOP")):
            log("STOP file found, stopping after the last finished run"); break
        todo = [x for x in queue() if not done(tag(*x))]
        if not todo: break
        c, s, r = todo[0]
        log(f"run {tag(c, s, r)} ({len(todo)} left)")
        if not run_one(c, s, r, cr):
            streak += 1
            if streak >= 3:
                tail = open(os.path.join(RUN_DIR, "stdout.txt"), errors="ignore").read().splitlines()[-12:]
                log("3 runs failed in a row, so this is a setup problem. Stopping. Last output:\n  " + "\n  ".join(tail))
                log("Fix it, delete this run folder, and start again.")
                return
            log(f"FAILED {tag(c, s, r)}, see stdout.txt. Moving it to the end of the queue")
            with open(os.path.join(RUN_DIR, "failures.txt"), "a") as fh: fh.write(f"{c} {s} {r}\n")
            q = os.path.join(RUN_DIR, "queue.txt"); lines = open(q).read().splitlines()
            lines = [l for l in lines if l.split("#")[0].split() != [c, str(s), str(r)]]
            if open(os.path.join(RUN_DIR, "failures.txt")).read().count(f"{c} {s} {r}\n") < 2:
                lines.append(f"{c} {s} {r}")
            open(q, "w").write("\n".join(lines) + "\n")
            continue
        streak = 0
        summary()
        fin = [x for x in queue() if done(tag(*x))]
        ti = sum(tokens(tag(*x))[0] for x in fin); to = sum(tokens(tag(*x))[1] for x in fin)
        costs = [run_cost(tag(*x)) for x in fin]
        cost = sum(v for v, _ in costs)
        kind = "billed" if all(k == "billed" for _, k in costs) else "est."
        ri, ro = tokens(tag(c, s, r))
        rc, rk = run_cost(tag(c, s, r))
        log(f"done {len(fin)}/{len(queue())}  this run: in {ri:,} out {ro:,} {usd(rc)} {rk}"
            f"  |  total: in {ti:,} out {to:,} {usd(cost)} {kind}"
            + (f"  |  CUT OFF at the output cap: {cut_off(tag(c, s, r))} call(s)" if cut_off(tag(c, s, r)) else ""))
        if not asked:
            per = cost / len(fin); total = per * len(queue())
            drain_typeahead()   # keys typed or pasted during the run must not answer this question
            ans = input(f"First run used {ti:,} input / {to:,} output tokens ({usd(per)} {kind}). "
                        f"Projected for all {len(queue())} runs: {usd(total)}. Continue? [y/N] ").strip().lower()
            asked = True
            if ans != "y": log("stopped after the first run, as asked"); break
    summary()
    shutil.make_archive(RUN_DIR, "zip", RUN_DIR)
    log(f"finished. Everything is in {RUN_DIR} and {RUN_DIR}.zip")

if __name__ == "__main__":
    main()
