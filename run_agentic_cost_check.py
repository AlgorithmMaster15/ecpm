"""Agentic cost check: real token use and cost of single runs, per arm. Stdlib only. Keeps everything local.

    python3 run_agentic_cost_check.py                  # all 5 seeds x 3 arms, silent break
    python3 run_agentic_cost_check.py --seeds 8        # quick check on one seed (3 runs)
    python3 run_agentic_cost_check.py --mode sto       # stochastic instead of deterministic
    python3 run_agentic_cost_check.py --dry            # same flow with no API calls

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
RUN_TIMEOUT = 3600                            # seconds per run
# ----------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry" in sys.argv
def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
SEEDS = [int(x) for x in _arg("--seeds", ",".join(map(str, SEEDS))).split(",")]
MODE = _arg("--mode", MODE); SETUP[SETUP.index("--mode") + 1] = MODE
HISTORY = _arg("--history", "none")                 # ablation: retained_reports_v2 | separate_reports_post_task_v1
MATCHED_PREP = "--matched-prep" in sys.argv          # ablation: preparation turn for every arm, as in ICL
SWITCHES = ["--mode-prompts", "--history", HISTORY] + (["--matched-prep"] if MATCHED_PREP else [])
RUN_DIR = os.path.join(ROOT, "runs", ("DRY_" if DRY else "") + datetime.date.today().isoformat()
                       + f"_agentic_cost_check_{MODE}_seeds{'-'.join(map(str, SEEDS))}"
                       + ("" if HISTORY == "none" else f"_{HISTORY.split('_')[0]}") + ("_prep" if MATCHED_PREP else ""))
CRED = os.path.expanduser("~/.ecpm_azure.json")

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
    return items

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

def log(msg):
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(os.path.join(RUN_DIR, "log.txt"), "a") as fh: fh.write(line + "\n")

def run_one(c, s, r, cr):
    t = tag(c, s, r)
    prov = ["--provider", "dry-run"] if DRY else ["--provider", "azure", "--model", cr["deployment"], "--azure-endpoint", cr["endpoint"]]
    if not DRY and cr["deployment"].lower().startswith(("gpt-5", "o1", "o3", "o4")):
        prov.append("--azure-reasoning-model")   # reasoning deployments need different request fields
    cmd = [sys.executable, "-B", "agentic_conditions.py", "run", c, "--repeat", str(r)] + SWITCHES + ["--"] + SETUP + \
          ["--seed", str(s), "--tag", t, "--out", RUN_DIR] + prov
    env = dict(os.environ, AZURE_OPENAI_API_KEY=cr.get("key", ""))
    with open(os.path.join(RUN_DIR, "stdout.txt"), "a") as out:
        try:
            ok = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=out, timeout=RUN_TIMEOUT).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    return ok and done(t)

def summary():
    subprocess.run([sys.executable, "-B", "agentic_conditions.py", "exposure", RUN_DIR], cwd=ROOT, stdout=subprocess.DEVNULL)

def main():
    if not os.path.exists(os.path.join(ROOT, "agentic_conditions.py")):
        sys.exit("agentic_conditions.py is not in this folder. Run git pull (or use the latest main) first.")
    os.makedirs(RUN_DIR, exist_ok=True)
    cr = {} if DRY else creds()
    log(f"start {'DRY ' if DRY else ''}batch in {RUN_DIR}")
    asked = DRY
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
        cost = ti / 1e6 * PRICE_IN + to / 1e6 * PRICE_OUT
        ri, ro = tokens(tag(c, s, r))
        log(f"done {len(fin)}/{len(queue())}  this run: in {ri:,} out {ro:,} ${ri / 1e6 * PRICE_IN + ro / 1e6 * PRICE_OUT:.2f}"
            f"  |  total: in {ti:,} out {to:,} ${cost:.2f}")
        if not asked:
            per = cost / len(fin); total = per * len(queue())
            ans = input(f"First run used {ti:,} input / {to:,} output tokens (${per:.2f}). "
                        f"Projected for all {len(queue())} runs: ${total:.0f}. Continue? [y/N] ").strip().lower()
            asked = True
            if ans != "y": log("stopped after the first run, as asked"); break
    summary()
    shutil.make_archive(RUN_DIR, "zip", RUN_DIR)
    log(f"finished. Everything is in {RUN_DIR} and {RUN_DIR}.zip")

if __name__ == "__main__":
    main()
