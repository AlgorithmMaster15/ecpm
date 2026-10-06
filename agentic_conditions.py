"""Model-first / task-only / graph-given conditions for the agentic arm.

One file, no edits to explore_agent.py or run_pilot.py: it wraps them at runtime.
task_only applies no wrapping at all, so it is the existing arm unchanged.

  model_first  Before every episode after the first, the model writes a free-text
               model of the network (wording from icl_model_first.MODEL_A/MODEL_B).
               The reply stays in the transcript and is not scored.
  graph_given  At each phase start the model receives the current specification
               (destinations and success probabilities). M1 shows the change.

Commands (repository root):
  python3 -B agentic_conditions.py test
  python3 -B agentic_conditions.py run CONDITION -- <run_pilot.py arguments>
  python3 -B agentic_conditions.py batch OUTDIR [CONDITION ...] -- <provider arguments>
  python3 -B agentic_conditions.py estimate [--seeds 51]
"""
import json, os, re, statistics as st, sys
import explore_agent as A
from resource_mdp import make_pair

CONDITIONS = ("task_only", "model_first", "graph_given")
MODEL_FIRST_FIRST = (
    "Before the next episode, construct a model of how this network behaves "
    "from what you have observed so far. Use whatever representation you find "
    "useful. Describe the current network so that your model can be used to "
    "choose actions later. Reply in free text; do not include a JSON object.")
MODEL_FIRST_UPDATE = (
    "Before the next episode, update your model of how this network behaves "
    "using everything you have observed so far. Use whatever representation "
    "you find useful. Describe the current network so that your model can be "
    "used to choose actions later. Reply in free text; do not include a JSON "
    "object.")
SEEDS_51 = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 66, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 81,
            83, 84, 85, 86, 87, 88, 89, 90, 91]

_ORIG = {"prompt": A.build_system_prompt, "policy": A._build_recording_policy, "run": A.run_explore_instance}
_STATE = {"condition": "task_only", "episodes": 0, "mdp": None, "descriptions": []}


def system_prompt(goal, condition):
    text = _ORIG["prompt"](goal)
    if condition == "graph_given":
        old = ("You do not know the network's structure or reliabilities in advance: "
               "you must learn them by trying actions and observing what happens. ")
        assert old in text, "explore_agent prompt changed; update agentic_conditions.py"
        text = text.replace(old, "At the start of each phase you receive the network's current "
                                 "specification: every location's actions with their "
                                 "destinations and success probabilities. ")
    if condition == "model_first":
        text += (" Between episodes you will also be asked to describe your "
                 "current model of the network; that reply needs no JSON object.")
    return text


def graph_text(mdp, labels):
    rows = ["Current network specification:", "location | action | destination | p_success"]
    for (u, v), p in sorted(mdp.p.items(), key=lambda e: (e[0][0], labels[e[0]])):
        rows.append(f"{u} | {labels[(u, v)]} | {v} | {p:g}")
    return "\n".join(rows)


def _policy(mdp, labels, cfg, messages, step_meta, **kw):
    """Called once per episode by run_explore_instance; adds the condition's extra input."""
    cond = _STATE["condition"]
    phase_start = mdp is not _STATE["mdp"]
    _STATE["mdp"] = mdp
    if cond == "graph_given" and phase_start:
        spec = graph_text(mdp, labels)
        kw["initial_note"] = spec if kw.get("initial_note") is None else kw["initial_note"] + "\n\n" + spec
    if cond == "model_first" and _STATE["episodes"] > 0:
        ask = MODEL_FIRST_FIRST if _STATE["episodes"] == 1 else MODEL_FIRST_UPDATE
        if messages and messages[-1]["role"] == "user":       # keep roles alternating
            messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n\n" + ask}
        else:
            messages.append({"role": "user", "content": ask})
        act = kw.get("act_fn")
        text, reasoning = act(kw["system_prompt"], messages) if act else (_STATE.get("dry_text", "(dry-run model description)"), "")
        messages.append({"role": "assistant", "content": text})
        _STATE["descriptions"].append({"before_episode": _STATE["episodes"], "text": text,
                                       "reasoning": reasoning, "scored": "not_scored_by_design"})
    _STATE["episodes"] += 1
    return _ORIG["policy"](mdp, labels, cfg, messages, step_meta, **kw)


def _run(inst, cfg, act_fn=None, node_policy_fn=None):
    _STATE.update(episodes=0, mdp=None, descriptions=[])
    res = _ORIG["run"](inst, cfg, act_fn=act_fn, node_policy_fn=node_policy_fn)
    res["descriptions"], res["condition"] = list(_STATE["descriptions"]), _STATE["condition"]
    return res


def activate(condition):
    """Switch explore_agent to `condition`. task_only restores the original functions."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    _STATE["condition"] = condition
    if condition == "task_only":
        A.build_system_prompt, A._build_recording_policy, A.run_explore_instance = \
            _ORIG["prompt"], _ORIG["policy"], _ORIG["run"]
    else:
        A.build_system_prompt = lambda goal: system_prompt(goal, condition)
        A._build_recording_policy, A.run_explore_instance = _policy, _run


# ---------------------------------------------------------------- run / batch
def run(condition, pilot_args):
    activate(condition)
    import run_pilot
    sys.argv = ["run_pilot.py"] + pilot_args
    run_pilot.main()
    out = pilot_args[pilot_args.index("--out") + 1] if "--out" in pilot_args else "pilot_artifacts"
    tag = pilot_args[pilot_args.index("--tag") + 1] if "--tag" in pilot_args else None
    if tag:   # sidecar: the condition is not in run_pilot's artifact schema
        os.makedirs(os.path.join(out, tag), exist_ok=True)
        json.dump({"condition": condition}, open(os.path.join(out, tag, "condition.json"), "w"))


def batch(outdir, conditions, provider_args):
    import subprocess
    failed = 0
    for cond in conditions:
        for s in SEEDS_51:
            tag = f"s{s}_{cond}"
            if any(f.startswith("pilot_") for f in os.listdir(os.path.join(outdir, tag))) if os.path.isdir(os.path.join(outdir, tag)) else False:
                continue   # resume
            cmd = [sys.executable, "-B", __file__, "run", cond, "--", "--pilot-type", "active", "--mode", "sto",
                   "--scenario", "seed7_silent_break", "--seed", str(s), "--tag", tag, "--out", outdir] + provider_args
            if subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode != 0:
                failed += 1; print(f"FAILED seed {s} {cond}", file=sys.stderr)
    print(f"done, {failed} failures")


# ---------------------------------------------------------------- cost estimate
def estimate(n_seeds=51, desc_tokens=250, step_out=(50, 1000), probe_out=300):
    """Offline: dry-run every seed, rebuild each call's input (system + trimmed transcript,
    as the live loop sends it) and count tokens with the harness estimate len(text)//4."""
    tok = lambda t: len(t) // 4
    wander = lambda mdp, labels: (lambda u, rng: rng.choice(sorted(v for (a, v) in mdp.p if a == u)))
    rows = {}
    for cond in CONDITIONS:
        for name, pol in (("efficient", A.dry_run_policy), ("wandering", wander)):
            ins, outs_n, outs_r = [], [], []
            for s in SEEDS_51[:n_seeds]:
                activate(cond); _STATE["dry_text"] = "x" * (4 * desc_tokens)
                inst = make_pair(s, "silent_break", matched=True)
                cfg = A.ExploreConfig(seed=s)
                res = A.run_explore_instance(inst, cfg, node_policy_fn=lambda m: pol(m, inst.labels))
                system, msgs = A.build_system_prompt(inst.m0.goal), res["messages"]
                total = steps = descs = 0
                for i, m in enumerate(msgs):
                    if m["role"] == "assistant":
                        if m["content"].startswith("xxxx"): descs += 1
                        else: steps += 1
                        total += tok(system) + sum(tok(x["content"]) for x in
                                                   A.trim_history(msgs[:i], cfg.max_context_tokens_est, cfg.keep_last_n_turns_min))
                total += 4 * (tok(system) + sum(tok(m["content"]) for m in msgs) + 150)   # four probes
                ins.append(total)
                outs_n.append(steps * step_out[0] + descs * desc_tokens + 4 * probe_out)
                outs_r.append(steps * step_out[1] + descs * (desc_tokens + step_out[1]) + 4 * (probe_out + step_out[1]))
            rows[(cond, name)] = tuple(st.mean(x) * n_seeds / 1e6 for x in (ins, outs_n, outs_r))
    activate("task_only"); _STATE.pop("dry_text", None)
    print(f"per model, {n_seeds} seeds; output assumes {step_out[0]} / {step_out[1]} tokens per step (no reasoning / reasoning)")
    print(f"{'condition':12} {'behaviour':10} {'input M':>8} {'out M':>7} {'out M reas.':>11} {'gpt-4o $':>9}")
    for (c, b), (i, on, orr) in rows.items():
        print(f"{c:12} {b:10} {i:8.1f} {on:7.2f} {orr:11.1f} {i * 2.5 + on * 10:9.0f}")
    return rows


# ---------------------------------------------------------------- tests
def test():
    import unittest
    golden = _ORIG["prompt"]("C")
    def scripted(log):
        def act(system, messages):
            last = messages[-1]["content"]
            log.append("describe" if "construct a model" in last or "update your model" in last else "step")
            if log[-1] == "describe":
                return "A has two actions; the route via B looks reliable.", ""
            legal = re.findall(r"Legal actions here: ([a0-9, ]+)", last)
            return '{"action": "%s"}' % (legal[-1].split(",")[0].strip() if legal else "a1"), ""
        return act
    def go(cond):
        activate(cond); log = []
        inst = make_pair(33, "silent_break", matched=True)
        cfg = A.ExploreConfig(max_episodes_m0=2, max_episodes_m1=2, max_steps_per_episode=6, seed=33)
        return inst, A.run_explore_instance(inst, cfg, act_fn=scripted(log)), log
    class T(unittest.TestCase):
        def test_task_only_untouched(self):
            activate("task_only")
            self.assertIs(A.build_system_prompt, _ORIG["prompt"])
            self.assertIs(A.run_explore_instance, _ORIG["run"])
        def test_unknown_rejected(self):
            with self.assertRaises(ValueError): activate("graph_first")
        def test_model_first(self):
            inst, res, log = go("model_first")
            self.assertEqual(len(res["descriptions"]), 3); self.assertEqual(log.count("describe"), 3)
            roles = [m["role"] for m in res["messages"]]
            self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])))
            self.assertNotEqual(A.build_system_prompt("C"), golden)
        def test_graph_given(self):
            inst, res, log = go("graph_given")
            specs = [m["content"] for m in res["messages"] if "Current network specification" in m["content"]]
            self.assertEqual(len(specs), 2)
            u, v = inst.change["edge"]
            self.assertIn(f"{u} | {inst.labels[(u, v)]} | {v} | 0", specs[1])
            self.assertEqual(res["descriptions"], [])
        def tearDown(self): activate("task_only")
    r = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    sys.exit(0 if r.wasSuccessful() else 1)


if __name__ == "__main__":
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    head = argv[:argv.index("--")] if "--" in argv else argv
    if not head or head[0] not in ("run", "batch", "estimate", "test"):
        print(__doc__); sys.exit(2)
    if head[0] == "test": test()
    elif head[0] == "run": run(head[1], rest)
    elif head[0] == "batch": batch(head[1], head[2:] or list(CONDITIONS), rest)
    else:
        n = int(head[head.index("--seeds") + 1]) if "--seeds" in head else 51
        estimate(n)
