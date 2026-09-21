"""Offline contract, fault-injection and transport tests. Never calls a model."""

import copy
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import icl_graph as graph
import run_pilot as pilot
from test_run_pilot import args as old_args, scenario


def args(profile="gemma_e4b", condition="graph_ab", provider="dry-run"):
    a = old_args()
    a.protocol = graph.PROTOCOL
    a.graph_condition, a.request_profile = condition, profile
    a.mode, a.max_tokens, a.provider = "det", 8192, provider
    a.model = graph.MODELS[profile]
    a.deployment_config = None
    return a


def config(profile="gemma_e4b", mode="off", seed_supported=True):
    """Synthetic evidence for mocks only, never a deployment attestation."""
    if profile == "sol":
        seed_supported = False
    c = {"profile": profile, "model_id": graph.MODELS[profile], "model": graph.MODELS[profile],
         "response_model": graph.MODELS[profile], "model_source": "synthetic test fixture",
         "endpoint": "http://localhost:1234/v1", "api_version": "openai-compatible-v1",
         "supported_request_fields": list(graph.intended_settings(profile, mode, 0, seed_supported)),
         "seed_supported": seed_supported,
         "effective": {"source": "synthetic test fixture", "runtime": "mock",
                       "reasoning_mode": mode, "template_sha256": "a" * 64,
                       "max_output_tokens": 8192, "history_truncation": False,
                       "thinking": mode == "on", "reasoning_budget": None,
                       "response_length_limit": False, "temperature": 1.0,
                       "top_p": .95, "top_k": 64, "repeat_penalty": 1.0,
                       "min_p_supported": True, "min_p": .05,
                       "quantization": "Q6_K", "eval_batch_size": 512,
                       "physical_batch_size": 256, "parallel": 1,
                       "flash_attention": True, "offload_kv_cache_to_gpu": True,
                       "speculative_decoding": False},
         "context": {"method": "utf8_upper_bound", "tokens": 32768,
                     "overhead_tokens_per_message": 256, "replay_expansion_bound": 1,
                     "source": "synthetic bound for test only"},
         "pricing": {"input_per_million": None, "output_per_million": None}}
    if profile == "sol":
        c["effective"].update(template_sha256=None, template_source="provider-managed; unavailable")
    if profile == "gemma_31b_together":
        c["endpoint"] = "https://api.together.ai/v1"
    if profile in graph.HOSTED_PROFILES:
        c['effective'].update(
            hosted_evidence={'policy':graph.HOSTED_READINESS_POLICY, 'scope':'exact_model',
                'model':c['model'], 'endpoint':c['endpoint'], 'date':'2026-09-17',
                'source':'synthetic exact-model support fixture, NOT real provider evidence',
                'effect':'disables_reasoning_computation' if mode=='off' else 'enables_reasoning',
                'request_fields':{k:v for k,v in graph.intended_settings(profile,mode,0).items()
                                  if k in ('reasoning','reasoning_effort')},
                'reasoning_tokens_source':'synthetic documented reasoning-token field'},
            output_context_source='synthetic output/context guarantee',
            no_reasoning_budget_source='synthetic no separate reasoning-budget guarantee',
            app_settings_inapplicable={'reasoning_budget':'No hosted Bionic toggle'})
    c["preflight"] = {"source": "synthetic test fixture", "http_status": 200,
                      "network_retries": [], "request": {"model": c["model"],
                          "messages": [{"role": "user", "content": graph.PREFLIGHT}],
                          **graph.intended_settings(profile, mode, 999, seed_supported)},
                      "response": envelope("301", profile, mode)}
    return c


def envelope(text, profile="gemma_e4b", mode="off", finish="stop"):
    return {"model": graph.MODELS[profile], "choices": [{"finish_reason": finish,
            "message": {"role": "assistant", "content": text,
                        "reasoning_content": "" if mode == "off" else "Recorded reasoning."}}],
            "usage": {"prompt_tokens": 5000, "completion_tokens": 800,
                      "completion_tokens_details": {"reasoning_tokens": 0 if mode == "off" else 50}}}


def local_config():
    """Synthetic tokenizer evidence for offline tests only."""
    c = config()
    identity = {k: "a" * 64 for k in ("backend_sha256", "model_sha256", "template_sha256",
        "frontend_sha256", "app_settings_sha256", "backend_settings_sha256", "loaded_instances_sha256")}
    identity["build_info"] = "synthetic"
    c["context"] = {"method": graph.LOCAL_CONTEXT, "tokens": 32768,
        "backend_url": "http://127.0.0.1:12345", "credential_env": "ECPM_LOCAL_BACKEND_API_KEY",
        "identity": identity, "source": "synthetic offline test only",
        "counting_settings": {"add_special": True, "parse_special": True,
            "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}}
    c["context"]["path_verification"] = {"source": "synthetic path verification",
        "preflight_response_sha256": pilot._canonical_sha256(c["preflight"]["response"]),
        "preflight_count": graph.local_count_record(c["preflight"]["request"], c, "synthetic",
                                                     [2] * 5000, identity)}
    return c


class Reply:
    status = 200

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def read(self):
        return json.dumps(self.data).encode()


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.sc = scenario()
        self.record, self.view, self.views, self.prompts = graph.prepare(self.sc)
        self.queried = pilot.queried_pairs_for_icl(self.record, self.sc)

    def test_prompts_and_fixed_evidence(self):
        expected_hashes = ("d0d3cae43bf0a1174fc7edeb73c8f1da8401dcc964da462ab0cfff82e31efc05",
                           "65479336f316fb30f81fa1e49c5316c0a24ef3181b5f1e7157627a137dff7d12")
        for i in (0, 1):
            non_graph = []
            for condition in graph.CONDITIONS:
                v, text = self.views[condition][i], self.prompts[condition][i]
                self.assertEqual(len(v["rows"]), 160)
                self.assertEqual(pilot.sha256_text("\n".join(v["rows"]) + "\n"), expected_hashes[i])
                self.assertEqual(text.count(graph.COMMON), 1)
                self.assertTrue(text.endswith(graph.ENDING + "\n"))
                self.assertIn(graph.PLACEHOLDERS, text)
                for q in self.queried:
                    self.assertEqual(text.count(f'{{"node":"{q["node"]}","action":"{q["action"]}","available":<BOOLEAN>'), 1)
                for word in ("graph_ab", "graph_a", "logs_only", "oracle", "intervention", "target", "controls", "seed"):
                    self.assertNotIn(word, text)
                non_graph.append(text.replace(graph.graph_block(v), "", 1) if v["graph"] is not None else text)
            self.assertEqual(len(set(non_graph)), 1)
        self.assertEqual(self.prompts["graph_ab"][0], self.prompts["graph_a"][0])
        self.assertEqual(self.prompts["graph_a"][1], self.prompts["logs_only"][1])
        self.assertNotIn("Period B", self.prompts["graph_ab"][0])
        self.assertIn("G | a1 | true | E | 0.0", self.prompts["graph_ab"][1])
        self.assertEqual(pilot.first_deterministic_gate_seed()["seed"], 8)
        self.assertFalse(pilot.deterministic_gate(1)["eligible"])

    def test_isolation_and_injected_view_defects(self):
        record = copy.deepcopy(self.record)
        record["world_post"] = {"oracle": "LEAK"}
        self.assertEqual(graph.build_prompt(graph.authorized_view(record, self.view, "pre", True, self.queried)), self.prompts["graph_ab"][0])
        for mutate in (lambda v: v.update(oracle="LEAK"),
                       lambda v: v["graph"][0].update(target=True),
                       lambda v: v.update(rows=[]),
                       lambda v: v["rows"].__setitem__(0, "[ORACLE, a1, E]"),
                       lambda v: v["graph"][0].update(p_success=.5),
                       lambda v: v["graph"].pop()):
            broken = copy.deepcopy(self.views["graph_ab"][0]); mutate(broken)
            with self.assertRaises(ValueError):
                graph.build_prompt(broken)
        with self.assertRaisesRegex(ValueError, "requires earlier"):
            graph.reference_answer(self.views["logs_only"][1])
        with self.assertRaises(ValueError):
            graph.prepare({**self.sc, "budget": 0})

    def test_sequential_references_same_parser_and_scores(self):
        for condition in graph.CONDITIONS:
            earlier = pre_score = None
            for i, v in enumerate(self.views[condition]):
                raw, earlier = graph.reference_answer(v, earlier)
                parsed = (pilot.parse_icl_turn_a if i == 0 else pilot.parse_icl_turn_b)(raw)
                visible = pilot.visible_transition_stats(v["rows"], v["menu"])
                scored = pilot.score_icl_turn(self.record, parsed, self.queried,
                                             "pre" if i == 0 else "post", ("G", "a1"), visible, pre_score)
                self.assertTrue(scored["correct"])
                self.assertEqual(scored["beliefs"]["p_mae_visible"], 0)
                self.assertEqual(scored["route"]["regret"], 0)
                self.assertEqual(scored["route"]["expected_cost"], 2 if i == 0 else 3)
                pre_score = scored["beliefs"]
                if i:
                    target = next(p for p in parsed["beliefs"]["pairs"] if p["node"] == "G")
                    self.assertEqual((target["available"], target["destination"], target["p_success"]), (True, "E", 0))
                    self.assertTrue(scored["control_preservation"]["all_four_controls_correct"])
            broken = json.loads(raw)
            broken["route"] = [{"node": "G", "action": "a1"}, {"node": "E", "action": "a2"}]
            parsed = pilot.parse_icl_turn_b(json.dumps(broken))
            scored = pilot.score_icl_turn(self.record, parsed, self.queried, "post", ("G", "a1"), visible, pre_score)
            self.assertFalse(scored["correct"])
            self.assertFalse(scored["route"].get("is_optimal"))

    def test_old_prompt_snapshots(self):
        expected = [
            "0869266a1503dea1ffcc6a6971e1b19a3d089028d8eaef0a2c66928d2bb6cb87",
            "3956374ebbe29520c0aa1122301955a15041b494fafffb78b9314bd93104df59",
            "e05f58a5f2e58b5198b032462dc5f10e05a922a43d89d4443322bbc6089fb346",
            "134bcee56435dc9833c5b01adbbff1070971209a41c23b57bf3155fbf94b33cf",
            "64a0a2bae2b5235effcd58d0c50ca8a5c7fae8b4fa850754b40f4b3c7a20bd30",
            "9ff49bc011972c1955892e4594e87726cec69ec21c0059a259a930a4e71554fd"]
        self.assertEqual([pilot.sha256_text(pilot.build_icl_prompt(self.view, level, period, self.queried))
                          for level in pilot.ICL_LEVELS for period in ("pre", "post")], expected)

    def test_seed8_complete_prompt_snapshots(self):
        expected = {
            'graph_ab': ['1a6a522b5575dfa3aff48a1e0ded174eaf2bca06c793ef59445a41105ddbda09',
                         '30047c971df6ceebe572afd06c991028a8628491cb7670d7b4279cce0006629f'],
            'graph_a': ['1a6a522b5575dfa3aff48a1e0ded174eaf2bca06c793ef59445a41105ddbda09',
                        'fa9240391fe746453b3076e5678ab52801ec8c7ecd44ff8699a9f54e45e15793'],
            'logs_only': ['1fca7f391cf71d28e7078f2125bcd3ec054797e4bc6f9992264b3da182589320',
                          'fa9240391fe746453b3076e5678ab52801ec8c7ecd44ff8699a9f54e45e15793']}
        self.assertEqual({c: [pilot.sha256_text(p) for p in ps] for c, ps in self.prompts.items()}, expected)

    def test_existing_eligibility_search_and_guards(self):
        from experiments.preview_icl_graph import eligibility_report
        report = eligibility_report()
        self.assertEqual(report['first_three'], [8, 13, 25])
        self.assertEqual(report['n_examined'], 1000)
        self.assertIn('pre_optimum_unique', report['checks_through_seed25'][0]['reasons'])
        with self.assertRaisesRegex(ValueError, 'supports seeds'):
            graph.prepare({**self.sc, 'seed': 1})
        with patch('run_pilot.deterministic_gate', return_value={'eligible': False}):
            with self.assertRaisesRegex(ValueError, 'fails deterministic eligibility'):
                graph.prepare(self.sc)
            with self.assertRaisesRegex(ValueError, 'selection drifted'):
                eligibility_report()

    def test_new_worlds_prompts_references_and_dry_runs(self):
        from experiments.preview_icl_graph import generate
        for seed in (13, 25):
            sc = {**self.sc, **pilot.SCENARIOS[f'icl_det_gate_seed{seed}'], 'name': f'icl_det_gate_seed{seed}'}
            record, view, variants, prompts = graph.prepare(sc)
            queried = pilot.queried_pairs_for_icl(record, sc)
            target = pilot.protocol_target_pair(record, sc)
            self.assertEqual(len(queried), 5)
            self.assertIn({'node': target[0], 'action': target[1]}, queried)
            self.assertNotEqual(queried, self.queried)
            if seed == 25:
                self.assertIn('a4', variants['graph_ab'][0]['menu']['C'])
            for period in (0, 1):
                self.assertEqual(len({prompts[c][period].replace(graph.graph_block(variants[c][period]), '', 1)
                    if variants[c][period]['graph'] is not None else prompts[c][period] for c in graph.CONDITIONS}), 1)
            poisoned = copy.deepcopy(record)
            poisoned['world_post'] = {'oracle': 'LEAK'}
            for c in graph.CONDITIONS:
                self.assertEqual(graph.build_prompt(graph.authorized_view(poisoned, view, 'pre', c != 'logs_only', queried)), prompts[c][0])
                self.assertNotIn('Period B', prompts[c][0])
                self.assertIn(f"Start: {view['start']}   Goal: {view['goal']}", prompts[c][0])
                for text in prompts[c]:
                    for word in ('oracle', 'evaluator', 'target', 'intervention', 'controls'):
                        self.assertNotIn(word, text)
            broken = copy.deepcopy(variants['graph_ab'][0])
            broken['rows'][0] = '[C, a99, A]'
            with self.assertRaisesRegex(ValueError, 'not in current menu'):
                graph.build_prompt(broken)
            broken = copy.deepcopy(variants['graph_ab'][0])
            broken['queried_pairs'][0]['target'] = True
            with self.assertRaisesRegex(ValueError, 'without role labels'):
                graph.build_prompt(broken)
            broken = copy.deepcopy(variants['graph_ab'][0])
            broken['queried_pairs'][1] = broken['queried_pairs'][0]
            with self.assertRaisesRegex(ValueError, 'distinct and sorted'):
                graph.build_prompt(broken)
            with tempfile.TemporaryDirectory() as directory:
                report = generate(Path(directory) / 'prompts', seed=seed)
                self.assertEqual(report['graph_seed'], seed)
                for condition in graph.CONDITIONS:
                    with patch('icl_graph.urllib.request.OpenerDirector.open', side_effect=AssertionError('offline only')):
                        outputs = graph.run_suite(sc, args(condition=condition), Path(directory) / condition)
                    self.assertEqual(len(outputs), 3)
                    for repeat, output in enumerate(outputs, 1):
                        a = json.loads(Path(output['path']).read_text())
                        self.assertEqual(a['scenario']['seed'], seed)
                        self.assertEqual(a['identity']['scenario']['seed'], seed)
                        self.assertEqual(a['queried_pairs'], queried)
                        self.assertEqual(a['model']['sampling_seed'], repeat - 1)
                        self.assertTrue(all(graph.audit_artifact(a).values()))
                        for name in ('A', 'B'):
                            t, ref = a['turns'][name], report['references'][condition][name]
                            self.assertEqual(t['scored'], ref['scored'])
                            self.assertTrue(t['scored']['correct'])
                            self.assertEqual(t['scored']['beliefs']['accuracy'], 1)
                            self.assertEqual(t['scored']['route']['regret'], 0)
                        b = a['turns']['B']['scored']
                        self.assertTrue(b['detection_localization']['detection_correct'])
                        self.assertTrue(b['detection_localization']['localization_correct'])
                        self.assertEqual(b['control_preservation']['mean_control_preservation'], 1)
                        self.assertTrue(b['control_preservation']['all_four_controls_correct'])

    def test_on_refuses_another_graphs_off_reference(self):
        for seed, other in ((13, 25), (25, 13)):
            sc = {**self.sc, 'seed': seed}
            record, view, variants, prompts = graph.prepare(sc)
            foreign_prompts = graph.prepare({**self.sc, 'seed': other})[3]
            for condition in graph.CONDITIONS:
                with tempfile.TemporaryDirectory() as directory:
                    for repeat in (1, 2, 3):
                        a = args(condition=condition, provider='openai')
                        ext = graph.GraphRun(a, variants[condition], prompts[condition], repeat - 1, config(), prompts)
                        values = [envelope(ext.dry_answers[t]) for t in ('A', 'B')]
                        with patch('icl_graph.urllib.request.OpenerDirector.open', side_effect=[Reply(v) for v in values]):
                            artifact, path, _ = pilot.run_icl_two_response_once(record, view, sc, True, a,
                                condition, repeat, repeat - 1, directory, ext)
                    artifact['turns']['A']['scored']['correct'] = False
                    pilot._write_json_atomic(path, artifact)
                    self.assertEqual(graph.matched_off(directory, config(mode='on'), condition, prompts[condition]), [1, 2, 3])
                    with self.assertRaisesRegex(ValueError, 'must match OFF'):
                        graph.matched_off(directory, config(mode='on'), condition, foreign_prompts[condition])

    def test_request_profiles_and_controls(self):
        for profile in graph.MODELS:
            for mode in ("off", "on"):
                c = config(profile, mode)
                graph.validate_deployment(c, profile, mode)
                settings = graph.intended_settings(profile, mode, 0, c["seed_supported"])
                self.assertNotIn("reasoning_budget", settings)
                if profile == "sol":
                    self.assertEqual(set(settings), {"reasoning_effort", "max_completion_tokens"})
                else:
                    control = {"enabled": mode == 'on'} if profile == 'gemma_31b_together' else mode
                    self.assertEqual(settings, {"reasoning": control, "max_tokens": 8192,
                                              "temperature": 1.0, "top_p": .95, "top_k": 64, "seed": 0})
                    self.assertNotIn("seed", graph.intended_settings(profile, mode, 0, False))
                for mutation in (lambda x: x.update(model="wrong"),
                                 lambda x: x.update(supported_request_fields=[]),
                                 lambda x: x["effective"].update(max_output_tokens=2048),
                                 lambda x: x["preflight"].update(network_retries=[1]),
                                 lambda x: x["preflight"]["request"].update(tools=[]),
                                 lambda x: x.update(api_key="secret")):
                    d = copy.deepcopy(c); mutation(d)
                    with self.assertRaises(ValueError):
                        graph.validate_deployment(d, profile, mode)
        d = envelope("answer")
        del d["usage"]["completion_tokens_details"]
        self.assertFalse(graph.reasoning_check(d, "off", {})["mode_verified"])
        d["choices"][0]["message"]["reasoning_content"] = "<think></think>"
        self.assertFalse(graph.reasoning_check(d, "on", {})["mode_verified"])
        d["choices"][0]["message"]["reasoning_content"] = "<|channel>thought\n<channel|>"
        self.assertFalse(graph.reasoning_check(d, "on", {})["mode_verified"])
        self.assertTrue(graph.reasoning_check(d, "off", {"off_empty_channel_source": "documented",
                                                         "off_template_verified": True})["mode_verified"])
        self.assertTrue(graph.reasoning_check(envelope("x", mode="on"), "off", {})["control_violation"])

    def test_context_and_identity(self):
        a = args(); c = config()
        ext = graph.GraphRun(a, self.views['graph_ab'], self.prompts['graph_ab'], 0, c, self.prompts)
        prior = copy.deepcopy(ext.identity_fields)
        c['context']['tokens'] += 1
        other = graph.GraphRun(a, self.views['graph_ab'], self.prompts['graph_ab'], 0, c, self.prompts)
        self.assertNotEqual(prior, other.identity_fields)
        messages = [{"role": "user", "content": p} for p in self.prompts['graph_ab']]
        self.assertTrue(graph.context_bound(messages, c, reserve_answer=8192)['fits'])
        messages.append({"role": "assistant", "content": "x" * 40000})
        self.assertFalse(graph.context_bound(messages, c)['fits'])
        with self.assertRaises(ValueError):
            graph.context_bound([], {"context": {"method": "guess"}})

    def test_local_count_boundary_and_exact_metadata_requests(self):
        c = local_config()
        graph.validate_deployment(c, 'gemma_e4b', 'off')
        body = c['preflight']['request']
        for n, fits in ((24576, True), (24577, False)):
            calls = []
            def metadata(url, data=None, key=None):
                calls.append((url, data))
                return {'prompt':'rendered with generation prefix'} if url.endswith('/apply-template') else {'tokens':[2]*n}
            with patch.dict(os.environ, {'ECPM_LOCAL_BACKEND_API_KEY':'synthetic'}), \
                    patch.object(graph, '_local_snapshot', return_value=c['context']['identity']), \
                    patch.object(graph, '_metadata', side_effect=metadata):
                result = graph.count_local_request(body, c)
            self.assertEqual(result['fits'], fits)
            self.assertEqual(result['input_tokens'], n)
            self.assertEqual(calls[0][1], {'messages':body['messages'], 'add_generation_prompt':True,
                'chat_template_kwargs':{'enable_thinking':False}})
            self.assertEqual(calls[1][1], {'content':'rendered with generation prefix',
                'add_special':True, 'parse_special':True})
            self.assertEqual(graph.audit_local_count(result, body, c), fits)
        c['context']['counting_settings']['add_special'] = False
        with self.assertRaisesRegex(ValueError, 'settings mismatch'):
            graph.validate_local_context(c)

    def test_local_actual_history_and_offline_audit(self):
        c = local_config(); counted = []
        malformed = '  prose before {bad JSON}\n\tand after  '
        def count(body, config):
            counted.append(copy.deepcopy(body))
            return graph.local_count_record(body, config, 'synthetic full rendered request',
                                             [2]*5000, config['context']['identity'])
        with tempfile.TemporaryDirectory() as directory, patch.object(graph, 'count_local_request', side_effect=count):
            art, path, bodies = self.run_mock(directory, [envelope(malformed), envelope('malformed B')], deployment=c)
        self.assertEqual(counted, bodies)
        self.assertEqual(counted[1]['messages'][1]['content'], malformed)
        self.assertEqual(art['turns']['A']['raw_response'], malformed)
        self.assertTrue(all(graph.audit_artifact(json.loads(json.dumps(art))).values()))
        for key, value in (('input_tokens',1), ('request_sha256','bad'), ('rendered_prompt','changed')):
            damaged = json.loads(json.dumps(art))
            damaged['turns']['B']['context_check'][key] = value
            self.assertFalse(all(graph.audit_artifact(damaged).values()), key)
        with patch.object(graph, 'context_bound', side_effect=AssertionError('no unseen-answer estimate')):
            graph.context_planning(c, self.prompts.values())

    def test_local_oversized_B_preserves_A_without_B_call(self):
        c = local_config(); counts = iter((5000, 24577))
        def count(body, config):
            return graph.local_count_record(body, config, 'synthetic', [2]*next(counts), config['context']['identity'])
        with tempfile.TemporaryDirectory() as directory, patch.object(graph, 'count_local_request', side_effect=count):
            with self.assertRaisesRegex(RuntimeError, 'history does not fit'):
                self.run_mock(directory, [envelope('complete malformed A')], deployment=c)
            saved = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(len(self.last_bodies), 1)
            self.assertEqual(saved['state'], 'incomplete')
            self.assertEqual(saved['turns']['A']['raw_response'], 'complete malformed A')
            self.assertNotIn('raw_response', saved['turns']['B'])
            self.assertNotIn('metrics', saved)
            self.assertFalse(all(graph.audit_artifact(saved).values()))
            path = next(Path(directory).glob('*.json'))
            summary, _ = pilot.write_icl_summary(directory, [{'path':str(path)}], graph.PROTOCOL,
                                                 ('graph_ab',), graph.audit_artifact)
            self.assertFalse(summary['operational_gate_pass'])
            self.assertEqual(summary['completed_runs'], 0)

    def test_local_count_failure_and_identity_change_stop_before_generation(self):
        c = local_config()
        cases = [(TimeoutError('synthetic tokenizer failure'), None),
                 (None, {'build_info':'wrong'}),
                 (None, [c['context']['identity'], {'build_info':'changed during count'}])]
        for failure, identity in cases:
            with self.subTest(failure=type(failure).__name__, identity=identity), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {'ECPM_LOCAL_BACKEND_API_KEY':'synthetic'}), \
                    patch.object(graph, '_local_snapshot', **({'side_effect':identity} if isinstance(identity,list)
                        else {'return_value':identity or c['context']['identity']})), \
                    patch.object(graph, '_metadata', side_effect=failure or [{'prompt':'x'},{'tokens':[2]}]):
                with self.assertRaisesRegex(RuntimeError, 'context counting failed'):
                    self.run_mock(directory, [], deployment=c)
                self.assertEqual(self.last_bodies, [])
                saved = json.loads(next(Path(directory).glob('*.json')).read_text())
                self.assertEqual(saved['failure']['type'], 'ContextCounting')
                self.assertEqual(saved['state'], 'incomplete')

    def test_local_usage_discrepancy_saved_and_fails(self):
        c = local_config()
        def count(body, config):
            return graph.local_count_record(body, config, 'synthetic', [2]*4999, config['context']['identity'])
        with tempfile.TemporaryDirectory() as directory, patch.object(graph, 'count_local_request', side_effect=count):
            with self.assertRaisesRegex(RuntimeError, 'operational failure'):
                self.run_mock(directory, [envelope('nonblank')], deployment=c)
            saved = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(saved['turns']['A']['prompt_usage_check']['difference'], 1)
            self.assertEqual(saved['turns']['A']['raw_response'], 'nonblank')
            self.assertEqual(len(self.last_bodies), 1)
            self.assertFalse(all(graph.audit_artifact(saved).values()))

    def test_local_readiness_and_credential_missing_are_not_success(self):
        c = local_config()
        with patch.dict(os.environ, {}, clear=True), patch.object(graph, '_metadata') as metadata:
            with self.assertRaisesRegex(ValueError, 'credential not configured'):
                graph.count_local_request(c['preflight']['request'], c)
            metadata.assert_not_called()
        c['context']['path_verification']['preflight_count'] = None
        with self.assertRaisesRegex(ValueError, 'counting/generation path not verified'):
            graph.validate_deployment(c, 'gemma_e4b', 'off')

    def run_mock(self, directory, responses, malformed_a=False, profile="gemma_e4b", deployment=None):
        a = args(profile, provider="openai"); c = deployment or config(profile)
        ext = graph.GraphRun(a, self.views['graph_ab'], self.prompts['graph_ab'], 0, c, self.prompts)
        actual_bodies = []
        self.last_bodies = actual_bodies
        def opened(request, **unused):
            actual_bodies.append(json.loads(request.data))
            data = responses[len(actual_bodies) - 1]
            if isinstance(data, Exception):
                raise data
            return Reply(data)
        with patch('icl_graph.urllib.request.OpenerDirector.open', side_effect=opened), \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-test-key'}):
            art, path, _ = pilot.run_icl_two_response_once(self.record, self.view, self.sc, True, a,
                                                         'graph_ab', 1, 0, directory, ext)
        return art, path, actual_bodies

    def test_exact_requests_raw_first_and_malformed_history(self):
        raw_b, _ = graph.reference_answer(self.views['graph_ab'][1],
                                         graph.reference_answer(self.views['graph_ab'][0])[1])
        for profile in graph.MODELS:
            with tempfile.TemporaryDirectory() as directory:
                art, path, bodies = self.run_mock(directory, [envelope('malformed A text', profile),
                                                             envelope(raw_b, profile)], profile=profile)
                self.assertEqual(len(bodies), 2)
                self.assertEqual(bodies[1]['messages'][1], {'role': 'assistant', 'content': 'malformed A text'})
                self.assertEqual(bodies[0], {"model": graph.MODELS[profile],
                    "messages": [{"role": "user", "content": self.prompts['graph_ab'][0]}],
                    **graph.intended_settings(profile, 'off', 0, profile != 'sol')})
                self.assertTrue(all(graph.audit_artifact(art).values()))
                self.assertFalse(art['turns']['A']['scored']['correct'])
                self.assertIn('provider_response_raw', art['turns']['A'])
                self.assertEqual(json.loads(Path(path).read_text()), art)
                for mutate in (lambda x: x['turns']['A'].update(response_sha256='bad'),
                               lambda x: x.update(run_id='bad'),
                               lambda x: x.update(repeat=4),
                               lambda x: x['turns']['B'].update(previous_response_sha256='bad'),
                               lambda x: x['turns']['B'].update(prompt='corrupted'),
                               lambda x: x['turns']['B'].update(truncated=True),
                               lambda x: x['turns']['B']['request_body'].update(reasoning_budget=1),
                               lambda x: x['graph_views'][1][0].update(p_success=0)):
                    d = copy.deepcopy(art); mutate(d)
                    self.assertFalse(all(graph.audit_artifact(d).values()))

    def test_actual_raw_persistence_precedes_answer_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = pilot.parse_icl_turn_a
            def inspect_before_parse(raw):
                saved = json.loads(next(Path(directory).glob('*.json')).read_text())
                self.assertEqual(saved['turns']['A']['raw_response'], raw)
                self.assertIn('provider_response_raw', saved['turns']['A'])
                self.assertNotIn('parsed', saved['turns']['A'])
                return parser(raw)
            with patch('run_pilot.parse_icl_turn_a', side_effect=inspect_before_parse):
                self.run_mock(directory, [envelope('malformed A'), envelope('malformed B')])

    def test_preflight_requires_nonblank_string(self):
        for content in ('', ' ', '\n', '\t', ' \n\t ', None, 1, True, [], {}):
            with self.subTest(content=content):
                c = config()
                c['preflight']['response']['choices'][0]['message']['content'] = content
                with self.assertRaisesRegex(ValueError, 'preflight control not verified'):
                    graph.validate_deployment(c, 'gemma_e4b', 'off')
        c = config()
        c['preflight']['response']['choices'][0]['message']['content'] = ' \nnot JSON\t '
        graph.validate_deployment(c, 'gemma_e4b', 'off')
        self.assertEqual(c['preflight']['response']['choices'][0]['message']['content'], ' \nnot JSON\t ')

    def test_together_requires_exact_endpoint_and_model(self):
        for mutate in (lambda c: c.update(endpoint='https://other.example/v1'),
                       lambda c: c.update(model='pearl-ai/gemma-4-31b-it'),
                       lambda c: c.update(response_model='different'),
                       lambda c: c['effective'].update(template_sha256=None),
                       lambda c: c['preflight']['request'].update(reasoning='off')):
            c = config('gemma_31b_together'); mutate(c)
            with self.assertRaises(ValueError):
                graph.validate_deployment(c, 'gemma_31b_together', 'off')

    def test_together_reasoning_aliases_and_unknown_counts(self):
        profile = 'gemma_31b_together'
        c = config(profile)
        for field in ('reasoning', 'reasoning_content'):
            response = envelope('answer', profile)
            message = response['choices'][0]['message']
            message.pop('reasoning_content')
            message[field] = ''
            self.assertTrue(graph.reasoning_check(response, 'off', c['effective'], profile, c)['mode_verified'])
            del response['usage']['completion_tokens_details']
            control = graph.reasoning_check(response, 'off', {}, profile)
            self.assertIsNone(control['reasoning_tokens'])
            self.assertEqual(control['evidence'], 'unknown')
            self.assertFalse(control['mode_verified'])
            message[field] = 'Actual thought text'
            on = config(profile, mode='on')
            self.assertTrue(graph.reasoning_check(response, 'on', on['effective'], profile, on)['mode_verified'])
            self.assertTrue(graph.reasoning_check(response, 'off', c['effective'], profile, c)['control_violation'])
        response = envelope('answer', profile)
        response['choices'][0]['message']['reasoning'] = 'hidden by empty alias'
        with self.assertRaisesRegex(ValueError, 'conflicting Together'):
            graph.provider_reasoning(response, profile)
        # Other profiles retain their original reasoning_content-only behavior.
        self.assertEqual(graph.provider_reasoning(response, 'gemma_e4b'), '')
        self.assertEqual(graph.provider_reasoning(response, 'sol'), '')

    def test_together_request_and_serialized_raw_audit(self):
        profile = 'gemma_31b_together'
        responses = [envelope('nonempty malformed answer', profile) for _ in range(2)]
        for response in responses:
            response['choices'][0]['message']['reasoning'] = response['choices'][0]['message'].pop('reasoning_content')
        with tempfile.TemporaryDirectory() as directory:
            _, path, bodies = self.run_mock(directory, responses, profile=profile)
            saved_text = Path(path).read_text()
        saved = json.loads(saved_text)
        self.assertTrue(all(graph.audit_artifact(saved).values()))
        for body in bodies:
            self.assertEqual(body['reasoning'], {'enabled': False})
            self.assertEqual(body['max_tokens'], 8192)
            self.assertEqual(body['seed'], 0)
            self.assertNotIn('reasoning_budget', body)
            self.assertNotIn('tools', body)
            self.assertNotIn('response_format', body)
        self.assertEqual(graph.intended_settings(profile, 'on', 0, True)['reasoning'], {'enabled': True})
        self.assertNotIn('seed', graph.intended_settings(profile, 'off', 0, False))
        for mutate in (lambda r: r['choices'][0]['message'].update(reasoning='positive reasoning'),
                       lambda r: r['choices'][0].update(finish_reason='length'),
                       lambda r: r['usage'].update(completion_tokens=1),
                       lambda r: r.update(model='wrong'),
                       lambda r: r['usage'].pop('completion_tokens_details')):
            saved = json.loads(saved_text)
            turn = saved['turns']['B']
            raw = json.loads(turn['provider_response_raw']); mutate(raw)
            self.refresh_envelope(turn, raw)
            self.assertFalse(all(graph.audit_artifact(saved).values()))
        saved = json.loads(saved_text)
        saved['turns']['B']['provider_reasoning'] = 'tampered'
        self.assertFalse(graph.audit_artifact(saved)['responses'])

    def test_together_missing_evidence_and_conflicts_stop_no_retry(self):
        profile = 'gemma_31b_together'
        for problem in ('invalid_count', 'conflicting_alias', 'invalid_alias'):
            response = envelope('answer', profile)
            if problem == 'invalid_count':
                response['usage']['completion_tokens_details']['reasoning_tokens'] = 'unknown'
            elif problem == 'conflicting_alias':
                response['choices'][0]['message']['reasoning'] = 'positive reasoning'
            else:
                response['choices'][0]['message']['reasoning'] = 123
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    self.run_mock(directory, [response], profile=profile)
                saved = json.loads(next(Path(directory).glob('*.json')).read_text())
                self.assertEqual(json.loads(saved['turns']['A']['provider_response_raw']), response)
                self.assertNotIn('raw_response', saved['turns']['B'])
                self.assertEqual(len(self.last_bodies), 1)

    def test_hosted_unknown_metadata_and_assurance(self):
        for profile in graph.HOSTED_PROFILES:
            c = config(profile)
            c['effective'].update(runtime=None, template_sha256=None,
                runtime_source='Synthetic provider-managed runtime, unavailable',
                template_source='Synthetic provider-managed template, unavailable',
                thinking=None, response_length_limit=None,
                app_settings_inapplicable={k:'Hosted API has no Bionic switch' for k in
                    ('thinking','reasoning_budget','response_length_limit')})
            response = c['preflight']['response']
            response['usage'].pop('completion_tokens_details')
            graph.validate_deployment(c, profile, 'off')
            result = graph.reasoning_check(response, 'off', c['effective'], profile, c)
            self.assertTrue(result['mode_verified'])
            self.assertIsNone(result['reasoning_tokens'])
            self.assertEqual(result['evidence'], 'unknown')
            self.assertEqual(result['verification_basis'], 'provider_documented_disable')
            for field in ('runtime_source','template_source'):
                bad = copy.deepcopy(c); bad['effective'].pop(field)
                with self.assertRaisesRegex(ValueError,'provenance'):
                    graph.validate_deployment(bad, profile, 'off')
            for field,value in [('scope','generic_hybrid_models'),('model','another-model'),
                                ('endpoint','https://another.example/v1'),('source',' '),
                                ('date','not-a-date'),('effect','hides_trace_only'),
                                ('request_fields',{})]:
                bad = copy.deepcopy(c); bad['effective']['hosted_evidence'][field]=value
                with self.assertRaises(ValueError):graph.validate_deployment(bad,profile,'off')
        for field in ('runtime','template_sha256'):
            c = config(); c['effective'][field]=None
            with self.assertRaises(ValueError):graph.validate_deployment(c,'gemma_e4b','off')
        local = graph.reasoning_check(envelope('301'),'off',{},'gemma_e4b')
        self.assertEqual(local,{'mode_verified':True,'reasoning_tokens':0,'evidence':'false','control_violation':False})

    def test_hosted_positive_reasoning_and_echoes_fail_off(self):
        for profile in graph.HOSTED_PROFILES:
            c=config(profile)
            self.assertEqual(graph.reasoning_check(c['preflight']['response'],'off',c['effective'],profile,c)
                             ['verification_basis'],'measured_zero')
            mutations=[lambda r:r['usage']['completion_tokens_details'].update(reasoning_tokens=1),
                       lambda r:r['choices'][0]['message'].update(reasoning_content='Thinking'),
                       lambda r:r['choices'][0]['message'].update(reasoning='Thinking'),
                       lambda r:r.update(reasoning_enabled=True),
                       lambda r:r.update(metadata={'reasoning_enabled':True})]
            for mutate in mutations:
                bad=copy.deepcopy(c); mutate(bad['preflight']['response'])
                with self.assertRaises(ValueError):graph.validate_deployment(bad,profile,'off')
            c['effective']['hosted_evidence']['control_echo']={
                'path':['provider_controls','thinking'],'expected':False,'source':'Synthetic echo schema'}
            c['preflight']['response']['provider_controls']={'thinking':True}
            with self.assertRaises(ValueError):graph.validate_deployment(c,profile,'off')

    def test_hosted_assurance_saved_raw_audit_and_identity(self):
        profile='gemma_31b_together'; c=config(profile)
        measured=graph.readiness_identity(c,'off')
        c['preflight']['response']['usage'].pop('completion_tokens_details')
        self.assertNotEqual(measured,graph.readiness_identity(c,'off'))
        responses=[envelope('nonempty malformed JSON',profile) for _ in range(2)]
        for response in responses:response['usage'].pop('completion_tokens_details')
        with tempfile.TemporaryDirectory() as directory:
            _,path,_=self.run_mock(directory,responses,profile=profile,deployment=c)
            text=Path(path).read_text()
        saved=json.loads(text)
        self.assertTrue(all(graph.audit_artifact(saved).values()))
        self.assertEqual(saved['identity']['hosted_readiness'],{
            'policy':graph.HOSTED_READINESS_POLICY,'verification_basis':'provider_documented_disable'})
        for turn in saved['turns'].values():
            self.assertFalse(turn['scored']['correct'])
            self.assertIsNone(turn['control_check']['reasoning_tokens'])
            self.assertEqual(turn['reasoning_evidence'],'unknown')
        for mutate in (lambda r:r.update(reasoning_enabled=True),
                       lambda r:r['usage'].update(completion_tokens=8193),
                       lambda r:r['choices'][0]['message'].update(reasoning_content='Thinking')):
            changed=json.loads(text); turn=changed['turns']['B']
            raw=json.loads(turn['provider_response_raw']); mutate(raw); self.refresh_envelope(turn,raw)
            self.assertFalse(all(graph.audit_artifact(changed).values()))
        changed=json.loads(text); changed['turns']['B']['provider_usage']['completion_tokens']=1
        self.assertFalse(graph.audit_artifact(changed)['responses'])
        changed=json.loads(text); changed['identity']['hosted_readiness']['policy']='old_policy'
        self.assertNotEqual(pilot._icl_run_id(changed['identity']),saved['run_id'])
        self.assertFalse(graph.audit_artifact(changed)['identity'])
        c['effective'].pop('hosted_evidence')
        with self.assertRaises(ValueError):graph.readiness_identity(c,'off')

    def test_hosted_experimental_controls_still_required(self):
        for field,value in [('max_output_tokens',8191),('history_truncation',True),
                            ('temperature',0),('top_p',.5),('top_k',1),('repeat_penalty',1.1),
                            ('min_p',0),('output_context_source',''),('no_reasoning_budget_source',''),
                            ('thinking',None),('response_length_limit',None)]:
            c=config('gemma_31b_together'); c['effective'][field]=value
            with self.assertRaises(ValueError):graph.validate_deployment(c,'gemma_31b_together','off')

    def test_blank_saved_answers_stop_without_rewriting_or_retry(self):
        for name in ('A', 'B'):
            for content in (' ', '\n', '\t', ' \n\t '):
                with self.subTest(turn=name, content=content), tempfile.TemporaryDirectory() as directory:
                    responses = [envelope(content)] if name == 'A' else [envelope('malformed A'), envelope(content)]
                    with self.assertRaisesRegex(RuntimeError, 'operational failure'):
                        self.run_mock(directory, responses)
                    saved = json.loads(next(Path(directory).glob('*.json')).read_text())
                    self.assertEqual(saved['failure']['turn'], name)
                    self.assertEqual(saved['turns'][name]['raw_response'], content)
                    self.assertEqual(saved['turns'][name]['provider_response_raw'], json.dumps(responses[-1]))
                    self.assertEqual(len(self.last_bodies), len(responses))
                    self.assertFalse(graph.audit_artifact(saved)['responses'])
        with tempfile.TemporaryDirectory() as directory:
            content = ' \nmalformed JSON\t '
            _, path, bodies = self.run_mock(directory, [envelope(content), envelope(content)])
            saved = json.loads(Path(path).read_text())
            self.assertTrue(all(graph.audit_artifact(saved).values()))
            self.assertEqual(bodies[1]['messages'][1]['content'], content)
            for turn in saved['turns'].values():
                self.assertEqual(turn['raw_response'], content)
                self.assertFalse(turn['parsed']['well_formed'])

    @staticmethod
    def refresh_envelope(turn, response):
        """Simulate a saved raw envelope with valid hashes, not shared objects."""
        turn['provider_response_raw'] = json.dumps(response)
        turn['provider_response_raw_sha256'] = pilot.sha256_text(turn['provider_response_raw'])
        turn['provider_response'] = json.loads(turn['provider_response_raw'])
        turn['provider_response_sha256'] = pilot._canonical_sha256(turn['provider_response'])

    def test_audit_rejects_completed_blank_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = self.run_mock(directory, [envelope('malformed A'), envelope('malformed B')])
            saved_text = Path(path).read_text()
        for name in ('A', 'B'):
            saved = json.loads(saved_text)
            turn = saved['turns'][name]
            response = json.loads(turn['provider_response_raw'])
            response['choices'][0]['message']['content'] = ' \n\t '
            self.refresh_envelope(turn, response)
            turn['raw_response'] = ' \n\t '
            turn['response_sha256'] = pilot.sha256_text(turn['raw_response'])
            if name == 'A':
                b = saved['turns']['B']
                b['previous_response_sha256'] = turn['response_sha256']
                b['request_body']['messages'][1]['content'] = turn['raw_response']
                b['request_sha256'] = pilot._canonical_sha256(b['request_body'])
            checks = graph.audit_artifact(saved)
            self.assertFalse(checks['responses'])
            self.assertFalse(checks['transport'])
            self.assertTrue(checks['links'])
            self.assertTrue(checks['request_match'])

    def test_audit_reconciles_copies_with_original_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = self.run_mock(directory, [envelope('malformed A'), envelope('malformed B')])
            saved_text = Path(path).read_text()
        for field, value in (('provider_usage', {'prompt_tokens': 5000, 'completion_tokens': 1}),
                             ('provider_finish_reason', 'length'), ('raw_response', 'different'),
                             ('provider_reasoning', 'different reasoning'), ('actual_response_model', 'wrong'),
                             ('system_fingerprint', 'wrong'), ('reasoning_evidence', 'true'),
                             ('reasoning_control_violation', True), ('control_check', {})):
            with self.subTest(field=field):
                saved = json.loads(saved_text)
                turn = saved['turns']['B']
                if field == 'provider_usage':
                    value = {**turn[field], 'completion_tokens': 1}
                turn[field] = value
                turn['response_sha256'] = pilot.sha256_text(turn['raw_response'])
                self.assertEqual(json.loads(turn['provider_response_raw'])['usage']['completion_tokens'], 800)
                self.assertFalse(graph.audit_artifact(saved)['responses'])
        saved = json.loads(saved_text)
        turn = saved['turns']['B']
        turn['provider_response']['usage']['completion_tokens'] = 1
        turn['provider_response_sha256'] = pilot._canonical_sha256(turn['provider_response'])
        self.assertFalse(graph.audit_artifact(saved)['responses'])

    def test_audit_operational_checks_use_original_values(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = self.run_mock(directory, [envelope('malformed A'), envelope('malformed B')])
            saved_text = Path(path).read_text()
        mutations = (
            ('transport', lambda r: r['choices'][0].update(finish_reason='length')),
            ('transport', lambda r: r['usage'].update(completion_tokens=8193)),
            ('context', lambda r: r['usage'].update(prompt_tokens=32768)),
            ('controls', lambda r: r['choices'][0]['message'].update(reasoning_content='reasoning')),
            ('controls', lambda r: r.update(model='wrong')),
        )
        for guard, mutate in mutations:
            with self.subTest(guard=guard):
                saved = json.loads(saved_text)
                turn = saved['turns']['B']
                response = json.loads(turn['provider_response_raw'])
                mutate(response)
                self.refresh_envelope(turn, response)
                # Copies still say stop/800/no reasoning/correct model.
                self.assertEqual(turn['provider_finish_reason'], 'stop')
                checks = graph.audit_artifact(saved)
                self.assertFalse(checks['responses'])
                self.assertFalse(checks[guard])
        for raw in ('not JSON', '{}', '{"choices": []}'):
            saved = json.loads(saved_text)
            turn = saved['turns']['B']
            turn['provider_response_raw'] = raw
            turn['provider_response_raw_sha256'] = pilot.sha256_text(raw)
            self.assertFalse(graph.audit_artifact(saved)['responses'])

    def test_preview_failure_guards(self):
        from experiments.preview_icl_graph import generate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'previews'
            report = generate(output, 4, 20)
            self.assertEqual(len(report['prompts']), 6)
            self.assertEqual(report['model_calls'], 0)
            with self.assertRaisesRegex(ValueError, 'exists'):
                generate(output)
            with patch('icl_graph.reference_answer', return_value=('{}', {})):
                with self.assertRaisesRegex(ValueError, 'reference did not pass'):
                    generate(Path(directory) / 'bad')
            self.assertFalse((Path(directory) / 'bad').exists())

    def test_failures_preserved_no_retry(self):
        for response in (envelope('partial', finish='length'), envelope(''),
                         envelope('answer', mode='on'), urllib.error.URLError('network failure')):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    self.run_mock(directory, [response])
                files = list(Path(directory).glob('*.json'))
                self.assertEqual(len(files), 1)
                saved = json.loads(files[0].read_text())
                self.assertIn('failure', saved)
                self.assertNotIn('raw_response', saved['turns']['B'])
                if isinstance(response, dict):
                    self.assertEqual(saved['turns']['A']['raw_response'], response['choices'][0]['message']['content'])
                self.assertEqual(len(self.last_bodies), 1)

    def test_context_loss_and_sensitive_envelopes_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'history does not fit'):
                self.run_mock(directory, [envelope('x' * 40000)])
            self.assertEqual(len(self.last_bodies), 1)
            saved = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(saved['failure']['type'], 'ContextBound')
            self.assertEqual(saved['turns']['A']['raw_response'], 'x' * 40000)
        with tempfile.TemporaryDirectory() as directory:
            sensitive = envelope('answer'); sensitive['api_key'] = 'mock-sensitive-value'
            with self.assertRaises(RuntimeError):
                self.run_mock(directory, [sensitive])
            saved_text = next(Path(directory).glob('*.json')).read_text()
            self.assertNotIn('mock-sensitive-value', saved_text)
            self.assertNotIn('provider_response_raw', json.loads(saved_text)['turns']['A'])

    def test_matched_on_uses_all_repeats_and_refuses_changed_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            c = config()
            for repeat in (1, 2, 3):
                a = args(provider='openai')
                ext = graph.GraphRun(a, self.views['graph_ab'], self.prompts['graph_ab'], repeat - 1, c, self.prompts)
                values = [envelope(ext.dry_answers[t]) for t in ('A', 'B')]
                with patch('icl_graph.urllib.request.OpenerDirector.open', side_effect=[Reply(v) for v in values]):
                    artifact, path, _ = pilot.run_icl_two_response_once(self.record, self.view, self.sc, True, a,
                            'graph_ab', repeat, repeat - 1, directory, ext)
            self.assertEqual(graph.matched_off(directory, config(mode='on'), 'graph_ab', self.prompts['graph_ab']), [])
            artifact['turns']['A']['scored']['correct'] = False
            pilot._write_json_atomic(path, artifact)
            self.assertEqual(graph.matched_off(directory, config(mode='on'), 'graph_ab', self.prompts['graph_ab']), [1, 2, 3])
            changed = config(mode='on'); changed['effective']['runtime'] = 'different'
            with self.assertRaisesRegex(ValueError, 'must match OFF'):
                graph.matched_off(directory, changed, 'graph_ab', self.prompts['graph_ab'])

    def test_suite_no_resume_and_conditional_on(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'new')
            results = graph.run_suite(self.sc, args(), out)
            artifacts = [json.loads(Path(r['path']).read_text()) for r in results]
            self.assertEqual(len(artifacts), 3)
            self.assertEqual(graph.on_repeats(artifacts), [])
            artifacts[0]['turns']['A']['scored']['correct'] = False
            self.assertEqual(graph.on_repeats(artifacts), [1, 2, 3])
            with self.assertRaises(ValueError):
                graph.on_repeats([])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                graph.run_suite(self.sc, args(), out)
            summary = json.loads((Path(out) / 'summary.json').read_text())
            self.assertEqual(summary['protocol'], graph.PROTOCOL)
            self.assertTrue(summary['operational_gate_pass'])
            self.assertEqual(summary['expected_runs'], 3)


if __name__ == '__main__':
    unittest.main()
