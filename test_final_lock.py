"""Regression cases found in the final source audit. No network calls."""

import copy
import contextlib
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agentic_conditions as agentic
import explore_agent
import explore_metrics
import icl_expanded as expanded
import icl_graph
import icl_model_first_runner
import run_pilot
from experiments import preview_icl_expanded
from test_icl_graph import config, envelope
from resource_mdp import make_pair


class FinalLock(unittest.TestCase):
    def test_route_annotations_do_not_count_as_replanning(self):
        world = expanded.build_world(8, 'det', 'no_change')
        answers = expanded.oracle_answers(world)
        b = json.loads(answers['B_task'])
        for route in b['routes']:
            route['note'] = 'same route'
            for step in route['steps']:
                step['note'] = 'extra format field'
        answers['B_task'] = json.dumps(b)
        scores = expanded.score_conversation(world, 'baseline_task',
            {k: {'raw_response': v} for k, v in answers.items()})
        self.assertFalse(scores['periods']['B']['task_parsed']['well_formed'])
        self.assertEqual(scores['updating']['unnecessary_replan_rate'],
                         expanded.fraction(0, 4))

    def test_nullable_reasoning_details_remain_unknown(self):
        for mode in ('off', 'on'):
            c = config('gemma_31b_together', mode)
            response = envelope('{}', 'gemma_31b_together', mode)
            response['usage']['completion_tokens_details'] = None
            check = icl_graph.reasoning_check(response, mode, c['effective'], c['profile'], c)
            self.assertTrue(check['mode_verified'])
            self.assertIsNone(check['reasoning_tokens'])
        response['usage']['completion_tokens_details'] = []
        with self.assertRaisesRegex(ValueError, 'completion token details'):
            icl_graph.reasoning_check(response, mode, c['effective'], c['profile'], c)

    def active_artifact(self, usage):
        sc = dict(run_pilot.SCENARIO_DEFAULTS, seed=8, condition='silent_break')
        args = SimpleNamespace(m0_episodes=1, m1_episodes=1, max_steps_per_episode=20,
            announce_change=False, explore_context_budget=12000, provider='openai',
            model='MOCK', max_tokens=4096, base_url='https://example.invalid/v1')
        original = explore_agent.run_explore_instance
        def exploration(inst, cfg, act_fn):
            act_fn('MOCK system', [{'role':'user', 'content':'MOCK step'}])
            return original(inst, cfg, node_policy_fn=lambda mdp:
                            explore_agent.dry_run_policy(mdp, inst.labels))
        with patch.object(explore_agent, 'run_explore_instance', side_effect=exploration), \
             patch.object(run_pilot, 'with_retry', return_value=('{}', copy.deepcopy(usage))), \
             contextlib.redirect_stdout(io.StringIO()):
            return run_pilot.run_pilot_active(sc, True, args)

    def test_active_usage_aliases_and_unknown_are_not_zero(self):
        for usage, expected in (({'input_tokens':9, 'output_tokens':4}, (45,20)),
                                 ({}, (None,None)),
                                 ({'prompt_tokens':9,'input_tokens':10,
                                   'completion_tokens':4}, (None,20))):
            with self.subTest(usage=usage):
                artifact = self.active_artifact(usage)
                totals = artifact['token_usage_total']
                self.assertEqual(totals['n_calls'], 5)
                self.assertEqual((totals['prompt_tokens'], totals['completion_tokens']), expected)
                self.assertEqual(totals['total_tokens'], sum(expected) if None not in expected else None)
                self.assertEqual(artifact['provider_usage_calls'], [usage] * 5)

    def test_recorded_agentic_system_includes_model_instruction(self):
        try:
            agentic.activate('model_first', mode_prompts=True)
            sent = explore_agent.build_system_prompt('H')
            self.assertEqual(agentic._STATE['system_prompt'], sent)
        finally:
            agentic.activate('task_only')

    def test_exposure_uses_the_metrics_artifact(self):
        inst = make_pair(8, 'silent_break', deterministic=True, matched=True)
        cfg = explore_agent.ExploreConfig(max_episodes_m0=1, max_episodes_m1=1,
                                        max_steps_per_episode=20, seed=8)
        result = explore_agent.run_explore_instance(inst, cfg,
            node_policy_fn=lambda mdp: explore_agent.dry_run_policy(mdp, inst.labels))
        metrics = explore_metrics.compute_explore_metrics(inst, result['m0_episodes'], result['m1_episodes'])
        usage = metrics['changed_action_usage']
        self.assertTrue(usage['m0_route_uses'])
        # Supplying metrics avoids regenerating a world in the sidecar export.
        artifact = {'instance': {'condition':'silent_break'}, 'explore': {'metrics':metrics}}
        with patch.object(agentic, 'make_pair', side_effect=AssertionError('duplicate calculation')):
            e = agentic.exposure(artifact)
        self.assertEqual(e['m0_any_use'], usage['m0']['n_choices'] > 0)
        self.assertEqual(e['m1_attempts'], usage['m1']['n_choices'])
        self.assertEqual(e['exposed'], usage['m0_route_uses'])
        empty = explore_metrics.compute_explore_metrics(inst, [], [])
        self.assertIsNone(empty['changed_action_usage']['m0_route_uses'])

    def test_export_distinguishes_optimal_and_correct_no_route(self):
        world = expanded.build_world(8, 'det', 'silent_break')
        args = preview_icl_expanded.arguments('baseline_task', 'off', expanded.HISTORY_POLICIES[0])
        with tempfile.TemporaryDirectory() as folder:
            artifact = icl_model_first_runner.run_once(world, 'baseline_task', 1,
                                                       args, None, folder, expanded)
        rows = preview_icl_expanded.metric_rows(artifact, 'valid')
        unreachable = 0
        for q, score in artifact['scores']['periods']['B']['routes'].items():
            values = {r['metric']:r for r in rows if r['period']=='B' and r['query_id']==q}
            self.assertEqual(values['route_optimal_solution']['value'], 1)
            self.assertEqual(values['route_oracle_reachable']['value'], int(score['oracle_reachable']))
            if not score['oracle_reachable']:
                unreachable += 1
                self.assertEqual(values['route_optimal_reachable']['denominator'], 0)
                self.assertIsNone(values['route_optimal_reachable']['value'])
                self.assertEqual(values['route_correct_no_route']['value'], 1)
        self.assertGreater(unreachable, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
