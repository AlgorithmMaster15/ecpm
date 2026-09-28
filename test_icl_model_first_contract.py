import copy
import json
import unittest
import icl_model_first as d

class ContractTests(unittest.TestCase):
    def test_fixture_coverage_and_changed_graph(self):
        for seed in (8,13,25):
            w=d.load(seed)
            for p in ('A','B'):
                self.assertEqual(len(w[p]['graph']),16)
                self.assertEqual(len(w[p]['rows']),160)
                self.assertEqual(len(w[p]['nodes']),8)
            changes=[(a,b) for a,b in zip(w['A']['graph'],w['B']['graph']) if a!=b]
            self.assertEqual(len(changes),1)
            a,b=changes[0]
            self.assertEqual(a['destination'],b['destination'])
            self.assertTrue(a['available'] and b['available'])
            self.assertEqual((a['p_success'],b['p_success']),(1,0))
            self.assertEqual(len(w['controls']),4)
            self.assertNotIn(w['target'],[[c['node'],c['action']] for c in w['controls']])

    def test_sequential_logs_reconstruct_full_world(self):
        for seed in (8,13,25):
            w=d.load(seed); previous=None
            for p in ('A','B'):
                previous=d.from_logs(w[p],previous)
                self.assertEqual(previous,w[p]['graph'])

    def test_evidence_shared_except_graph(self):
        for seed in (8,13,25):
            w=d.load(seed)
            for p in ('A','B'):
                common=d.evidence(w,p,'task_only')
                self.assertEqual(common,d.evidence(w,p,'model_first'))
                self.assertEqual(common,d.evidence(w,p,'graph_given').replace(d.graph_text(w[p])+'\n','',1))

    def test_b_graph_is_updated(self):
        w=d.load(8); p=d.prompts(w,'B','graph_given')['task']
        self.assertIn('G | a1 | true | E | 0',p)
        self.assertNotIn('G | a1 | true | E | 1',p)

    def test_no_future_information_in_a(self):
        w=d.load(8); changed=copy.deepcopy(w); changed['B']={'bad':'FUTURE SENTINEL'}; changed['target']=['X','bad']; changed['controls']=[]
        for arm in d.ARMS:
            self.assertEqual(d.prompts(w,'A',arm),d.prompts(changed,'A',arm))
        self.assertEqual(d.queries(w),d.queries(changed))

    def test_no_representation_or_route_queries_before_construction(self):
        w=d.load(8)
        p=d.prompts(w,'A','model_first')['model'].lower()
        for banned in ('graph','json','q0','q1','route queries','target','control'):
            self.assertNotIn(banned,p)
        self.assertIn('whatever representation',p)
        self.assertNotIn('construct a model',d.prompts(w,'A','task_only')['task'])

    def test_common_tasks_and_measurement_text(self):
        for p in ('A','B'):
            w=d.load(8)
            for arm in d.ARMS:
                self.assertTrue(d.prompts(w,p,arm)['task'].endswith(d.task(w,p)))
                self.assertTrue(d.prompts(w,p,arm)['readout'].endswith(d.readout(w,p)))

    def test_measurement_branch_excludes_current_tasks(self):
        w=d.load(8)
        answers={p+'_'+s:f'UNIQUE_ANSWER_{p}_{s}' for p in ('A','B') for s in ('model','task','readout')}
        for arm in d.ARMS:
            calls=d.schedule(w,arm,answers)
            for call in calls:
                texts=[x['content'] for x in call['messages']]
                self.assertFalse(any(answers[p+'_readout'] in t for p in ('A','B') for t in texts))
                if call['id'].endswith('readout'):
                    p=call['id'][0]
                    self.assertNotIn(answers[p+'_task'],texts)
                    self.assertNotIn(d.task(w,p),texts)
                    self.assertNotIn('Route queries:',texts[-1])
                if call['id'].startswith('B_'):
                    self.assertIn(answers['A_task'],texts)
                if arm=='model_first' and call['id'].endswith(('task','readout')):
                    self.assertIn(answers[call['id'][0]+'_model'],texts)

    def test_matched_allowances_and_request_count(self):
        w=d.load(8); answers={p+'_'+s:'visible answer' for p in ('A','B') for s in ('model','task','readout')}
        counts=[]
        for arm in d.ARMS:
            calls=d.schedule(w,arm,answers); counts.append(len(calls))
            self.assertEqual(sum(c['max_output_tokens'] for c in calls),24576)
            for p in ('A','B'):
                self.assertEqual(sum(c['max_output_tokens'] for c in calls if c['id'].startswith(p) and not c['id'].endswith('readout')),8192)
        self.assertEqual(counts,[6,4,4]); self.assertEqual(sum(counts)*3,42)

    def test_queries_unique_and_reproducible(self):
        for seed in (8,13,25):
            w=d.load(seed); q=d.queries(w)
            self.assertEqual(q,d.queries(w))
            self.assertTrue(all(d.shortest(w['A']['graph'],x['start'],x['goal'])['reachable'] for x in q))
            self.assertEqual(len({(x['start'],x['goal']) for x in q}),4)
            self.assertTrue(all(x['start']!=x['goal'] for x in q))
            self.assertEqual((q[0]['start'],q[0]['goal']),(w['A']['start'],w['A']['goal']))
            self.assertEqual([x['query_id'] for x in q],['q0','q1','q2','q3'])

    def test_goal_is_query_specific_and_unreachable_explicit(self):
        rows=d.load(8)['A']['graph']
        self.assertEqual(d.shortest(rows,'G','E'),{'reachable':True,'steps':[{'state':'G','action':'a1'}],'cost':1.0})
        self.assertEqual(d.shortest(rows,'D','G'),{'reachable':False,'steps':[],'cost':None})
        self.assertEqual(d.shortest(rows,'G','G'),{'reachable':True,'steps':[],'cost':0.0})

    def test_expected_attempt_cost_and_zero_probability(self):
        rows=[{'node':'X','action':'a1','available':True,'destination':'Y','p_success':0.5}]
        self.assertEqual(d.shortest(rows,'X','Y')['cost'],2)
        rows[0]['p_success']=0
        self.assertIsNone(d.shortest(rows,'X','Y')['cost'])

    def test_reference_routes_execute_and_match_anchor(self):
        for seed in (8,13,25):
            w=d.load(seed); previous=None
            for p in ('A','B'):
                estimates=d.from_logs(w[p],previous); ref=d.reference(w,p,estimates,previous)
                self.assertEqual(ref['costs']['q0'],2 if p=='A' else 3)
                lookup={(x['node'],x['action']):x for x in estimates}
                for q,route in zip(d.queries(w),ref['task']['routes']):
                    if not route['reachable']:
                        self.assertEqual(route['steps'],[]); continue
                    state=q['start']; cost=0
                    for step in route['steps']:
                        self.assertEqual(step['state'],state)
                        row=lookup[(state,step['action'])]; self.assertGreater(row['p_success'],0)
                        cost+=1/row['p_success']; state=row['destination']
                    self.assertEqual(state,q['goal'])
                    self.assertEqual(cost,ref['costs'][q['query_id']])
                if p=='B':
                    self.assertEqual(ref['task']['changed_pair'],{'state':w['target'][0],'action':w['target'][1]})
                    self.assertEqual(sum(x['changed'] for x in ref['readout']['pairs']),1)
                previous=estimates

    def test_reference_detection_does_not_read_target(self):
        w=d.load(8); a=d.from_logs(w['A']); b=d.from_logs(w['B'],a)
        expected=d.reference(w,'B',b,a)
        w['target']=['WRONG','WRONG']
        self.assertEqual(d.reference(w,'B',b,a),expected)
        unchanged=d.reference(w,'B',a,a)
        self.assertFalse(unchanged['task']['changed']); self.assertIsNone(unchanged['task']['changed_pair'])

    def test_unknown_destination_rejected_in_offline_fixture(self):
        w=d.load(8)
        with self.assertRaises(ValueError): d.from_logs(w['B'])

if __name__=='__main__': unittest.main(verbosity=2)
