"""Locked model-first prompts, row-level scoring and offline references."""
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path

PROTOCOL = 'icl_model_first_v1'
ARMS = ('model_first', 'task_only', 'graph_given')
ROOT = Path(__file__).resolve().parent / "fixtures" / "icl_model_first"
SYSTEM = 'Follow the instructions in each message and use the supplied evidence.'
COMMON = '''You will answer questions about a system with states and actions. Each
period supplies observations of that system. Period B may or may not differ
from Period A.

Each available action has one destination within a period. A successful
attempt moves to that destination; a failed attempt stays at the current
state. An action can be retried after failure. Every attempt costs 1.
Each observation is [current_state, action, observed_next_state].

An action is available when it is listed in the current action menu.
Availability does not mean success. Its destination is the state reached
on success, even when its current success probability is zero.

If a current-period specification is supplied, use its stated transitions.
Otherwise, estimate from the observations for that period: count an attempt
as successful exactly when its next state differs from its current state.
For each action, divide its successful attempts by all its observed attempts.
Count every row for that state-action pair, wherever it appears in the list.
Do not combine observations from different periods. Zero successes means
probability 0. Use a successful observation to identify the destination.
If the current period has no successes, retain the latest destination
supported by earlier supplied evidence; do not retain the earlier probability.'''
MODEL_A = '''Construct a model of how this system behaves from the supplied evidence.
Use whatever representation you find useful. Describe the current system
so that your model can be used to answer questions about it later.'''
MODEL_B = '''Update your model of how this system behaves using the Period B evidence.
Use whatever representation you find useful. Describe the current system
so that your model can be used to answer questions about it later.'''
ROUTE_RULES = '''For each query below, find a route that minimizes the expected number of
attempts from its start to its goal in the current period. Each step gives
the state where an action is taken and that action. After failure, retry
the same action; after success, take the next step. Stop on reaching the
query's goal: do not append a goal item or an action after arrival.

Use reachable=false and steps=[] only if no route can reach that goal
with finite expected cost. Otherwise use reachable=true and give the
complete action sequence. Include each query exactly once, with at most
32 action steps per route. Use only supplied state and action names.'''
COMPARE = '''Compare Period B with Period A. A change means a difference in action
availability, destination or success probability, not a difference in your
wording. Report whether anything changed. If so, identify the changed
state-action pair; otherwise set changed=false and changed_pair=null.'''
JSON_END = 'Return exactly one JSON object, without commentary or code fences. Include every required field. Replace all placeholders with actual values.'

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def load(seed):
    if seed not in (8, 13, 25):
        raise ValueError('only locked seeds 8, 13 and 25 are supported')
    path = f'worlds/seed_{seed}.json'
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != lock()['files_sha256'][path]:
        raise ValueError('world fixture hash mismatch')
    return json.loads(raw)

def queries(world):
    # Selects from pairs reachable in Period A. Never reads Period B.
    a = world['A']; anchor = (a['start'], a['goal'])
    candidates = [(s,t) for s in sorted(a['nodes']) for t in sorted(a['nodes']) if s!=t and (s,t)!=anchor and shortest(a['graph'],s,t)['reachable']]
    ranked = sorted(candidates, key=lambda x:(digest(f'{PROTOCOL}|route-probes-v1|{world["seed"]}|{x[0]}|{x[1]}'),x))
    return [{'query_id':f'q{i}','start':s,'goal':t,'kind':'anchor' if i==0 else 'sampled'} for i,(s,t) in enumerate([anchor]+ranked[:3])]

def graph_text(view):
    lines=['Current-period specification:','state | action | available | destination | p_success']
    for x in view['graph']:
        lines.append(f'{x["node"]} | {x["action"]} | true | {x["destination"]} | {x["p_success"]:g}')
    return '\n'.join(lines)

def evidence(world, period, arm):
    v=world[period]
    lines=[COMMON, f'Period {period}', 'States: '+', '.join(v['nodes']), 'Current action menu:']
    lines += [f'{s}: '+(', '.join(v['menu'][s]) or '(no actions)') for s in sorted(v['menu'])]
    if arm=='graph_given': lines += [graph_text(v)]
    lines += [f'Shuffled single-step observations, Period {period}:'] + v['rows']
    return '\n\n'.join(lines[:3])+'\n'+'\n'.join(lines[3:])

def task(world, period):
    q=queries(world)
    lines=[f'Period {period} tasks',ROUTE_RULES,'Route queries:']
    lines += [f'{x["query_id"]}: {x["start"]} to {x["goal"]}' for x in q]
    if period=='B': lines += [COMPARE]
    route='{ "query_id": "<query id>", "reachable": <true or false>, "steps": [{"state":"<state>","action":"<action>"}] }'
    schema='{ "routes": [ '+route+', <one entry for each remaining query> ]'
    if period=='B': schema+=', "changed": <true or false>, "changed_pair": <null or {"state":"<state>","action":"<action>"}>'
    schema+=' }'
    lines += ['Required output shape (use steps=[] for an unreachable query):',schema,JSON_END]
    return '\n\n'.join(lines)

def readout(world, period):
    v=world[period]
    pairs=[(s,a) for s in sorted(v['menu']) for a in v['menu'][s]]
    text=f'''Report the current Period {period} transition for each state-action pair listed
below. Include each pair exactly once. Every entry must contain state,
action, available, destination and p_success. Available is a boolean.
When available=true, destination must be a supplied state name and
p_success a number from 0 to 1. Probability zero does not remove the action
or its destination. When available=false, use destination=null and
p_success=null. Do not use the string "null".
'''
    if period=='B': text+='\nAlso include changed for every pair: true if its availability, destination\nor probability differs from Period A; otherwise false.\n'
    text+='\nPairs to report:\n'+'\n'.join(f'{s}: {a}' for s,a in pairs)
    text+='\n\nRequired output shape:\n{"pairs":[{"state":"<state>","action":"<action>","available":<true or false>,"destination":<state name or null>,"p_success":<number or null>'
    if period=='B': text+=',"changed":<true or false>'
    text+='}, <one entry for each remaining pair>]}\n'+JSON_END
    return text

def prompts(world, period, arm):
    e=evidence(world,period,arm)
    if arm=='model_first':
        return {'model':e+'\n\n'+(MODEL_A if period=='A' else MODEL_B), 'task':task(world,period), 'readout':readout(world,period)}
    return {'task':e+'\n\n'+task(world,period), 'readout':e+'\n\n'+readout(world,period)}

def schedule(world, arm, answers):
    """Returns actual request histories with supplied synthetic/saved answer text.
Readout runs after task but forks from before the current task. Its answers
are never inserted in main history. Private provider reasoning is not input.
"""
    history=[{'role':'system','content':SYSTEM}]; calls=[]
    for period in ('A','B'):
        p=prompts(world,period,arm)
        if arm=='model_first':
            history.append({'role':'user','content':p['model']})
            calls.append({'id':period+'_model','max_output_tokens':4096,'messages':copy.deepcopy(history)})
            history.append({'role':'assistant','content':answers[period+'_model']})
        before_task=copy.deepcopy(history)
        history.append({'role':'user','content':p['task']})
        calls.append({'id':period+'_task','max_output_tokens':4096 if arm=='model_first' else 8192,'messages':copy.deepcopy(history)})
        history.append({'role':'assistant','content':answers[period+'_task']})
        calls.append({'id':period+'_readout','max_output_tokens':4096,'messages':before_task+[{'role':'user','content':p['readout']}]})
    return calls

def shortest(rows, start, goal):
    """Independent weighted reference: expected attempts sum of 1/p."""
    queue=[(0.0,start,[])]; best={start:0.0}
    while queue:
        cost,state,steps=heapq.heappop(queue)
        if cost!=best[state]: continue
        if state==goal: return {'reachable':True,'steps':steps,'cost':cost}
        for r in sorted(rows,key=lambda r:(r['node'],r['action'])):
            if r['node']!=state or not r['available'] or r['p_success']<=0: continue
            nxt=r['destination']; new=cost+1/r['p_success']
            if new<best.get(nxt,float('inf')):
                best[nxt]=new
                heapq.heappush(queue,(new,nxt,steps+[{'state':state,'action':r['action']}]))
    return {'reachable':False,'steps':[],'cost':None}

def from_logs(view, earlier=None):
    estimates=[]; old={} if earlier is None else {(r['node'],r['action']):r for r in earlier}
    for s in sorted(view['menu']):
        for action in view['menu'][s]:
            rows=[line.strip('[]').split(', ') for line in view['rows']]
            obs=[r[2] for r in rows if r[:2]==[s,action]]
            if not obs: raise ValueError('This pilot requires full observation coverage')
            successes=[d for d in obs if d!=s]
            if len(set(successes))>1: raise ValueError('Non-unique destination')
            dest=successes[0] if successes else old.get((s,action),{}).get('destination')
            if dest is None: raise ValueError('Unknown destination requires a different protocol')
            estimates.append({'node':s,'action':action,'available':True,'destination':dest,'p_success':len(successes)/len(obs)})
    return estimates

def reference(world, period, estimates, earlier=None):
    routes=[]; costs={}
    for q in queries(world):
        r=shortest(estimates,q['start'],q['goal']); costs[q['query_id']]=r.pop('cost')
        routes.append({'query_id':q['query_id'],**r})
    main={'routes':routes}
    changes=set()
    if period=='B':
        if earlier is None: raise ValueError('B reference needs previous estimates')
        previous={(r['node'],r['action']):r for r in earlier}
        changes={(r['node'],r['action']) for r in estimates if previous.get((r['node'],r['action']))!=r}
        if len(changes)>1: raise ValueError('This pilot has at most one changed pair')
        changed_pair=next(iter(changes)) if changes else None
        main.update(changed=bool(changes),changed_pair=None if changed_pair is None else {'state':changed_pair[0],'action':changed_pair[1]})
    pairs=[]
    for r in estimates:
        item={'state':r['node'],**{k:v for k,v in r.items() if k!='node'}}
        if period=='B': item['changed']=(r['node'],r['action']) in changes
        pairs.append(item)
    return {'task':main,'readout':{'pairs':pairs},'costs':costs}


SCORER = 'icl_model_first_rows_v1'
TRANSITION_FIELDS = ('available', 'destination', 'p_success')


def lock():
    return json.loads((ROOT / 'lock.json').read_text())


def verify_world(seed):
    """Rebuild with the unchanged generator, including observation order/controls."""
    import icl_graph as graph
    import run_pilot as pilot
    w = load(seed)
    sc = dict(pilot.SCENARIO_DEFAULTS)
    sc.update(pilot.SCENARIOS[f'icl_det_gate_seed{seed}'])
    record, view, variants, _ = graph.prepare(sc)
    target = pilot.protocol_target_pair(record, sc)
    for p, v in zip(('A', 'B'), variants['graph_ab']):
        if v != w[p]:
            raise ValueError('generated period does not match locked fixture: ' + p)
    if list(target) != w['target']:
        raise ValueError('generated target differs from fixture')
    controls = [q for q in pilot.queried_pairs_for_icl(record, sc)
                if (q['node'], q['action']) != tuple(target)]
    if controls != w['controls']:
        raise ValueError('generated controls differ from fixture')
    return w


def verify_prompts(w):
    results = {}
    for arm in ARMS:
        for period in ('A', 'B'):
            for stage, text in prompts(w, period, arm).items():
                path = f'prompts/seed_{w["seed"]}/{arm}/{period}_{stage}.txt'
                # Preview files have one terminal LF; request text follows the
                # locked executable schedule, without adding that file delimiter.
                actual = digest(text + '\n')
                if actual != lock()['files_sha256'][path]:
                    raise ValueError('locked prompt drift: ' + path)
                results[path] = {'file_sha256': actual, 'request_text_sha256': digest(text)}
    return results


def verify_histories(w):
    answers = {p + '_' + s: f'SYNTHETIC {p}_{s}; NOT A MODEL RESULT'
               for p in ('A', 'B') for s in ('model', 'task', 'readout')}
    result = {}
    for arm in ARMS:
        raw = json.dumps(schedule(w, arm, answers), indent=2) + '\n'
        path = f'prompts/seed_{w["seed"]}/{arm}/request_history_example.json'
        if digest(raw) != lock()['files_sha256'][path]:
            raise ValueError('locked history example drift: ' + path)
        result[path] = digest(raw)
    return result


def strict_object(raw):
    """Whole-response JSON only. Duplicate member names invalidate the document."""
    def members(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON member: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('non-finite JSON constant: ' + value)

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('non-finite JSON number')
        return parsed

    try:
        value = json.loads(raw, object_pairs_hook=members, parse_constant=constant, parse_float=number)
        if not isinstance(value, dict):
            raise ValueError('top level is not an object')
        return value, []
    except (ValueError, TypeError) as exc:
        return None, [str(exc)]


def pair_keys(view):
    return [(s, a) for s in sorted(view['menu']) for a in view['menu'][s]]


def pair_id(pair):
    return ':'.join(pair)


def finite_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def transition_valid(row, nodes):
    if not isinstance(row, dict) or not all(k in row for k in TRANSITION_FIELDS):
        return False
    if type(row['available']) is not bool:
        return False
    if not row['available']:
        return row['destination'] is None and row['p_success'] is None
    return (isinstance(row['destination'], str) and row['destination'] in nodes
            and finite_probability(row['p_success']))


def indexed_entries(value, field, expected, key_fn, errors):
    buckets = {key: [] for key in expected}
    if not isinstance(value, dict) or not isinstance(value.get(field), list):
        errors.append('missing or invalid ' + field)
        return buckets
    for i, row in enumerate(value[field]):
        try:
            key = key_fn(row)
            if key not in buckets:
                raise ValueError()
            buckets[key].append(row)
        except (KeyError, TypeError, ValueError):
            errors.append(f'{field}[{i}]: unknown or invalid identifier')
    return buckets


def parse_readout(raw, view):
    obj, errors = strict_object(raw)
    buckets = indexed_entries(obj, 'pairs', pair_keys(view),
                              lambda r: (r['state'], r['action']), errors)
    identifiable_table = not errors
    if obj is not None and set(obj) != {'pairs'}:
        errors.append('unexpected top-level fields')
    rows = {}
    for key, entries in buckets.items():
        row = entries[0] if len(entries) == 1 else None
        row_errors = []
        if row is None:
            row_errors.append('missing pair' if not entries else 'duplicate pair')
        valid = row is not None and transition_valid(row, view['nodes'])
        changed_valid = row is not None and type(row.get('changed')) is bool
        if row is not None:
            allowed = {'state', 'action', *TRANSITION_FIELDS}
            if view['period'] == 'B':
                allowed.add('changed')
            if set(row) - allowed:
                row_errors.append('unexpected row fields')
            if not valid:
                row_errors.append('missing or invalid transition fields')
            if view['period'] == 'B' and not changed_valid:
                row_errors.append('missing or invalid changed')
        rows[pair_id(key)] = {'value': row, 'transition_valid': valid,
                             'changed_valid': changed_valid if view['period'] == 'B' else None,
                             'format_valid': not row_errors, 'errors': row_errors}
    well_formed = not errors and all(r['format_valid'] for r in rows.values())
    return {'status': 'ok' if well_formed else 'invalid_json' if obj is None else 'partial',
            'well_formed': well_formed, 'errors': errors, 'rows': rows,
            'complete_model': identifiable_table and all(r['transition_valid'] for r in rows.values())}


def parse_task(raw, w, period):
    obj, errors = strict_object(raw)
    expected_fields = {'routes'} | ({'changed', 'changed_pair'} if period == 'B' else set())
    if obj is not None and set(obj) - expected_fields:
        errors.append('unexpected top-level fields')
    buckets = indexed_entries(obj, 'routes', [q['query_id'] for q in queries(w)],
                              lambda r: r['query_id'], errors)
    routes = {}
    v = w[period]
    for key, entries in buckets.items():
        row = entries[0] if len(entries) == 1 else None
        row_errors = []
        valid = isinstance(row, dict) and type(row.get('reachable')) is bool
        steps = row.get('steps') if row else None
        valid = valid and isinstance(steps, list) and len(steps) <= 32
        if valid:
            valid = all(isinstance(s, dict) and {'state', 'action'} <= set(s)
                        and isinstance(s['state'], str) and isinstance(s['action'], str)
                        and s['action'] in v['menu'].get(s['state'], []) for s in steps)
            valid = valid and (row['reachable'] or steps == [])
        if isinstance(steps, list):
            row_errors.extend(f'unexpected route step fields at index {i}' for i, s in enumerate(steps)
                              if isinstance(s, dict) and set(s) - {'state', 'action'})
        if not valid:
            row_errors.append('duplicate query' if len(entries) > 1 else 'missing or invalid route fields')
        if row and set(row) - {'query_id', 'reachable', 'steps'}:
            row_errors.append('unexpected route fields')
        routes[key] = {'value': row, 'valid': bool(valid), 'errors': row_errors}
    detection = localization = None
    if period == 'B':
        obj_or_empty = obj or {}
        changed = obj_or_empty.get('changed')
        loc = obj_or_empty.get('changed_pair')
        detection = {'valid': type(changed) is bool, 'value': changed}
        loc_valid = 'changed_pair' in obj_or_empty and (loc is None or (
            isinstance(loc, dict) and {'state', 'action'} <= set(loc)
            and isinstance(loc['state'], str) and isinstance(loc['action'], str)
            and loc['action'] in v['menu'].get(loc['state'], [])))
        if type(changed) is bool:
            loc_valid = loc_valid and ((not changed and loc is None) or (changed and loc is not None))
        loc_errors = ['unexpected localization fields'] if isinstance(loc, dict) and set(loc) - {'state', 'action'} else []
        localization = {'valid': bool(loc_valid), 'value': loc, 'errors': loc_errors}
    well_formed = not errors and all(r['valid'] and not r['errors'] for r in routes.values())
    if period == 'B':
        well_formed &= detection['valid'] and localization['valid'] and not localization['errors']
    return {'status': 'ok' if well_formed else 'invalid_json' if obj is None else 'partial',
            'well_formed': bool(well_formed), 'errors': errors, 'routes': routes,
            'detection': detection, 'localization': localization}


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def score_readout(parsed, w, period):
    old = {(r['node'], r['action']): r for r in w['A']['graph']}
    details = {}
    destinations, errors, valid_count, changed_count = [], [], 0, 0
    for truth in w[period]['graph']:
        key = (truth['node'], truth['action'])
        row = parsed['rows'][pair_id(key)]
        value = row['value'] or {}
        valid = row['transition_valid']
        valid_count += valid
        exact = valid and all(value[f] == truth[f] for f in TRANSITION_FIELDS)
        changed_truth = any(old[key][f] != truth[f] for f in TRANSITION_FIELDS)
        label = (row['changed_valid'] and value.get('changed') == changed_truth) if period == 'B' else None
        if valid and truth['available']:
            destinations.append(value['destination'] == truth['destination'])
            # A well-formed false availability has null p; destination is wrong,
            # but a numerical MAE is unavailable rather than invented.
            if value['p_success'] is not None:
                errors.append(abs(value['p_success'] - truth['p_success']))
        if period == 'B':
            changed_count += bool(label)
        details[pair_id(key)] = {'transition_exact': bool(exact),
                                'changed_correct': bool(label) if period == 'B' else None,
                                'joint_exact': bool(exact and label) if period == 'B' else bool(exact)}
    return {'transition_exact': fraction(sum(r['transition_exact'] for r in details.values()), 16),
            'joint_exact': fraction(sum(r['joint_exact'] for r in details.values()), 16),
            'changed_accuracy': fraction(changed_count, 16) if period == 'B' else None,
            'complete_graph_exact': parsed['complete_model'] and all(r['transition_exact'] for r in details.values()),
            'valid_rows': fraction(valid_count, 16),
            'destination_accuracy_conditional': fraction(sum(destinations), len(destinations)),
            'p_mae_truth_conditional': {'sum': sum(errors), 'n': len(errors),
                                        'value': sum(errors) / len(errors) if errors else None},
            'rows': details}


def grade_route(parsed, rows, query):
    oracle = shortest(rows, query['start'], query['goal'])
    result = {'oracle_reachable': oracle['reachable'], 'oracle_cost': oracle['cost'],
              'reachability_correct': False, 'solution_correct': False,
              'valid': False, 'optimal': False, 'cost': None, 'regret': None,
              'status': 'invalid_format', 'loops': False}
    if not parsed['valid']:
        return result
    answer = parsed['value']
    result['reachability_correct'] = answer['reachable'] == oracle['reachable']
    if not answer['reachable']:
        result.update(status='correct_no_route' if not oracle['reachable'] else 'false_unreachable',
                      solution_correct=not oracle['reachable'])
        return result
    lookup = {(r['node'], r['action']): r for r in rows}
    state, cost, seen = query['start'], 0.0, {query['start']}
    for step in answer['steps']:
        if state == query['goal']:
            result['status'] = 'extra_goal_item'
            return result
        if step['state'] != state:
            result['status'] = 'discontinuous_route'
            return result
        r = lookup.get((state, step['action']))
        if r is None or not r['available'] or r['p_success'] <= 0:
            result['status'] = 'unavailable_or_zero_probability'
            return result
        cost += 1 / r['p_success']
        state = r['destination']
        result['loops'] |= state in seen
        seen.add(state)
    if state != query['goal']:
        result['status'] = 'incomplete_route'
        return result
    result.update(status='valid', valid=True, solution_correct=True, cost=cost,
                  regret=cost - oracle['cost'], optimal=math.isclose(cost, oracle['cost'], abs_tol=1e-9))
    return result


def reported_graph(readout):
    return [{'node': r['value']['state'], **{k: r['value'][k] for k in ('action', *TRANSITION_FIELDS)}}
            for r in readout['rows'].values() if r['transition_valid']]


def route_diagnostic(route, readout, query):
    if not route['valid']:
        return {'consistency': 'not_applicable', 'own_report_optimal': None, 'own_report_no_route_correct': None}
    answer = route['value']
    rows = reported_graph(readout)
    complete = readout['complete_model']
    if not answer['reachable']:
        if not complete:
            return {'consistency': 'route_unresolvable', 'own_report_optimal': None, 'own_report_no_route_correct': None}
        no_route = not shortest(rows, query['start'], query['goal'])['reachable']
        return {'consistency': 'consistent' if no_route else 'route_inconsistent',
                'own_report_optimal': None, 'own_report_no_route_correct': no_route}
    lookup = {(r['node'], r['action']): r for r in rows}
    missing, inconsistent = False, False
    steps = answer['steps']
    inconsistent |= not steps or steps[0]['state'] != query['start']
    for i, step in enumerate(steps):
        r = lookup.get((step['state'], step['action']))
        if r is None:
            missing = True
            continue
        next_state = steps[i + 1]['state'] if i + 1 < len(steps) else query['goal']
        inconsistent |= (step['state'] == query['goal'] or not r['available']
                         or r['p_success'] <= 0 or r['destination'] != next_state)
    status = 'route_inconsistent' if inconsistent else 'route_unresolvable' if missing else 'consistent'
    own = grade_route(route, rows, query) if complete else None
    return {'consistency': status, 'own_report_optimal': own['optimal'] if own else None,
            'own_report_no_route_correct': None}


def preservation(a, b, w):
    self_values, truth_values, details = [], [], {}
    truth = {(r['node'], r['action']): r for r in w['B']['graph']}
    for control in w['controls']:
        key = (control['node'], control['action'])
        x, y = a['rows'][pair_id(key)], b['rows'][pair_id(key)]
        self_scored = x['transition_valid'] and y['transition_valid'] and y['changed_valid']
        truth_scored = y['transition_valid'] and y['changed_valid']
        same = self_scored and not y['value']['changed'] and all(
            x['value'][f] == y['value'][f] for f in TRANSITION_FIELDS)
        correct = truth_scored and not y['value']['changed'] and all(
            y['value'][f] == truth[key][f] for f in TRANSITION_FIELDS)
        if self_scored:
            self_values.append(bool(same))
        if truth_scored:
            truth_values.append(bool(correct))
        details[pair_id(key)] = {'self_scored': bool(self_scored), 'truth_scored': bool(truth_scored),
                                'self_preserved': bool(same), 'truth_preserved': bool(correct)}
    return {name: {'all_controls': fraction(sum(values), 4),
                   'conditional': fraction(sum(values), len(values)),
                   'unscored': 4 - len(values), 'all_four_correct': len(values) == 4 and all(values)}
            for name, values in (('self', self_values), ('truth', truth_values))} | {'controls': details}


def score_conversation(w, arm, turns):
    result = {'scorer': SCORER, 'periods': {},
              'explicit_model': {'status': 'pending' if arm == 'model_first' else 'not_applicable'}}
    parsed_readouts = {}
    for period in ('A', 'B'):
        task_parsed = parse_task(turns[period + '_task']['raw_response'], w, period)
        read_parsed = parse_readout(turns[period + '_readout']['raw_response'], w[period])
        parsed_readouts[period] = read_parsed
        scored = score_readout(read_parsed, w, period)
        routes = {}
        for q in queries(w):
            route = task_parsed['routes'][q['query_id']]
            routes[q['query_id']] = {'query': q, **grade_route(route, w[period]['graph'], q),
                                     **route_diagnostic(route, read_parsed, q)}
        result['periods'][period] = {'task_parsed': task_parsed, 'readout_parsed': read_parsed,
                                      'beliefs': scored, 'routes': routes,
                                      'detection_correct': None, 'localization_correct': None}
        if period == 'B':
            result['periods'][period].update(
                detection_correct=task_parsed['detection']['valid'] and task_parsed['detection']['value'] is True,
                localization_correct=task_parsed['localization']['valid'] and
                    task_parsed['localization']['value'] is not None and
                    (task_parsed['localization']['value']['state'], task_parsed['localization']['value']['action'])
                    == tuple(w['target']))
    result['preservation'] = preservation(parsed_readouts['A'], parsed_readouts['B'], w)
    return result


def manual_form(artifact, reviewer, source_file_sha256=None):
    """A blank extraction form, not fabricated reviewer work or oracle hints."""
    if artifact['identity']['condition'] != 'model_first' or reviewer not in ('Pavlos', 'Maciej'):
        raise ValueError('manual extraction is for the two model-first reviewers')
    return {'run_id': artifact['run_id'], 'reviewer': reviewer, 'complete': False,
            'source_artifact_sha256': source_file_sha256 or digest(json.dumps(artifact, sort_keys=True)),
            'artifact_hash_kind': 'file_bytes' if source_file_sha256 else 'canonical_json_synthetic_fixture',
            'source_hashes': {p: digest(artifact['turns'][p + '_model']['raw_response']) for p in ('A', 'B')},
            'periods': {p: {pair_id(key): {field: {'status': 'unknown', 'value': None, 'citations': []}
                                          for field in TRANSITION_FIELDS}
                            for key in pair_keys(artifact['world'][p])} for p in ('A', 'B')}}


def validate_manual(form, artifact, source_file_sha256=None):
    expected = manual_form(artifact, form.get('reviewer'), source_file_sha256)
    if (form.get('run_id') != expected['run_id'] or form.get('source_hashes') != expected['source_hashes']
            or form.get('source_artifact_sha256') != expected['source_artifact_sha256']
            or form.get('artifact_hash_kind') != expected['artifact_hash_kind']
            or type(form.get('complete')) is not bool or set(form.get('periods', {})) != {'A', 'B'}):
        raise ValueError('manual source identity/period mismatch')
    for period in ('A', 'B'):
        rows = form['periods'][period]
        if set(rows) != set(expected['periods'][period]):
            raise ValueError('manual review must cover all 16 pairs, including unknowns')
        for key, fields in rows.items():
            if set(fields) != set(TRANSITION_FIELDS):
                raise ValueError('manual fields must be explicit or unknown')
            for field, claim in fields.items():
                if (not isinstance(claim, dict) or set(claim) != {'status', 'value', 'citations'}
                        or claim['status'] not in ('explicit', 'unknown', 'conflict')
                        or not isinstance(claim['citations'], list)):
                    raise ValueError('invalid manual claim')
                if claim['status'] == 'explicit':
                    value = claim['value']
                    valid = (type(value) is bool if field == 'available' else
                             value is None or (isinstance(value, str) and value in artifact['world'][period]['nodes'])
                             if field == 'destination' else value is None or finite_probability(value))
                    if not valid or not claim['citations']:
                        raise ValueError('explicit manual fact needs typed value and citations')
                elif claim['value'] is not None:
                    raise ValueError('unknown/conflicting manual facts must not invent values')
                cited_periods = set()
                for cite in claim['citations']:
                    if not isinstance(cite, dict) or set(cite) != {'period', 'start', 'end', 'quote', 'source_sha256'}:
                        raise ValueError('invalid citation shape')
                    p = cite['period']
                    if p not in (('A',) if period == 'A' else ('A', 'B')):
                        raise ValueError('future/unknown source citation')
                    raw = artifact['turns'][p + '_model']['raw_response']
                    if (type(cite['start']) is not int or type(cite['end']) is not int
                            or not 0 <= cite['start'] < cite['end'] <= len(raw)
                            or raw[cite['start']:cite['end']] != cite['quote']
                            or cite['source_sha256'] != digest(raw)):
                        raise ValueError('manual source span/hash mismatch')
                    cited_periods.add(p)
                if period == 'B' and 'A' in cited_periods and 'B' not in cited_periods:
                    raise ValueError('inheritance needs both the B rule and original A citation')
    return form


def manual_results(artifact, reviews, resolutions=None, source_file_sha256=None):
    """Resolve by reviewed citations, never by reading the truth to fill gaps.

    A resolution selects an existing reviewer's claim or leaves a disagreement
    unknown. The validator checks provenance, not whether a quote entails a fact.
    """
    if artifact['identity']['condition'] != 'model_first':
        return {'status': 'not_applicable'}
    checked = [validate_manual(r, artifact, source_file_sha256) for r in reviews]
    if len({r['reviewer'] for r in checked}) != len(checked):
        raise ValueError('duplicate reviewer')
    if len(checked) != 2 or not all(r['complete'] for r in checked):
        return {'status': 'pending', 'review_count': len(checked), 'scores': None}
    resolutions = resolutions or {}
    used, disagreements, output = set(), [], {}
    for period in ('A', 'B'):
        parsed = {'rows': {}, 'complete_model': True}
        coverage_fields = 0
        for key in pair_keys(artifact['world'][period]):
            value = {'state': key[0], 'action': key[1]}
            all_explicit = True
            for field in TRANSITION_FIELDS:
                claims = [r['periods'][period][pair_id(key)][field] for r in checked]
                chosen = claims[0]
                if (claims[0]['status'], claims[0]['value']) != (claims[1]['status'], claims[1]['value']):
                    issue = f'{period}/{pair_id(key)}/{field}'
                    selection = resolutions.get(issue)
                    disagreements.append({'field': issue, 'resolution': selection, 'claims': claims})
                    if selection is not None:
                        if set(selection) != {'reviewer', 'note'} or not selection['note']:
                            raise ValueError('resolution needs reviewer selection and note')
                        used.add(issue)
                        if selection['reviewer'] not in ('Pavlos', 'Maciej', 'unknown'):
                            raise ValueError('resolution cannot invent new facts')
                        chosen = next((c for r, c in zip(checked, claims) if r['reviewer'] == selection['reviewer']),
                                      {'status': 'unknown', 'value': None})
                    else:
                        chosen = {'status': 'unknown', 'value': None}
                explicit = chosen['status'] == 'explicit'
                coverage_fields += explicit
                all_explicit &= explicit
                value[field] = chosen['value'] if explicit else None
            valid = all_explicit and transition_valid(value, artifact['world'][period]['nodes'])
            parsed['rows'][pair_id(key)] = {'value': value, 'transition_valid': bool(valid), 'changed_valid': False}
            parsed['complete_model'] &= bool(valid)
        truth = {(r['node'], r['action']): r for r in artifact['world'][period]['graph']}
        valid_rows = [r for r in parsed['rows'].values() if r['transition_valid']]
        exact = sum(all(r['value'][f] == truth[(r['value']['state'], r['value']['action'])][f]
                        for f in TRANSITION_FIELDS) for r in valid_rows)
        tasks = parse_task(artifact['turns'][period + '_task']['raw_response'], artifact['world'], period)
        output[period] = {'field_coverage': fraction(coverage_fields, 48), 'row_coverage': fraction(len(valid_rows), 16),
                          'exact_all_rows': fraction(exact, 16), 'exact_conditional': fraction(exact, len(valid_rows)),
                          'routes': {q['query_id']: route_diagnostic(tasks['routes'][q['query_id']], parsed, q)
                                     for q in queries(artifact['world'])}}
    if set(resolutions) != used:
        raise ValueError('unknown or unnecessary adjudication entry')
    return {'status': 'review_complete', 'disagreements': disagreements, 'scores': output,
            'basis': 'human extracted, span-checked; entailment not automatically verified'}
