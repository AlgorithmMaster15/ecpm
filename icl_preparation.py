"""Conservative extraction of explicitly stated preparation transitions.

This is not a free-text quality judge. Unsupported or conflicting statements
are marked for review, never silently scored as absent or correct.
"""

import json
import math
import re


def words(text):
    return len(text.split())


def extract(text, view):
    known = {(r['node'], r['action']) for r in view['graph']}
    states = set(view['nodes'])
    candidates, review, global_defaults = {}, [], set()

    def add(state, action, destination, probability, quote, global_default=False):
        if not isinstance(state, str) or not isinstance(action, str):
            review.append(quote)
            return
        key = state, action
        if key not in known:
            return
        valid_p = (probability is None or type(probability) in (int, float)
                   and math.isfinite(probability) and 0 <= probability <= 1)
        if (destination is not None and (not isinstance(destination, str) or destination not in states)) or not valid_p:
            review.append(quote)
            return
        row = dict(state=state, action=action, destination=destination,
                   p_success=probability, quote=quote)
        if key in candidates and candidates[key]['destination'] == destination:
            # A later destination-only reminder does not contradict an earlier
            # explicit probability. A later explicit value fills a missing one.
            if probability is None:
                return
            if candidates[key]['p_success'] is None:
                candidates[key] = row
                if global_default:
                    global_defaults.add(key)
                else:
                    global_defaults.discard(key)
                return
        if key in candidates and any(candidates[key][k] != row[k]
                                    for k in ('destination', 'p_success')):
            review.append(quote)
        else:
            # An explicit row must not later be overwritten by a global rule.
            if global_default and key not in candidates:
                global_defaults.add(key)
            elif not global_default:
                global_defaults.discard(key)
            candidates[key] = row

    # JSON and Markdown explicitly bind a state, action, destination and value.
    value = None
    try:
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', stripped)
        value = json.loads(stripped)
    except ValueError:
        pass
    if isinstance(value, dict):
        value = value.get('pairs', value.get('transitions'))
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                add(row.get('state', row.get('node')), row.get('action'),
                    row.get('destination'), row.get('p_success'), json.dumps(row, sort_keys=True))

    columns = None
    table = False
    aliases = {'state': 'state', 'node': 'state', 'action': 'action',
               'destination': 'destination', 'success destination': 'destination',
               'p_success': 'p_success', 'success probability': 'p_success',
               'estimated p': 'p_success', 'period b p_success': 'p_success',
               'next state': 'destination', 'next_state': 'destination',
               'p': 'p_success', 'probability': 'p_success'}
    clean = text.replace('\\(', '').replace('\\)', '').replace('\\to', '→')
    clean = clean.replace('p_{\\text{success}}', 'p_success').replace('**', '').replace('`', '')
    for line in clean.splitlines():
        if not line.lstrip().startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        labels = [aliases.get(c.lower()) for c in cells]
        if all(k in labels for k in ('state', 'action', 'destination')):
            table = True
            if 'p_success' not in labels:
                review.append(line)
        if all(k in labels for k in ('state', 'action', 'destination', 'p_success')):
            columns, table = labels, True
            continue
        if columns and len(columns) == len(cells):
            r = dict(zip(columns, cells))
            if (r.get('state'), r.get('action')) in known:
                try:
                    probability = float(r['p_success']) if r['p_success'] != 'null' else None
                except ValueError:
                    review.append(line)
                    continue
                add(r['state'], r['action'], None if r['destination'] == 'null' else r['destination'], probability, line)

    # Grouped lists are also models. Global p=1 is used only when explicitly
    # stated for all actions. Merely observing success does not supply a value.
    global_one = bool(re.search(
        r'(?:each|every|all)[^\n]{0,120}success probabilit(?:y|ies)[^\n]{0,45}(?:is|are|of|\s)\s*1(?:\.0+)?(?![\w]|\.[\d.])', clean, re.I))
    current = None
    for line in clean.splitlines():
        state = re.match(r'\s*[-*]?\s*([A-Za-z][\w]*):', line)
        if state and state[1] in states:
            current = state[1]
        for match in re.finditer(r'\b(a\d+)\s*(?:→|->)\s*([A-Za-z]\w*)', line):
            if current is None:
                review.append(line)
                continue
            p = re.match(r'\s*[,;]?\s*\(?p\s*=\s*([0-9.]+)', line[match.end():])
            try:
                probability = float(p[1]) if p else 1.0 if global_one else None
            except ValueError:
                review.append(line)
                continue
            add(current, match[1], match[2], probability, line,
                global_default=p is None and global_one)
    # An action-specific probability can refine a global rule. It cannot
    # silently overwrite a conflicting explicit row. Preserve the quote.
    for line in clean.splitlines():
        pair = re.search(r'\b([A-Za-z]\w*)\s*:\s*(a\d+)\b', line)
        probability = re.search(
            r'(?:success probability|probability)[^\n]{0,25}(?:is|therefore|of)\s*'
            r'([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)(?![\w]|\.[\d.])', line, re.I)
        if pair and (pair[1], pair[2]) in candidates and probability:
            key = pair[1], pair[2]
            row = candidates[key]
            value = float(probability[1])
            if not math.isfinite(value) or not 0 <= value <= 1:
                review.append(line)
            elif key in global_defaults or row['p_success'] is None:
                row['p_success'] = value
                global_defaults.discard(key)
            elif row['p_success'] != value:
                review.append(line)
            row['quote'] += '\n' + line

    # Mentions outside supported rows are unresolved, not guessed. This flag
    # prevents a prose model from becoming a false zero in scientific exports.
    mentioned = {k for k in known if re.search(r'\b' + re.escape(k[1]) + r'\b', clean)}
    unresolved = len(candidates) < len(known) and bool(mentioned)
    truth = {(r['node'], r['action']): r for r in view['graph']}
    if candidates and any(r['p_success'] is None and truth[(r['state'], r['action'])]['available'] for r in candidates.values()):
        unresolved = True
    return dict(table_present=table, transition_content=True if candidates else None if unresolved or review else False,
                status='needs_review' if review or unresolved else 'explicit_extraction',
                rows=list(candidates.values()), review_fragments=review)


def score(text, view, tolerance=0):
    result = extract(text, view)
    truth = {(r['node'], r['action']): r for r in view['graph']}
    rows = result['rows']
    result.update(word_count=words(text), length_target=250,
                  length_target_met=200 <= words(text) <= 300,
                  reported_pairs=len(rows), required_pairs=len(truth))
    details, errors = [], []
    for r in rows:
        expected = truth[(r['state'], r['action'])]
        probability_known = r['p_success'] is not None and expected['p_success'] is not None
        absent_correct = not expected['available'] and r['destination'] is None and r['p_success'] is None
        error = abs(r['p_success'] - expected['p_success']) if probability_known else None
        if error is not None:
            errors.append(error)
        details.append(r | dict(destination_correct=r['destination'] == expected['destination'],
            probability_error=error, pair_exact=absent_correct or probability_known and r['destination'] == expected['destination'] and error == 0,
            pair_within_tolerance=absent_correct or probability_known and r['destination'] == expected['destination'] and error <= tolerance + 1e-12))
    result['rows'] = details
    result['scored'] = result['status'] == 'explicit_extraction'
    # Unsupported prose is excluded pending extraction review, rather than
    # scored as a failure. The unreviewed count must accompany every table.
    result['pair_exact'] = dict(numerator=sum(r['pair_exact'] for r in details) if result['scored'] else None,
        denominator=len(truth) if result['scored'] else 0)
    result['pair_tolerance'] = dict(numerator=sum(r['pair_within_tolerance'] for r in details) if result['scored'] else None,
        denominator=len(truth) if result['scored'] else 0)
    result['destination_correct'] = dict(numerator=sum(r['destination_correct'] for r in details) if result['scored'] else None,
        denominator=len(truth) if result['scored'] else 0)
    result['p_mae_truth'] = dict(sum=sum(errors) if result['scored'] else 0, n=len(errors) if result['scored'] else 0,
        value=sum(errors)/len(errors) if errors and result['scored'] else None)
    return result
