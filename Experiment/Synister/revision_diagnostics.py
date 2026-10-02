"""E0: reconstruct denominators and descriptive limits without new searches.

Original minimum attempts, later numeric attempts, and downstream classification
are deliberately separate. All six nominal query slots have a disposition.
"""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from Experiment.Synister.enumeration_benchmark import size_bin
from Experiment.Synister.report_enumeration_performance import (
    compact, digest, load_study, summarize,
)
from Experiment.Synister.report_enumeration_structure import build as structure_report


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'paper/synister/evidence'
QUERIES = ('minimum', 'at_minimum', 'plus_1', 'plus_2', 'plus_4', 'reference_cd')
BIN_LABELS = ('1–20', '21–40', '41–60', '61–80', '>80')


def read(path):
    return json.loads(path.read_text())


def disposition(record):
    if not record['complete']:
        return 'search_incomplete'
    if record.get('termination') == 'proved_empty_precheck':
        if record.get('mapping_count') != 0:
            raise ValueError('Precheck claims nonempty output')
        return 'elementary_impossible_target'
    if record.get('mapping_count') is None:
        raise ValueError('Completed attempt lacks a map count')
    return 'searched_empty' if record['mapping_count'] == 0 else 'enumerated_nonempty'


def query_inventory(study, structural_rows):
    """Require exactly one classification alias for every actual solver attempt."""
    aliases = {}
    for row in structural_rows:
        for alias in row['query_aliases']:
            key = alias['task_id']
            if key in aliases:
                raise ValueError('Duplicate structural query alias')
            aliases[key] = row
    attempts = {r['task']['task_id']: r for r in study['records']}
    if set(aliases) != set(attempts):
        raise ValueError('Structural aliases do not cover all solver attempts')
    methods = study['manifest']['methods']
    rows = []
    for case in study['inputs']:
        key = case['benchmark_id']
        proofs = {r['minimum_doubled_cd'] for r in study['records']
                  if r['task']['benchmark_id'] == key
                  and r['task']['query'] == 'minimum' and r.get('minimum_proved')}
        if len(proofs) > 1:
            raise ValueError('Conflicting minimum proofs')
        for query in QUERIES:
            for method in methods:
                task_id = f'{key}.{query}.{method}'
                record = attempts.get(task_id)
                row = {k: case[k] for k in ('benchmark_id', 'source', 'atoms', 'size_bin')}
                row.update(query=query, method=method, task_id=task_id,
                           minimum_proved_by_any_original_attempt=bool(proofs))
                if record is None:
                    if query == 'minimum' or query != 'reference_cd' and proofs:
                        raise ValueError('Missing required query attempt')
                    row.update(disposition=('reference_target_unavailable' if query == 'reference_cd'
                                            else 'minimum_dependent_target_unavailable'),
                               classification_status='not_attempted', complete=False,
                               parent_seconds=None, mapping_count=None,
                               minimum_proved=False, target_doubled_cd=None)
                else:
                    if query not in ('minimum', 'reference_cd'):
                        offset = {'at_minimum': 0, 'plus_1': 2, 'plus_2': 4, 'plus_4': 8}[query]
                        if not proofs or record['task']['target_doubled_cd'] != next(iter(proofs))+offset:
                            raise ValueError('Numeric query differs from its proved minimum')
                    structural = aliases[task_id]
                    row.update(compact(record), disposition=disposition(record),
                               classification_task_id=structural['task_id'],
                               classification_status=structural['status'],
                               classification_source_task_id=structural['source_task_id'])
                rows.append(row)
    if {r['task_id'] for r in rows if r['disposition'] not in (
            'reference_target_unavailable', 'minimum_dependent_target_unavailable')} != set(attempts):
        raise ValueError('Unrecognized query outside declared inventory')
    return rows


def minimum_multiplicity(inputs, structural_rows):
    result = []
    for case in inputs:
        candidates = [r for r in structural_rows if r['benchmark_id'] == case['benchmark_id']
                      and any(a['query'] == 'minimum' for a in r['query_aliases'])]
        # An unproved minimum may have its own unresolved group alongside the
        # numeric group at the minimum proved by the other implementation.
        exact = [r for r in candidates if r['status'] == 'verified_exact']
        if len(exact) > 1:
            raise ValueError('Duplicate verified minimum group')
        row = {k: case[k] for k in ('benchmark_id', 'source', 'atoms', 'size_bin')}
        row.update(category='structural_result_unavailable', indexed_maps=None,
                   product_orbits=None, its_classes=None, source_task_id=None)
        if exact:
            item = exact[0]
            if not item['indexed_maps'] or not item['its_classes']:
                raise ValueError('Proved minimum set is empty')
            row.update({k: item[k] for k in ('indexed_maps', 'product_orbits', 'its_classes', 'source_task_id')})
            row['category'] = ('one_indexed_map' if item['indexed_maps'] == 1 else
                               'multiple_maps_one_its' if item['its_classes'] == 1 else 'multiple_its')
        result.append(row)
    return {'rows': result, 'counts': dict(Counter(r['category'] for r in result)),
            'by_source': {s: dict(Counter(r['category'] for r in result if r['source'] == s))
                          for s in sorted({r['source'] for r in result})},
            'scope': 'One reaction each, using all saved attempts at a proved minimum; not original minimum-query success.'}


def concentration(rows, denominator):
    """Exact rational contribution accounting; unresolved weight stays explicit."""
    if denominator <= 0 or len(rows) > denominator:
        raise ValueError('Invalid comparison denominator')
    if len({r['reaction_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate reaction in comparison')
    ordered = sorted(rows, key=lambda r: (-Fraction(r['width']), r['reaction_id']))
    total = sum((Fraction(r['width']) for r in ordered), Fraction())
    cumulative, curve = Fraction(), []
    for i, row in enumerate(ordered, 1):
        lo, hi, width = (Fraction(row[k]) for k in ('lower', 'upper', 'width'))
        if not -1 <= lo <= hi <= 1 or hi-lo != width:
            raise ValueError('Invalid comparison interval')
        cumulative += width
        curve.append({**row, 'rank': i, 'cumulative_width': str(cumulative),
                      'fraction_resolved_width': str(cumulative/total) if total else None})
    thresholds = {}
    for q in ('1/2', '4/5', '9/10'):
        thresholds[q] = next((r['rank'] for r in curve
                              if Fraction(r['cumulative_width']) >= Fraction(q)*total), None) if total else 0
    return {'common_valid_predictions': denominator, 'resolved': len(rows),
            'unresolved': denominator-len(rows), 'positive_width': sum(Fraction(r['width']) > 0 for r in rows),
            'resolved_width_sum': str(total), 'resolved_contribution_to_cohort_width': str(total/denominator),
            'unresolved_contribution_to_outer_width': str(Fraction(2*(denominator-len(rows)), denominator)),
            'reactions_for_fraction_of_resolved_width': thresholds, 'ranked_rows': curve,
            'scope': 'Descriptive concentration of saved envelope width, not a selected reference convention or chemical prevalence.'}


def primary_records(evidence, tag):
    directory = evidence / f'identifiability_{tag}_primary_v1'
    manifest, audit = read(directory/'manifest.json'), read(directory/'audit.json')
    resources = read(evidence/f'{tag}_resources_v1.json')
    for name in ('manifest', 'summary'):
        if digest(directory/f'{name}.json') != audit[f'{name}_sha256']:
            raise ValueError('Primary archive changed after audit')
    if resources['primary_audit_sha256'] != digest(directory/'audit.json'):
        raise ValueError('Primary resource inventory refers to another audit')
    if digest(directory/'inputs.json') != manifest['inputs_sha256']:
        raise ValueError('Primary input identity changed')
    if set(resources['record_hashes']) != {p.name for p in (directory/'cases').iterdir()}:
        raise ValueError('Primary record inventory changed')
    for name, expected in resources['record_hashes'].items():
        if digest(directory/'cases'/name) != expected:
            raise ValueError('Primary attempt changed after resource inventory')
    inputs = read(directory/'inputs.json')
    rows = []
    for case in inputs:
        exact = read(directory/'cases'/f'{case["case_id"]}.exact.json')
        if exact['case_id'] != case['case_id']:
            raise ValueError('Primary attempt identity mismatch')
        n = exact.get('numerical_domain', {}).get('heavy_atoms')
        if n is None:
            from synkit.Chem.Mapper.identifiability import parse_reaction
            n = len(parse_reaction(case['reaction'])[0].atomic_numbers)
        rows.append({'case_id': case['case_id'], 'reaction_id': case['reaction_id'],
                     'atoms': n, 'size_bin': size_bin(n), 'status': exact['status'],
                     'complete': exact['status'] == 'complete',
                     'minimum_proved': exact.get('minimum_proved', False),
                     'parent_seconds': exact.get('parent_seconds'),
                     'output_scope': exact.get('mapping_scope'),
                     'emitted_representatives': exact.get('emitted_representatives')})
    return {'manifest': manifest, 'rows': rows,
            'ranking_concentration': concentration(audit['resolved_rows'], audit['counts']['common_valid_predictions']),
            'source_hashes': {str(p.relative_to(evidence)): digest(p) for p in
                              (directory/'audit.json', directory/'manifest.json', directory/'inputs.json',
                               evidence/f'{tag}_resources_v1.json')}}


def policies(evidence):
    directory = evidence/'identifiability_c1_annotations_v1'
    audit = read(directory/'audit.json')
    for name in ('manifest', 'summary'):
        if digest(directory/f'{name}.json') != audit[f'{name}_sha256']:
            raise ValueError('Policy archive changed after audit')
    inputs = read(directory/'inputs.json')
    if digest(directory/'inputs.json') != read(directory/'manifest.json')['inputs_sha256']:
        raise ValueError('Policy input identity changed')
    names = {'canonical_its': 'Least canonical full attributed ITS code; independent of predictor scores.',
             'nearest_a': 'Maximize SLAP bond F1, tie break by canonical ITS code; method-favouring diagnostic.',
             'nearest_b': 'Maximize RXNMapper bond F1, tie break by canonical ITS code; method-favouring diagnostic.'}
    rows, hashes = [], {}
    for case in inputs:
        path = directory/'cases'/f'{case["case_id"]}.annotations.json'
        record = read(path)
        hashes[str(path.relative_to(evidence))] = digest(path)
        if record['case_id'] != case['case_id']:
            raise ValueError('Policy attempt identity mismatch')
        for name in names:
            item = record.get('policies', {}).get(name, {})
            value = item.get('difference')
            if value is not None and Fraction(value) != Fraction(item['a_score'])-Fraction(item['b_score']):
                raise ValueError('Policy difference disagrees with scores')
            rows.append({'case_id': case['case_id'], 'policy': name, 'difference': value})
    summaries = {}
    for name, definition in names.items():
        values = [Fraction(r['difference']) for r in rows if r['policy'] == name and r['difference'] is not None]
        n, total = len(values), len(inputs)
        summed = sum(values, Fraction())
        mean = str(summed/n) if n else None
        bounds = [str(summed/total-Fraction(total-n, total)), str(summed/total+Fraction(total-n, total))]
        if (n != audit['policy_record_counts'][name]
                or mean != audit['policy_reported_conditional_means'][name]
                or bounds != audit['policy_reported_outer_bounds'][name]):
            raise ValueError('Recomputed policy totals disagree with audit')
        summaries[name] = {'definition': definition, 'selected': total, 'resolved': n,
                           'unresolved': total-n, 'conditional_mean': mean, 'outer_bounds': bounds}
    return {'scope': 'Existing predeclared C1 common-reference policies. Neither method-nearest policy is neutral chemical evidence.',
            'policies': summaries, 'rows': rows, 'source_hashes': hashes}


def build(evidence=EVIDENCE):
    matched = load_study(evidence/'enumeration_main_v1', 'matched')
    structural = structure_report(evidence/'enumeration_classification_v1', evidence/'enumeration_main_v1')
    inventory = query_inventory(matched, structural['rows'])
    inputs = {r['benchmark_id']: r for r in matched['inputs']}
    minimum = [r for r in matched['records'] if r['task']['query'] == 'minimum']
    grouped = []
    for source in ('all', *sorted({r['source'] for r in inputs.values()})):
        for b, label in enumerate(BIN_LABELS):
            for method in matched['manifest']['methods']:
                selected = [r for r in minimum if r['task']['method'] == method
                            and inputs[r['task']['benchmark_id']]['size_bin'] == b
                            and (source == 'all' or inputs[r['task']['benchmark_id']]['source'] == source)]
                ids = {r['task']['benchmark_id'] for r in selected}
                groups = [r for r in structural['rows'] if r['benchmark_id'] in ids
                          and any(a['query'] == 'minimum' for a in r['query_aliases'])]
                grouped.append({'source': source, 'size_bin': b, 'heavy_atoms': label,
                                'method': method, **summarize(selected),
                                'verified_minimum_classifications_any_attempt': len({r['benchmark_id'] for r in groups
                                                                                  if r['status'] == 'verified_exact'})})
    by_query = []
    for query in QUERIES:
        for method in matched['manifest']['methods']:
            slots = [r for r in inventory if r['query'] == query and r['method'] == method]
            attempts = [r for r in matched['records'] if r['task']['query'] == query and r['task']['method'] == method]
            by_query.append({'query': query, 'method': method, 'selected_reactions': len(inputs),
                             'dispositions': dict(Counter(r['disposition'] for r in slots)), **summarize(attempts)})
    primary = {tag: primary_records(evidence, tag) for tag in ('c1', 'c2')}
    c1 = primary['c1']
    settings = [
        {'study': 'C1', 'selected': len(c1['rows']), 'sources': ['FlowER'],
         'size_bins': dict(Counter(str(r['size_bin']) for r in c1['rows'])),
         'seed': c1['manifest']['settings']['seed_policy'],
         'subgroup': 'Verified product automorphisms preserving element, charge and hydrogen count; full group not guaranteed.',
         'output': 'Collected representatives; one witness per fixed changed-bond and joint label. No indexed expansion.',
         'search_seconds': c1['manifest']['settings']['search_seconds'], 'parent_seconds_limit': 65,
         'max_mappings': 100000, 'cap_unit': 'representatives',
         'timing': 'Parent process includes startup, parse, seeded collected minimum search, label construction and export; excludes frozen predictions and subsequent scoring.'},
        {'study': 'Main matched benchmark', 'selected': len(inputs), 'sources': ['FlowER', 'Rhea'],
         'size_bins': dict(Counter(str(r['size_bin']) for r in inputs.values())),
         'seed': matched['manifest']['seed'],
         'subgroup': 'Synister: fully enumerated verified cyclic product subgroup for indexed expansion; MILP: indexed assignments.',
         'output': matched['manifest']['output_unit'], 'search_seconds': matched['manifest']['seconds'],
         'parent_seconds_limit': matched['manifest']['external_seconds'], 'max_mappings': matched['manifest']['max_maps'],
         'cap_unit': 'indexed maps',
         'timing': 'Parent process includes startup, parse, minimum proof, enumeration, expansion, integer checks, sorting and writing; excludes subsequent ITS classification.'},
    ]
    return {'schema': 'synister.revision-diagnostics.v1', 'generator_sha256': digest(Path(__file__)),
            'semantics': {'completion': 'Terminal recorded elapsed times at fixed resource limits, not independent budget reruns. Incomplete attempts remain in every denominator.',
                          'classification': 'Classification may use another complete attempt at the same proved CD; this never changes an original solver outcome.',
                          'unavailable': 'Nominal slots with no constructible target are not solver attempts or proved-empty sets.',
                          'resources': 'Missing remains null. Success-only timings are explicitly conditional. Repeated runs are not pooled.',
                          'causality': 'Settings comparison is descriptive; E2 is required to attribute seed, input, subgroup and output effects.'},
            'upstream': {'matched': matched['provenance'], 'structure': structural['provenance']},
            'original_minimum_by_source_size': grouped, 'by_query': by_query,
            'query_inventory': inventory, 'minimum_multiplicity': minimum_multiplicity(matched['inputs'], structural['rows']),
            'study_settings': settings, 'primary_studies': primary, 'reference_policies': policies(evidence)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'minimum_multiplicity': result['minimum_multiplicity']['counts'],
                      'nominal_query_slots': len(result['query_inventory'])}))
