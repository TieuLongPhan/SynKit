from copy import deepcopy
from fractions import Fraction

import pytest

from Experiment.Synister.revision_diagnostics import (
    EVIDENCE, build, concentration, disposition, minimum_multiplicity, query_inventory,
)


def fixture():
    cases = [{'benchmark_id': name, 'source': 'FlowER', 'atoms': 12, 'size_bin': 0}
             for name in ('a', 'b')]
    records, groups = [], []
    for case in cases:
        name = case['benchmark_id']
        # No proof on b: dependent targets must remain unavailable, not empty.
        queries = [('minimum', 'minimal')] + ([('at_minimum', 4), ('plus_1', 6),
                   ('plus_2', 8), ('plus_4', 12)] if name == 'a' else [])
        for query, target in queries:
            key = f'{name}.{query}.synister'
            complete = query == 'at_minimum'
            records.append({'task': {'task_id': key, 'benchmark_id': name, 'query': query,
                                     'method': 'synister', 'target_doubled_cd': target},
                            'complete': complete, 'minimum_proved': name == 'a' and query == 'minimum',
                            'minimum_doubled_cd': 4 if name == 'a' else None,
                            'mapping_count': 2 if complete else None, 'parent_seconds': 1,
                            'termination': 'complete' if complete else 'time_limit'})
        aliases = [{'task_id': r['task']['task_id'], 'query': r['task']['query']}
                   for r in records if r['task']['benchmark_id'] == name]
        groups.append({'task_id': name+'.cd_4', 'benchmark_id': name,
                       'query_aliases': aliases, 'status': 'verified_exact' if name == 'a' else 'enumeration_incomplete',
                       'source_task_id': 'a.at_minimum.synister' if name == 'a' else None,
                       'indexed_maps': 2 if name == 'a' else None,
                       'product_orbits': 1 if name == 'a' else None, 'its_classes': 1 if name == 'a' else None})
    return {'manifest': {'methods': ['synister']}, 'inputs': cases, 'records': records}, groups


def test_later_complete_set_never_changes_original_minimum_outcome():
    study, groups = fixture()
    rows = query_inventory(study, groups)
    assert len(rows) == 12
    original = next(r for r in rows if r['task_id'] == 'a.minimum.synister')
    assert not original['complete'] and original['minimum_proved']
    assert original['disposition'] == 'search_incomplete'
    assert original['classification_status'] == 'verified_exact'
    assert original['classification_source_task_id'] == 'a.at_minimum.synister'
    assert sum(r['disposition'] == 'minimum_dependent_target_unavailable' for r in rows) == 4
    assert sum(r['disposition'] == 'reference_target_unavailable' for r in rows) == 2
    result = minimum_multiplicity(study['inputs'], groups)
    assert result['counts'] == {'multiple_maps_one_its': 1, 'structural_result_unavailable': 1}
    assert result['rows'][1]['indexed_maps'] is None


def test_missing_required_attempts_and_duplicate_aliases_fail_closed():
    study, groups = fixture()
    groups[0]['query_aliases'].append(deepcopy(groups[0]['query_aliases'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        query_inventory(study, groups)
    study, groups = fixture()
    study['records'].pop(1)
    groups[0]['query_aliases'].pop(1)
    with pytest.raises(ValueError, match='Missing required'):
        query_inventory(study, groups)
    study, groups = fixture()
    groups[0]['query_aliases'].pop(0)
    with pytest.raises(ValueError, match='cover all'):
        query_inventory(study, groups)


def test_relative_targets_require_correct_proven_minimum():
    study, groups = fixture()
    study['records'][1]['task']['target_doubled_cd'] = 5
    with pytest.raises(ValueError, match='proved minimum'):
        query_inventory(study, groups)


def test_empty_precheck_searched_empty_and_censored_are_distinct():
    assert disposition({'complete': False, 'mapping_count': 0}) == 'search_incomplete'
    assert disposition({'complete': True, 'mapping_count': 0}) == 'searched_empty'
    assert disposition({'complete': True, 'mapping_count': 3}) == 'enumerated_nonempty'
    assert disposition({'complete': True, 'mapping_count': 0,
                        'termination': 'proved_empty_precheck'}) == 'elementary_impossible_target'
    with pytest.raises(ValueError):
        disposition({'complete': True, 'mapping_count': None})


def test_unverified_structure_never_becomes_an_exact_minimum_count():
    study, groups = fixture()
    groups[0]['status'] = 'verified_bounds'
    result = minimum_multiplicity(study['inputs'], groups)
    assert result['counts'] == {'structural_result_unavailable': 2}
    assert all(r['its_classes'] is None for r in result['rows'])


def test_concentration_preserves_zero_width_and_unresolved_weight():
    rows = [dict(reaction_id='a', lower='-1/2', upper='1/2', width='1'),
            dict(reaction_id='b', lower='0', upper='0', width='0'),
            dict(reaction_id='c', lower='0', upper='1/4', width='1/4')]
    result = concentration(rows, 4)
    assert result['reactions_for_fraction_of_resolved_width'] == {'1/2': 1, '4/5': 1, '9/10': 2}
    assert result['positive_width'] == 2 and len(result['ranked_rows']) == 3
    assert Fraction(result['resolved_contribution_to_cohort_width']) == Fraction(5, 16)
    assert Fraction(result['unresolved_contribution_to_outer_width']) == Fraction(1, 2)
    assert concentration(rows[1:2], 2)['reactions_for_fraction_of_resolved_width']['1/2'] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        concentration(rows+rows, 6)
    rows[0]['width'] = '2'
    with pytest.raises(ValueError, match='Invalid comparison'):
        concentration(rows, 4)


def test_saved_archive_denominators_and_json_roundtrip():
    import json
    from collections import Counter
    result = build()
    assert json.loads(json.dumps(result)) == result
    rows = result['query_inventory']
    counts = Counter(r['disposition'] for r in rows)
    assert len(rows) == 1200
    assert counts['minimum_dependent_target_unavailable'] == 424
    assert counts['reference_target_unavailable'] == 100
    assert len(rows)-424-100 == 676
    assert result['minimum_multiplicity']['counts'] == {
        'one_indexed_map': 9, 'multiple_maps_one_its': 24,
        'multiple_its': 14, 'structural_result_unavailable': 53}
    # Independent aggregation of original files, not the reporter's grouping.
    cases = {r['benchmark_id']: r for r in json.loads((EVIDENCE/'enumeration_main_v1/inputs.json').read_text())}
    for method in ('synister', 'milp'):
        raw = [json.loads(p.read_text()) for p in
               (EVIDENCE/'enumeration_main_v1/cases').glob(f'*.minimum.{method}.json')]
        for b in range(5):
            selected = [r for r in raw if cases[r['task']['benchmark_id']]['size_bin'] == b]
            observed = next(r for r in result['original_minimum_by_source_size']
                            if r['method'] == method and r['size_bin'] == b and r['source'] == 'all')
            assert observed['attempts'] == len(selected) == 20
            assert observed['complete'] == sum(r['complete'] for r in selected)
            assert observed['minimum_proved'] == sum(r.get('minimum_proved', False) for r in selected)
