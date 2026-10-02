"""Independently replay the small comparator oracle from archived endpoints."""

from itertools import combinations, permutations
import json
from pathlib import Path

from Experiment.Synister.audit_published_comparator import COMMIT, CONTROLS, digest


ROOT = Path(__file__).resolve().parents[3]


def test_published_controls_reconstruct_literal_sets_and_known_incompatibilities():
    report = json.loads((ROOT/'paper/synister/evidence/published_comparator_compatibility_v1.json').read_text())
    assert report['commit'] == COMMIT
    assert report['auditor_sha256'] == digest(ROOT/'Experiment/Synister/audit_published_comparator.py')
    assert [r['reaction'] for r in report['rows']] == CONTROLS
    agreement = {'binary': 0, 'bond_order': 0}
    for row in report['rows']:
        reactant, product = (row['endpoints'][k] for k in ('reactant', 'product'))
        r, p = [{tuple(pair): weight for pair, weight in mol['bonds']} for mol in (reactant, product)]
        scores = {'binary': {}, 'bond_order': {}}
        for images in permutations(product['atoms']):
            mapping = dict(zip(reactant['atoms'], images))
            if any(reactant['atomic_numbers'][str(i)] != product['atomic_numbers'][str(j)]
                   for i, j in mapping.items()):
                continue
            before_after = [(r.get((i, j), 0), p.get(tuple(sorted((mapping[i], mapping[j]))), 0))
                            for i, j in combinations(reactant['atoms'], 2)]
            scores['binary'][images] = sum((a != 0) != (b != 0) for a, b in before_after)
            scores['bond_order'][images] = sum(abs(a-b) for a, b in before_after)
        for name, costs in scores.items():
            minimum = min(costs.values())
            expected = {m for m, cost in costs.items() if cost == minimum}
            assert row['literal'][name]['minimum'] == minimum
            assert set(map(tuple, row['literal'][name]['optimal_maps'])) == expected
            for route in row['routes'].values():
                assert route['clique_sets_agree_with_networkx']
                for mode in ('unfiltered_indexed', 'filter1', 'filter2'):
                    result = route.get(mode, {})
                    if 'by_objective' not in result:
                        continue
                    maps = set(map(tuple, result['valid_maps']))
                    audit = result['by_objective'][name]
                    assert set(map(tuple, audit['missing_optimal_maps'])) == expected-maps
                    assert set(map(tuple, audit['nonoptimal_maps'])) == maps-expected
                    assert audit['complete_sets_equal'] == (not result['invalid_maps'] and maps == expected)
            published = row['routes']['published_readme']['unfiltered_indexed']['by_objective'][name]
            agreement[name] += published['complete_sets_equal']
    assert agreement == {'binary': 10, 'bond_order': 8}
    lookup = {r['reaction']: r for r in report['rows']}
    assert lookup['CO>>CO']['routes']['published_readme']['unfiltered_indexed']['unique_valid_count'] == 0
    assert lookup['CO>>CO']['routes']['retain_isolated_nodes_diagnostic']['unfiltered_indexed']['unique_valid_count'] == 1
    assert lookup['CC>>CC']['routes']['retain_isolated_nodes_diagnostic']['error'] == 'ValueError'
