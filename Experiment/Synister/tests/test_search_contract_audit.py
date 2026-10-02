import ast
from hashlib import sha256
import json

from Experiment.Synister.audit_search_contract import (
    ROOT, SOURCE_PATHS, cases, prefix_checks, query_checks, cyclic_settings)


def test_prefix_bounds_and_forced_edges_match_literal_completions():
    name, r, p = next(cases())
    report = prefix_checks(name, r, p)
    assert report['passed'] and report['prefixes'] == 326
    assert report['forced_assignment_entries'] > 0


def test_supported_streaming_tolerance_fixed_and_symmetry_queries():
    queries = query_checks()
    assert len(queries) == 72
    assert {q['tolerance'] for q in queries} == {0, 1e-9, .24999999999999997}
    assert any(q['dynamic_row_order'] for q in queries)
    assert any(q['fixed'] for q in queries)


def test_cyclic_default_budget_bound_and_boundary_case():
    settings = cyclic_settings()
    assert settings['boundary_check_passed']
    assert settings['maximum_orbit_generator_applications'] == 65280
    assert settings['maximum_distinct_nonidentity_generators'] == 255


def test_saved_contract_controls_and_benchmark_sources_are_bound():
    directory = ROOT/'paper/synister/evidence/search_contract_v1'
    summary = json.loads((directory/'summary.json').read_text())
    for name, expected in summary['file_sha256'].items():
        assert sha256((directory/name).read_bytes()).hexdigest() == expected
    assert summary['all_passed']
    assert (summary['cases'], summary['prefixes'], summary['forced_assignment_entries'],
            summary['attained_zero_profile_rows'], summary['streamed_queries']) == (13, 2387, 5125, 61, 72)
    sources = json.loads((directory/'sources.json').read_text())
    assert set(sources) == set(SOURCE_PATHS)
    for study in ('enumeration_main_v1', 'ablation_main_v1', 'output_scaling_v1'):
        snapshot = json.loads((ROOT/'paper/synister/evidence'/study/'sources.json').read_text())
        for path in SOURCE_PATHS:
            if path.startswith('synkit/'):
                assert snapshot[path] == sources[path]
    # The record binds the source version tested; it is not a claim that a
    # later corrected implementation has the same digest.


def test_historical_primary_workers_do_not_use_expanded_output():
    for study in ('identifiability_c1_primary_v1', 'identifiability_c2_primary_v1'):
        sources = json.loads((ROOT/'paper/synister/evidence'/study/'all_sources.json').read_text())
        worker = ast.parse(sources['Experiment/Synister/worker.py'])
        calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'enumerate_distance_mappings']
        assert calls
        for call in calls:
            keywords = {item.arg: item.value for item in call.keywords}
            assert 'expand_symmetry' not in keywords
        search = ast.parse(sources['synkit/Chem/Mapper/exact/distance.py'])
        function = next(node for node in search.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'enumerate_distance_mappings')
        defaults = dict(zip((arg.arg for arg in function.args.kwonlyargs), function.args.kw_defaults))
        assert ast.literal_eval(defaults['expand_symmetry']) is False
