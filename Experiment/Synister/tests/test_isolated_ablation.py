import pytest

from Experiment.Synister.ablation_worker import feasible_seed
from Experiment.Synister.all_distance_oracle import literal_sets, targets_for, weighted_cases
from Experiment.Synister.isolated_ablation import compile_variant, VARIANTS
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings


def test_isolated_removals_match_all_literal_shells_and_minima():
    functions = {v: compile_variant(v) for v in VARIANTS}
    for _, r, p in weighted_cases(12):
        expected = literal_sets(r, p)
        seed, _ = feasible_seed(r, p)
        for target in ['minimal', *targets_for(expected, False)]:
            wanted = expected.get(min(expected) if target == 'minimal' else target, set())
            for variant, (function, identity) in functions.items():
                maps = []
                result = function([r.graph(), p.graph()], CD=target if target == 'minimal' else target/2,
                    binary=False, max_bijections=None, initial_mapping=seed, tolerance=0,
                    collect_mappings=False, mapping_callback=lambda m,c: maps.append(tuple(m)),
                    compute_minimum_cost=target == 'minimal', **identity['options'])
                assert result.complete, (variant, target)
                assert set(maps) == wanted and len(maps) == len(wanted), (variant, target)
                if target == 'minimal':
                    assert result.minimum_cost*2 == min(expected)


def test_private_recursion_and_unchanged_full_search():
    original = enumerate_distance_mappings
    with pytest.raises(ValueError, match='Unknown isolated'):
        compile_variant('no_everything')
    for _, r, p in weighted_cases(8):
        function, identity = compile_variant('full')
        assert function.__globals__['enumerate_distance_mappings'] is function
        for target in ('minimal', 3):
            options = dict(CD=target, binary=False, max_bijections=None,
                           collect_mappings=False, mapping_callback=lambda m,c: None,
                           **identity['options'])
            a = original([r.graph(), p.graph()], **options)
            b = function([r.graph(), p.graph()], **options)
            assert (a.complete, a.minimum_cost, a.selected_mapping_count, a.visited_nodes) == (
                b.complete, b.minimum_cost, b.selected_mapping_count, b.visited_nodes)
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings as unchanged
    assert unchanged is original


def test_separate_observer_preserves_outputs_and_counts_proof_and_enumeration(tmp_path):
    import json
    from Experiment.Synister.isolated_ablation_worker import perform
    from Experiment.Synister.worked_oracle import REACTION
    for variant in VARIANTS:
        task = {'variant':variant,'reaction':REACTION,'target_doubled_cd':'minimal',
                'seconds':30,'max_maps':100000,'map_path':str(tmp_path/(variant+'.ordinary.json'))}
        ordinary = perform(task)
        instrumented = perform({**task,'instrumented':True,'map_path':str(tmp_path/(variant+'.observed.json'))})
        assert ordinary['complete'] and instrumented['complete']
        assert json.loads((tmp_path/(variant+'.ordinary.json')).read_text()) == json.loads((tmp_path/(variant+'.observed.json')).read_text())
        phases = instrumented['instrumentation']['phases']
        assert phases['minimum_proof']['counters']['visited_nodes'] == instrumented['minimum_proof_nodes']
        assert phases['enumeration']['counters']['visited_nodes'] == instrumented['visited_nodes']
        assert phases['enumeration']['timings']['indexed_expansion_and_callback_seconds'] > 0
        assert ordinary['visited_nodes'] == instrumented['visited_nodes']


def test_frozen_instrumented_worker_and_source_bound_control_gate(tmp_path):
    from pathlib import Path
    from Experiment.Synister.seed_output_benchmark import freeze, isolated
    from Experiment.Synister.isolated_ablation_benchmark import validate_controls, run
    output = tmp_path/'frozen'
    freeze(output,Path('paper/synister/evidence/enumeration_main_v1/inputs.json'))
    task = {'variant':'no_conditioned_filter','reaction':'CC(=O)O>>CC(O)=O',
            'target_doubled_cd':'minimal','seconds':10,'memory_gib':6,'max_maps':100000,
            'map_path':str(tmp_path/'maps.json'),'instrumented':True}
    record = isolated(output,'isolated_ablation_worker',task,20)
    assert record['complete'] and record['minimum_proved'], record
    assert record['experimental_solver']['changed_anchors'] == 1
    assert record['instrumentation']['phases']['minimum_proof']['counters']['visited_nodes'] >= 1
    controls = Path('paper/synister/evidence/isolated_ablation_controls_v1')
    assert validate_controls(controls)['all_passed']
    with pytest.raises(ValueError,match='follows an audited ordinary'):
        run(tmp_path/'rejected',controls,instrumented=True)
    assert not (tmp_path/'rejected').exists()
