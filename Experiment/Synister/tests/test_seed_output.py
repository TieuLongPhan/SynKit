from hashlib import sha256
import json
from pathlib import Path

import pytest

from Experiment.Synister.all_distance_oracle import literal_sets, weighted_cases
from Experiment.Synister.mapping_landscape import product_automorphisms
from Experiment.Synister.seed_output_worker import diagnostic, perform
from synkit.Chem.Mapper.exact.symmetry import largest_cyclic_subgroup
from synkit.Chem.Mapper.identifiability import parse_reaction


def test_staged_outputs_and_seed_conditions_match_literal_full_sets():
    for _, r, p in weighted_cases(8):
        oracle = literal_sets(r, p)
        group = largest_cyclic_subgroup(product_automorphisms(p))
        for seed in (None, min(oracle[max(oracle)])):
            outputs = {unit: diagnostic(r, p, seed, unit, 10, 100000, group)
                       for unit in ('proof', 'representatives', 'indexed')}
            assert all(v['complete'] and v['minimum_proved'] for v in outputs.values())
            assert all(v['minimum_doubled_cd'] == min(oracle) for v in outputs.values())
            assert len(outputs['proof']['mappings']) == 1
            reps = outputs['representatives']['mappings']
            expanded = {tuple(g[j] for j in m) for m in reps for g in group}
            assert len(expanded) == len(reps)*len(group)
            assert expanded == set(outputs['indexed']['mappings']) == oracle[min(oracle)]
            assert outputs['proof']['enumeration_seconds'] is None
            assert outputs['representatives']['expansion_seconds'] is None
            assert outputs['indexed']['expansion_seconds'] >= 0


def test_cap_and_timeout_preserve_proof_output_distinction():
    r, p = parse_reaction('C.C>>C.C')
    identity = [(0, 1)]
    proof = diagnostic(r, p, None, 'proof', 5, 1, identity)
    indexed = diagnostic(r, p, None, 'indexed', 5, 1, identity)
    assert proof['complete'] and proof['minimum_proved']
    assert indexed['minimum_proved'] and not indexed['complete']
    limited = diagnostic(r, p, None, 'indexed', 0, 1000, identity)
    assert not limited['complete'] and not limited['minimum_proved']
    with pytest.raises(ValueError, match='changes an ITS attribute'):
        r, p = parse_reaction('CO>>CO')
        diagnostic(r, p, None, 'indexed', 5, 1000, [(0, 1), (1, 0)])


def test_native_workers_share_fresh_prediction_without_fixing_assignments(tmp_path):
    reaction = 'CC(=O)O>>CC(O)=O'
    seed = perform({'stage': 'seed', 'reaction': reaction})
    maps = []
    for method in ('synister', 'milp'):
        for condition in ('none', 'slap'):
            path = tmp_path/f'{method}.{condition}.json'
            result = perform({'stage': 'matched', 'method': method, 'reaction': reaction,
                'seed_condition': condition, 'initial_mapping': seed['prediction']['mapping'] if condition == 'slap' else None,
                'seconds': 10, 'max_maps': 100000, 'map_path': str(path)})
            assert result['complete'] and result['minimum_proved']
            assert result['expansion_seconds'] is None
            maps.append(json.loads(path.read_text()))
    assert all(m == maps[0] for m in maps)
    failed = perform({'stage': 'matched', 'method': 'synister', 'reaction': reaction,
                      'seed_condition': 'slap', 'initial_mapping': None})
    assert failed['termination'] == 'seed_unavailable' and not failed['complete']


def test_source_frozen_worker_and_input_only_selection(tmp_path):
    from Experiment.Synister.seed_output_benchmark import freeze, isolated
    output = tmp_path/'pilot'
    rows = freeze(output, Path('paper/synister/evidence/enumeration_main_v1/inputs.json'))
    assert len(rows) == 20
    rest = json.loads((output/'extension_inputs.json').read_text())
    assert len(rest) == 80 and {r['benchmark_id'] for r in rows}.isdisjoint(r['benchmark_id'] for r in rest)
    hard = json.loads((output/'difficult_inputs.json').read_text())
    assert len(hard) == 20 and all(r['size_bin'] >= 3 for r in hard)
    result = isolated(output, 'seed_output_worker', {'stage': 'seed', 'reaction': 'CO>>CO', 'memory_gib': 6}, 10)
    assert result['complete'], result
    assert Path(result['execution_root']) == output/'frozen_source'
    assert result['worker_sha256'] == sha256((output/'frozen_source/Experiment/Synister/seed_output_worker.py').read_bytes()).hexdigest()


def test_extension_requires_terminal_audit_and_uses_all_reserved_inputs(tmp_path, monkeypatch):
    from Experiment.Synister import seed_output_benchmark as study
    parent = tmp_path/'pilot'
    study.freeze(parent, Path('paper/synister/evidence/enumeration_main_v1/inputs.json'))
    (parent/'summary.json').write_text('{}')
    valid = {'all_output_comparisons_consistent': True, 'attempts': 200,
             'summary_sha256': sha256(b'{}').hexdigest()}
    (parent/'audit.json').write_text(json.dumps({**valid,'attempts':199}))
    with pytest.raises(ValueError, match='gate not satisfied'):
        study.extend(tmp_path/'rejected',parent)
    assert not (tmp_path/'rejected').exists()
    (parent/'audit.json').write_text(json.dumps(valid))
    def fake_execute(output,task,preparation=False):
        record={'task':task,'complete':False,'termination':'injected_limit','parent_seconds':.1}
        folder='preparation' if preparation else 'cases'
        (output/folder/(task['task_id']+'.json')).write_text(json.dumps(record))
        return record
    monkeypatch.setattr(study,'execute',fake_execute)
    summary=study.extend(tmp_path/'extension',parent)
    assert summary['selected']==80 and summary['matched_attempts']==320
    tasks=json.loads((tmp_path/'extension/tasks.json').read_text())
    assert all(t['seconds']==60 and t['stage']=='matched' for t in tasks)
    assert len({t['benchmark_id'] for t in tasks})==80
    with pytest.raises(ValueError, match='requires the completed 60-second'):
        study.extend(tmp_path/'followup',parent,followup=True)
    extension=tmp_path/'extension'
    (extension/'audit.json').write_text(json.dumps({**valid,'attempts':320,
        'summary_sha256':sha256((extension/'summary.json').read_bytes()).hexdigest()}))
    followup=tmp_path/'followup'
    later=study.extend(followup,parent,followup=True,main_extension=extension)
    assert later['selected']==20 and later['attempts']==80
    later_tasks=json.loads((followup/'tasks.json').read_text())
    assert all(t['seconds']==300 and t['stage']=='matched' for t in later_tasks)
    protocol=json.loads((followup/'protocol.json').read_text())
    assert protocol['search_seconds']==300 and protocol['parent_seconds']==315
    assert '300s search' in protocol['accounting'] and '315s search-process' in protocol['accounting']
    assert (followup/'sources.json').read_bytes()==(parent/'sources.json').read_bytes()


def test_complete_representatives_classify_without_claiming_indexed_input(tmp_path):
    from Experiment.Synister.classify_seed_representatives import perform as classify_reps
    reaction = 'CC(=O)O>>CC(O)=O'
    r, p = parse_reaction(reaction)
    group = largest_cyclic_subgroup(product_automorphisms(p))
    result = diagnostic(r, p, None, 'representatives', 10, 100000, group)
    path = tmp_path/'maps.json'
    path.write_text(json.dumps(result['mappings']))
    task = {'reaction': reaction, 'map_path': str(path), 'mapping_sha256': sha256(path.read_bytes()).hexdigest(),
            'source_group': group, 'mapping_count': len(result['mappings']), 'source_complete': True,
            'target_doubled_cd': result['minimum_doubled_cd'], 'detail_path': str(tmp_path/'details.json')}
    classified = classify_reps(task)
    assert classified['complete'] and classified['structural_audit']['verified']
    assert classified['input_unit'] == 'complete_verified_subgroup_representatives'
    assert classified['input_count'] == len(result['mappings'])
    assert classified['implied_indexed_count'] == len(literal_sets(r, p)[result['minimum_doubled_cd']])
    assert classified['its_classes'] == 1
    with pytest.raises(ValueError, match='lacks complete audited input'):
        classify_reps({**task, 'source_complete': False})
