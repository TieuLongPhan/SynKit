from hashlib import sha256
import json

import pytest

from Experiment.Synister.report_enumeration_performance import (
    PAPER, build, conditional_ratios, describe, load_study, summarize)


def row(name, method, complete, seconds, **kwargs):
    return {'task': {'benchmark_id': name, 'query': 'minimum', 'method': method},
            'complete': complete, 'parent_seconds': seconds,
            'termination': 'complete' if complete else 'external_time_limit', **kwargs}


def test_incomplete_and_missing_measurements_are_not_completed_or_zero():
    result = summarize([row('a', 'synister', True, 2, mapping_count=8, peak_rss_kib=100),
                        row('b', 'synister', False, 75)])
    assert result['attempts'] == 2 and result['complete'] == 1
    assert result['completion_events_parent_seconds'] == [2]
    assert result['resources_all_attempts']['peak_rss_kib'] == {
        'observed': 1, 'missing': 1, 'min': 100, 'median': 100, 'max': 100}
    assert result['resources_completed_only']['parent_seconds']['median'] == 2
    assert result['resources_all_attempts']['parent_seconds']['median'] == 38.5
    assert describe([None], 1)['median'] is None
    with pytest.raises(ValueError, match='Invalid'):
        describe([float('nan')], 1)


def test_unproved_default_zero_is_not_an_instant_minimum_proof():
    result = summarize([row('a', 'synister', False, 60, minimum_proved=False, minimum_proof_seconds=0),
                        row('b', 'synister', True, 2, minimum_proved=True, minimum_proof_seconds=.5)])
    assert result['resources_all_attempts']['minimum_proof_seconds'] == {
        'observed': 1, 'missing': 1, 'min': .5, 'median': .5, 'max': .5}


def test_elementary_empty_targets_remain_visible_and_separate():
    empty = row('a', 'synister', True, 1, mapping_count=0)
    empty['task']['query'] = 'plus_1'
    empty['termination'] = 'proved_empty_precheck'
    result = summarize([empty, row('b', 'synister', True, 2, mapping_count=2)])
    assert result['complete'] == 2 and result['precheck_empty'] == 1
    assert result['search_attempts'] == 1
    assert result['search_completion_events_parent_seconds'] == [2]
    empty['complete'] = False
    with pytest.raises(ValueError, match='elementary'):
        summarize([empty])


def test_conditional_ratios_keep_all_four_completion_outcomes():
    records = []
    for name, ca, cb in [('a', True, True), ('b', True, False),
                         ('c', False, True), ('d', False, False)]:
        records.extend([row(name, 'synister', ca, 2), row(name, 'milp', cb, 10)])
    result = conditional_ratios(records, 'method', 'synister', 'milp')
    assert result['paired_attempts'] == 4 and result['jointly_complete'] == 1
    assert result['reference_only_complete'] == result['other_only_complete'] == result['neither_complete'] == 1
    assert result['other_over_reference_parent_seconds']['median'] == 5
    assert result['other_over_reference_parent_seconds']['missing'] == 3
    with pytest.raises(ValueError, match='Unmatched'):
        conditional_ratios(records[:-1], 'method', 'synister', 'milp')


def test_main_report_reconciles_every_query_and_retains_timer_scope():
    result = build({'matched': PAPER/'evidence/enumeration_main_v1'})
    main = result['studies']['matched']
    assert main['attempts'] == 676 and main['selected'] == 100
    assert len(main['attempt_measurements']) == 676
    assert {k: v['complete'] for k, v in main['by_configuration'].items()} == {'milp': 144, 'synister': 263}
    assert main['conditional_pairs'][0]['jointly_complete'] == 144
    for method, proof, complete in [('synister', 41, 41), ('milp', 42, 28)]:
        minimum = main['completion_curves']['minimum'][method]
        numeric = main['completion_curves']['specified_cd'][method]
        assert minimum['attempts'] == 100 and minimum['minimum_proved'] == proof
        assert minimum['resources_all_attempts']['minimum_proof_seconds']['observed'] == proof
        assert minimum['complete'] == complete and numeric['attempts'] == 238
        assert numeric['precheck_empty'] == 12
    assert 'not first feasible' in result['semantics']['synister_first_map_seconds']


def fixture_study(path):
    (path/'cases').mkdir()
    (path/'maps').mkdir()
    task = {'task_id': 'a', 'benchmark_id': 'a', 'query': 'minimum', 'method': 'synister'}
    record = row('a', 'synister', False, 75)
    record['task'] = task
    values = {'manifest.json': {}, 'minimum_tasks.json': [task], 'numeric_tasks.json': [],
              'inputs.json': [{'benchmark_id': 'a'}], 'cases/a.json': record}
    for name, value in values.items():
        (path/name).write_text(json.dumps(value))
    summary = {'attempts': 1, 'selected': 1,
               'all_record_hashes': {'a.json': sha256((path/'cases/a.json').read_bytes()).hexdigest()}}
    (path/'summary.json').write_text(json.dumps(summary))
    audit = {'all_output_comparisons_consistent': True, 'attempts': 1, 'selected': 1,
             'summary_sha256': sha256((path/'summary.json').read_bytes()).hexdigest(),
             'manifest_sha256': sha256((path/'manifest.json').read_bytes()).hexdigest()}
    (path/'audit.json').write_text(json.dumps(audit))
    return audit


def test_changed_or_unaudited_study_is_rejected(tmp_path):
    audit = fixture_study(tmp_path)
    assert len(load_study(tmp_path, 'matched')['records']) == 1
    (tmp_path/'cases/a.json').write_text('{}')
    with pytest.raises(ValueError, match='Attempt changed'):
        load_study(tmp_path, 'matched')
    audit['all_output_comparisons_consistent'] = False
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='successful terminal audit'):
        load_study(tmp_path, 'matched')
