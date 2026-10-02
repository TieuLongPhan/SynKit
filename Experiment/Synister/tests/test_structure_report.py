import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.classify_enumeration import run
from Experiment.Synister.report_enumeration_structure import build, compact, summarize
from Experiment.Synister.tests.test_classify_enumeration import upstream_fixture


def record(*, complete=True, verified=True):
    return {
        'task': {'task_id': 'a.cd_2', 'benchmark_id': 'a', 'source': 'Rhea',
                 'atoms': 5, 'size_bin': 0, 'target_doubled_cd': 2,
                 'query_aliases': [{'query': 'minimum'}, {'query': 'at_minimum'}],
                 'complete_indexed_set_available': True, 'mapping_count': 16,
                 'source_task_id': 'a.minimum.synister', 'partial_mapping_lower_bound': 16},
        'detail_sha256': 'example', 'classification_complete': complete,
        'termination': 'complete' if complete else 'classification_time_limit',
        'structural_audit': {'verified': verified, 'consistent': True if verified else None,
                             'termination': 'verified' if verified else 'audit_time_limit'},
        'indexed_maps': 16, 'group_order': 2, 'product_orbits': 8,
        'its_classes': 4 if complete else None, 'its_class_lower_bound': 4,
        'its_class_upper_bound': 4 if complete else 7,
        'bond_pattern_count': 5 if complete else None,
        'bond_pattern_lower_bound': 5, 'joint_pattern_count': 6 if complete else None,
        'joint_pattern_lower_bound': 6,
    }


def test_exact_bounds_and_unverified_are_distinct():
    exact = compact(record())
    bounded = compact(record(complete=False))
    unchecked = compact(record(verified=False))
    assert exact['status'] == 'verified_exact' and exact['its_classes'] == 4
    assert bounded['status'] == 'verified_bounds' and bounded['its_classes'] is None
    assert (bounded['its_class_lower_bound'], bounded['its_class_upper_bound']) == (4, 7)
    assert unchecked['status'] == 'classification_not_independently_verified'
    assert unchecked['indexed_maps'] == 16
    assert unchecked['its_classes'] is None and unchecked['product_orbits'] is None
    assert unchecked['its_class_lower_bound'] is None
    assert unchecked['raw_classification_claims']['its_classes'] == 4


def test_incomplete_enumeration_is_not_zero_and_keeps_aliases():
    raw = record(complete=False, verified=False)
    raw.pop('detail_sha256')
    raw['task'].update(complete_indexed_set_available=False, source_task_id=None)
    row = compact(raw)
    assert row['status'] == 'enumeration_incomplete' and row['indexed_maps'] is None
    assert len(row['query_aliases']) == 2 and row['partial_mapping_lower_bound'] == 16
    totals = summarize([row, compact(record())])
    assert totals['unique_queries'] == 2 and totals['original_attempts'] == 4
    assert totals['proved_empty_sets'] == 0
    assert totals['verified_exact_nonempty_queries'] == 1
    assert totals['classification_resources_all_queries']['parent_seconds']['missing'] == 2


def test_empty_and_failed_classification_are_not_conflated():
    raw = record()
    raw['task']['mapping_count'] = 0
    for key in ('indexed_maps', 'product_orbits', 'its_classes', 'its_class_lower_bound', 'its_class_upper_bound'):
        raw[key] = 0
    empty = compact(raw)
    failed = record(complete=False, verified=False)
    failed.pop('detail_sha256')
    failed = compact(failed)
    assert empty['its_classes'] == 0 and empty['status'] == 'verified_exact'
    assert failed['status'] == 'classification_unavailable' and failed['indexed_maps'] == 16
    totals = summarize([empty, failed])
    assert totals['proved_empty_sets'] == 1 and totals['verified_exact_nonempty_queries'] == 0


def test_invalid_relations_or_contradictions_fail_closed():
    raw = record()
    raw['product_orbits'] = 3
    with pytest.raises(ValueError, match='count relations'):
        compact(raw)
    raw = record()
    raw['structural_audit']['consistent'] = False
    with pytest.raises(ValueError, match='contradiction'):
        compact(raw)


def test_report_replays_terminal_accounting_and_rejects_tampering(tmp_path):
    upstream = upstream_fixture(tmp_path/'upstream')
    output = tmp_path/'classification'
    run(SimpleNamespace(matched=upstream, output=output, after=None,
                        seconds=5, audit_seconds=5, memory_gib=2, workers=1))
    report = build(output, upstream)
    assert report['summary']['unique_queries'] == 3
    assert report['summary']['original_attempts'] == 8
    assert report['summary']['status_counts'] == {'enumeration_incomplete': 1, 'verified_exact': 2}
    assert report['summary']['proved_empty_sets'] == 1
    assert report['summary']['verified_exact_nonempty_queries'] == 1
    audit_path = output/'audit.json'
    saved = audit_path.read_text()
    changed = json.loads(saved)
    changed['complete_classifications'] += 1
    audit_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='accounting differs'):
        build(output, upstream)
    audit_path.write_text(saved)
    detail_path = next((output/'details').glob('*.json'))
    detail_path.write_text(detail_path.read_text()+' ')
    with pytest.raises(ValueError, match='details changed'):
        build(output, upstream)
