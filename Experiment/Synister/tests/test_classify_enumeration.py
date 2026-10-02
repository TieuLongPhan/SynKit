from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.all_distance_oracle import literal_sets
from Experiment.Synister.audit_classification import audit_details
from Experiment.Synister.audit_enumeration_benchmark import audit as audit_upstream
from Experiment.Synister.classification_worker import classify
from Experiment.Synister.classify_enumeration import audit, digest, encode, prepare, run
from Experiment.Synister.mapping_landscape import product_automorphisms, toy_endpoints


def detail_record(result):
    return {key: result[key] for key in ('class_codes', 'representatives', 'failures',
                                       'bond_pattern_frequencies', 'joint_pattern_frequencies')}


def test_independent_graph_partition_checks_every_saved_assignment():
    r, p = toy_endpoints()
    maps = literal_sets(r, p)[4]
    result = classify(r, p, maps, 4, group=product_automorphisms(p))
    checked = audit_details(r, p, maps, result, detail_record(result), 4)
    assert checked['verified'] and checked['independent_observed_classes'] == 4
    assert checked['checked_orbits'] == 8 and checked['isomorphism_comparisons'] >= 4
    interrupted = audit_details(r, p, maps, result, detail_record(result), 4, seconds=0)
    assert not interrupted['verified'] and interrupted['consistent'] is None
    assert interrupted['termination'] == 'audit_time_limit'


def test_independent_check_rejects_false_class_merging_and_splitting():
    r, p = toy_endpoints()
    maps = literal_sets(r, p)[4]
    for code, message in ((lambda m, t: ('same', None), 'merged'),
                          (lambda m, t: (tuple(m), None), 'split')):
        result = classify(r, p, maps, 4, group=product_automorphisms(p), code_function=code)
        with pytest.raises(ValueError, match=message):
            audit_details(r, p, maps, result, detail_record(result), 4)


def test_incomplete_classification_bounds_and_empty_case_are_verified():
    r, p = toy_endpoints()
    maps = literal_sets(r, p)[4]
    for result in (classify(r, p, maps, 4, seconds=0),
                   classify(r, p, maps, 4, code_function=lambda *args: (None, 'time_limit'))):
        assert audit_details(r, p, maps, result, detail_record(result), 4)['verified']
        bad = deepcopy(result)
        bad['its_class_upper_bound'] = 0
        with pytest.raises(ValueError, match='claim'):
            audit_details(r, p, maps, bad, detail_record(bad), 4)
    empty = classify(r, p, [], 0, seconds=0)
    assert audit_details(r, p, [], empty, detail_record(empty), 0)['verified']


def test_independent_check_rejects_changed_bond_pattern():
    r, p = toy_endpoints()
    maps = literal_sets(r, p)[4]
    result = classify(r, p, maps, 4)
    details = deepcopy(detail_record(result))
    details['representatives'][0]['bond_pattern'] = 'invented'
    with pytest.raises(ValueError, match='change pattern'):
        audit_details(r, p, maps, result, details, 4)


def upstream_fixture(directory):
    """Small explicit saved-output fixture, checked by the real upstream auditor."""
    directory.mkdir()
    (directory/'cases').mkdir()
    (directory/'maps').mkdir()
    rows = [{'benchmark_id': 'example', 'source': 'Rhea', 'atoms': 2, 'size_bin': 0, 'reaction': 'CC>>CC'}]
    (directory/'inputs.json').write_text(encode(rows))
    (directory/'sources.json').write_text('{}\n')
    limits = {'seconds': 5, 'memory_gib': 2, 'max_maps': 100}
    tasks = []
    for query, target in (('minimum', 'minimal'), ('at_minimum', 0), ('plus_1', 2), ('reference_cd', 4)):
        for method in ('synister', 'milp'):
            key = f'example.{query}.{method}'
            task = {'task_id': key, 'benchmark_id': 'example', 'reaction': 'CC>>CC',
                    'query': query, 'target_doubled_cd': target, 'method': method, **limits}
            tasks.append(task)
            record = {'task': task, 'complete': query != 'reference_cd',
                      'termination': 'complete' if query != 'reference_cd' else 'external_time_limit',
                      'minimum_proved': query == 'minimum'}
            if query == 'minimum':
                record['minimum_doubled_cd'] = 0
            if record['complete']:
                maps = [[0, 1], [1, 0]] if query in ('minimum', 'at_minimum') else []
                path = directory/'maps'/f'{key}.json'
                path.write_text(json.dumps(maps, separators=(',', ':')))
                record.update(mapping_count=len(maps), mapping_sha256=digest(path), output_bytes=path.stat().st_size)
            (directory/'cases'/f'{key}.json').write_text(encode(record))
    (directory/'minimum_tasks.json').write_text(encode([t for t in tasks if t['query'] == 'minimum']))
    (directory/'numeric_tasks.json').write_text(encode([t for t in tasks if t['query'] != 'minimum']))
    manifest = {**limits, 'inputs_sha256': digest(directory/'inputs.json'), 'sources_sha256': digest(directory/'sources.json')}
    (directory/'manifest.json').write_text(encode(manifest))
    summary = {'selected': 1, 'attempts': len(tasks),
               'all_record_hashes': {p.name: digest(p) for p in (directory/'cases').glob('*.json')}}
    (directory/'summary.json').write_text(encode(summary))
    (directory/'audit.json').write_text(encode(audit_upstream(directory)))
    return directory


def test_query_deduplication_keeps_original_failures_and_attempts(tmp_path):
    upstream = upstream_fixture(tmp_path/'upstream')
    rows, tasks = prepare(upstream, seconds=5, audit_seconds=5, memory_gib=2)
    assert len(rows) == 1 and len(tasks) == 3
    assert sum(len(t['query_aliases']) for t in tasks) == 8
    minimum, empty, unavailable = tasks
    assert minimum['mapping_count'] == 2 and len(minimum['query_aliases']) == 4
    assert minimum['source_task_id'] == 'example.at_minimum.synister'
    assert empty['mapping_count'] == 0 and empty['complete_indexed_set_available']
    assert not unavailable['complete_indexed_set_available'] and unavailable['source_task_id'] is None
    assert all(a['termination'] == 'external_time_limit' for a in unavailable['query_aliases'])


def test_isolated_classification_and_independent_audit_end_to_end(tmp_path):
    upstream = upstream_fixture(tmp_path/'upstream')
    args = SimpleNamespace(matched=upstream, output=tmp_path/'classification', after=None,
                           seconds=5, audit_seconds=5, memory_gib=2, workers=1)
    report = run(args)
    assert report['accounting_verified'] and report['unique_queries'] == 3
    assert report['original_attempts'] == 8 and report['original_paired_queries'] == 4
    assert report['complete_indexed_sets_available'] == 2 and report['no_complete_indexed_set'] == 1
    assert report['independently_verified_complete_classifications'] == 2
    assert report['structural_contradictions'] == 0
    assert report['all_saved_classifications_independently_verified']
    with pytest.raises(FileExistsError):
        run(args)
    detail = args.output/'details'/'example.cd_0.json'
    detail.write_text(detail.read_text()+' ')
    with pytest.raises(ValueError, match='details changed'):
        audit(args.output, upstream)


def test_missing_and_changed_source_records_are_not_silently_skipped(tmp_path):
    upstream = upstream_fixture(tmp_path/'upstream')
    record = upstream/'cases'/'example.minimum.synister.json'
    original = record.read_text()
    record.write_text(original+' ')
    with pytest.raises(ValueError, match='record changed'):
        prepare(upstream, seconds=5, audit_seconds=5, memory_gib=2)


def test_unproved_minimum_is_retained_separately_from_a_numeric_query(tmp_path):
    upstream = upstream_fixture(tmp_path/'upstream')
    # Keep only the unknown-minimum and numeric reference queries. All four
    # attempts end without a proved minimum or complete mapping set.
    tasks = json.loads((upstream/'minimum_tasks.json').read_text())
    tasks += [t for t in json.loads((upstream/'numeric_tasks.json').read_text()) if t['query'] == 'reference_cd']
    for path in (upstream/'cases').glob('*.json'):
        if path.stem not in {t['task_id'] for t in tasks}:
            path.unlink()
        else:
            record = json.loads(path.read_text())
            record = {'task': record['task'], 'complete': False, 'minimum_proved': False,
                      'termination': 'external_time_limit'}
            path.write_text(encode(record))
    (upstream/'numeric_tasks.json').write_text(encode([t for t in tasks if t['query'] == 'reference_cd']))
    summary = {'selected': 1, 'attempts': 4,
               'all_record_hashes': {p.name: digest(p) for p in (upstream/'cases').glob('*.json')}}
    (upstream/'summary.json').write_text(encode(summary))
    (upstream/'audit.json').write_text(encode(audit_upstream(upstream)))
    _, planned = prepare(upstream, seconds=5, audit_seconds=5, memory_gib=2)
    assert len(planned) == 2 and sum(len(t['query_aliases']) for t in planned) == 4
    assert planned[0]['task_id'] == 'example.minimum_unproved'
    assert planned[0]['target_doubled_cd'] is None
    assert planned[1]['target_doubled_cd'] == 4
    assert all(not t['complete_indexed_set_available'] for t in planned)
