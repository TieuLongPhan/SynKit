"""Structural counts and resources from an audited, terminal CD-set study.

Enumeration, canonical classification and independent structural verification
are separate conditions. Unverified counts remain in the raw record, but are
not promoted to independently supported counts in this report.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from Experiment.Synister.classify_enumeration import audit, digest, encode
from Experiment.Synister.report_enumeration_performance import describe, load_study


METRICS = ('parent_seconds', 'worker_seconds', 'cpu_seconds', 'peak_rss_kib',
           'input_seconds', 'group_seconds', 'orbit_seconds', 'label_seconds',
           'canonical_seconds', 'export_seconds', 'end_to_end_seconds', 'output_bytes')
AUDIT_METRICS = ('parent_seconds', 'worker_seconds', 'cpu_seconds', 'peak_rss_kib',
                 'elapsed_seconds', 'checked_orbits', 'isomorphism_comparisons')


def compact(record):
    """Keep unavailable, unverified, bounded and exact outcomes distinct."""
    task = record['task']
    available = task['complete_indexed_set_available']
    checked = record['structural_audit']
    verified = checked.get('verified') is True
    complete = record.get('classification_complete') is True
    if checked.get('consistent') is False:
        raise ValueError('Structural contradiction; cannot publish this report')
    if verified and (not available or checked.get('consistent') is not True):
        raise ValueError('Verification lacks a complete input set or consistency')
    if not available:
        status = 'enumeration_incomplete'
    elif 'detail_sha256' not in record:
        status = 'classification_unavailable'
    elif not verified:
        status = 'classification_not_independently_verified'
    else:
        status = 'verified_exact' if complete else 'verified_bounds'
    row = {key: task[key] for key in ('task_id', 'benchmark_id', 'source', 'atoms',
                                     'size_bin', 'target_doubled_cd', 'query_aliases')}
    row.update(status=status, complete_indexed_set_available=available,
               source_task_id=task['source_task_id'],
               indexed_maps=task.get('mapping_count') if available else None,
               partial_mapping_lower_bound=task['partial_mapping_lower_bound'],
               classification_complete=complete, independently_verified=verified,
               termination=record['termination'], audit_termination=checked['termination'])
    fields = ('product_orbits', 'group_order', 'its_class_lower_bound',
              'its_class_upper_bound', 'bond_pattern_count', 'joint_pattern_count',
              'bond_pattern_lower_bound', 'joint_pattern_lower_bound')
    row.update({key: record.get(key) if verified else None for key in fields})
    row['its_classes'] = record.get('its_classes') if verified and complete else None
    row['raw_classification_claims'] = {key: record.get(key) for key in
                                        (*fields, 'indexed_maps', 'its_classes')}
    row['resources'] = {key: record.get(key) for key in METRICS}
    row['audit_resources'] = {key: checked.get(key) for key in AUDIT_METRICS}
    if verified:
        l, q, h = row['indexed_maps'], row['product_orbits'], row['group_order']
        lo, hi = row['its_class_lower_bound'], row['its_class_upper_bound']
        if (record['indexed_maps'] != l or h < 1 or l != q*h
                or not (int(l > 0) <= lo <= hi <= q)):
            raise ValueError('Invalid independently verified count relations')
        if complete and not (lo == hi == row['its_classes']):
            raise ValueError('Exact classification differs from its bounds')
    return row


def summarize(rows):
    """Use unique reaction/CD groups, not aliases, as structural denominators."""
    exact = [r for r in rows if r['status'] == 'verified_exact']
    nonempty = [r for r in exact if r['indexed_maps'] > 0]
    return {
        'unique_queries': len(rows),
        'reactions': len({r['benchmark_id'] for r in rows}),
        'original_attempts': sum(len(r['query_aliases']) for r in rows),
        'status_counts': dict(sorted(Counter(r['status'] for r in rows).items())),
        'complete_indexed_sets': sum(r['complete_indexed_set_available'] for r in rows),
        'proved_empty_sets': sum(r['indexed_maps'] == 0 for r in rows),
        'verified_exact_nonempty_queries': len(nonempty),
        'verified_exact_nonempty_multiple_its': sum(r['its_classes'] > 1 for r in nonempty),
        'verified_exact_nonempty_product_symmetry_reduction': sum(
            r['product_orbits'] < r['indexed_maps'] for r in nonempty),
        'verified_exact_nonempty_further_its_reduction': sum(
            r['its_classes'] < r['product_orbits'] for r in nonempty),
        'classification_resources_all_queries': {
            key: describe([r['resources'][key] for r in rows], len(rows)) for key in METRICS},
        'audit_resources_all_queries': {
            key: describe([r['audit_resources'][key] for r in rows], len(rows)) for key in AUDIT_METRICS},
    }


def build(directory, matched):
    directory, matched = Path(directory), Path(matched)
    # Verify all upstream attempts and map files, including unselected aliases.
    upstream = load_study(matched, 'matched')
    saved = json.loads((directory/'audit.json').read_text())
    fresh = audit(directory, matched)
    # Auditor bytes can differ in a later release; its recomputed accounting
    # must still agree in full. Preserve both identities in the report.
    if {k: v for k, v in fresh.items() if k != 'auditor_sha256'} != {
            k: v for k, v in saved.items() if k != 'auditor_sha256'}:
        raise ValueError('Saved structural accounting differs from fresh audit')
    if not fresh['accounting_verified'] or fresh['structural_contradictions']:
        raise ValueError('Structural study lacks consistent terminal accounting')
    rows = [compact(json.loads(path.read_text())) for path in sorted((directory/'cases').glob('*.json'))]
    summary = summarize(rows)
    if (summary['unique_queries'] != fresh['unique_queries']
            or summary['original_attempts'] != fresh['original_attempts']
            or summary['complete_indexed_sets'] != fresh['complete_indexed_sets_available']):
        raise ValueError('Structural report denominator mismatch')
    return {
        'schema': 'synister.structural-report.v1',
        'scope': 'All unique reaction/CD groups; original solver attempts retained as aliases',
        'count_meanings': {
            'indexed_maps': 'L: complete indexed map count; null if enumeration incomplete',
            'product_orbits': 'Q=L/|H| for a newly verified cyclic product subgroup; not the full symmetry quotient',
            'its_classes': 'K: independently verified exact full attributed ITS count; null otherwise',
            'bounds': 'Verified lower/upper K bounds are not complete classification',
            'patterns': 'Fixed-reactant-coordinate bond and joint change patterns, distinct from ITS equivalence',
        },
        'resource_scope': 'Classification and independent audit measured separately after enumeration; missing is not zero; nested phase times must not be summed',
        'provenance': {
            'directory': str(directory), 'matched_directory': str(matched),
            'sha256': {name: digest(directory/name) for name in ('manifest.json', 'summary.json', 'audit.json')},
            'upstream': upstream['provenance'],
            'saved_auditor_sha256': saved['auditor_sha256'],
            'replayed_auditor_sha256': fresh['auditor_sha256'],
            'reporter_sha256': digest(Path(__file__)),
        },
        'summary': summary,
        'by_source': {source: summarize([r for r in rows if r['source'] == source])
                      for source in sorted({r['source'] for r in rows})},
        'rows': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification', type=Path, required=True)
    parser.add_argument('--matched', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.classification, args.matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encode(report))
    print(encode(report['summary']))
