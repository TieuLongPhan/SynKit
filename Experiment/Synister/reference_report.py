"""Post-primary reference-admission stratification of saved secondary records.

Descriptive aggregation, not new search or an independent closure proof.
Full-ITS admission follows mapping admission by weighted-CD invariance.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    groups = {}
    admission = {name: Counter() for name in ('mapping', 'full_its', 'bond', 'atom')}
    for row in rows:
        assessed = row.get('status') == 'evaluated'
        value = row.get('reference', {}).get('mapping_in_minimum') if assessed else None
        if value is not None and type(value) is not bool:
            raise ValueError('Invalid reference membership')
        category = 'unassessed' if value is None else ('admitted' if value else 'outside')
        admission['mapping'][category] += 1
        admission['full_its'][category] += 1
        for name, metric in [('bond', 'bond_f1'), ('atom', 'atom_f1')]:
            record = row.get('metrics', {}).get(metric, {}) if assessed else {}
            member = record.get('reference_orbit_in_minimum')
            state = 'unassessed'
            if record.get('reference_status') == 'complete':
                if type(member) is not bool:
                    raise ValueError('Missing or invalid completed orbit membership')
                state = 'admitted' if member else 'outside'
            admission[name][state] += 1
        group = groups.setdefault(category, {'selected': 0, 'intervals': []})
        group['selected'] += 1
        metric = row.get('metrics', {}).get('bond_f1', {}) if assessed else {}
        if metric.get('status') == 'complete':
            lo, hi = (Fraction(metric[end]['difference']) for end in ('lower', 'upper'))
            if not -1 <= lo <= hi <= 1:
                raise ValueError('Invalid paired interval')
            group['intervals'].append((lo, hi))
    for group in groups.values():
        values = group.pop('intervals')
        n, k = group['selected'], len(values)
        lo = sum((x[0] for x in values), Fraction())
        hi = sum((x[1] for x in values), Fraction())
        group.update(resolved=k, unresolved=n-k,
                     positive_width=sum(a < b for a, b in values),
                     local_reversal=sum(a < 0 < b for a, b in values),
                     conditional_envelope=[str(lo/k), str(hi/k)] if k else None,
                     outer_envelope=[str((lo-(n-k))/n), str((hi+(n-k))/n)])
    return {'admission': {k: dict(v) for k, v in admission.items()}, 'strata': groups}


def report(directory, replay_manifest):
    manifest = json.loads((directory / 'manifest.json').read_text())
    replay = json.loads(replay_manifest.read_text())
    if sha(directory / 'manifest.json') != replay['parent_manifest_sha256']:
        raise ValueError('Replay parent manifest mismatch')
    if sha(directory / 'inputs.json') != manifest['inputs_sha256']:
        raise ValueError('Input hash mismatch')
    inputs = json.loads((directory / 'inputs.json').read_text())
    paths = [directory / 'cases' / f"{row['case_id']}.annotations.json" for row in inputs]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate case IDs')
    expected = {}
    for name, digest in replay['artifact_sha256'].items():
        if name.endswith('.annotations.json'):
            key = Path(name).name
            if key in expected:
                raise ValueError('Duplicate replay binding')
            expected[key] = digest
    if set(expected) != {path.name for path in paths}:
        raise ValueError('Replay annotation inventory mismatch')
    for path in paths:
        if sha(path) != expected[path.name]:
            raise ValueError(f'Record hash mismatch: {path.name}')
    rows = [json.loads(path.read_text()) for path in paths]
    for source, row in zip(inputs, rows):
        if source['case_id'] != row['case_id']:
            raise ValueError('Case identity mismatch')
    return dict(summarize(rows), scope=__doc__, reporter_sha256=sha(Path(__file__)),
                replay_manifest_sha256=sha(replay_manifest),
                inputs_sha256=sha(directory / 'inputs.json'),
                manifest_sha256=sha(directory / 'manifest.json'),
                record_hashes={path.name: sha(path) for path in paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-manifest', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.directory, args.replay_manifest)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in ('admission', 'strata')}, indent=2))
