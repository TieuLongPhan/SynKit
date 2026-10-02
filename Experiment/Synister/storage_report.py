"""Report retained attempt-file bytes and exports, not peak temporary storage."""
import argparse
import json
from pathlib import Path

from Experiment.Synister.resource_report import report as resource_report, sha


def report(directory):
    bound = resource_report(directory)
    stages = {}
    for name, digest in bound['record_hashes'].items():
        path = directory / 'cases' / name
        raw = path.read_bytes()
        row = json.loads(raw)
        assert sha(path) == digest
        stage = stages.setdefault(row['stage'], {'files': 0, 'bytes': 0})
        stage['files'] += 1
        stage['bytes'] += len(raw)
        if row['stage'] == 'exact':
            for field in ('emitted_representatives', 'labels', 'joint_labels'):
                metric = stage.setdefault(field, {'observed': 0, 'missing': 0,
                                                  'total': 0, 'maximum': None})
                value = row.get(field)
                if value is None:
                    metric['missing'] += 1
                    continue
                if field != 'emitted_representatives':
                    if not isinstance(value, list):
                        raise ValueError(f'Invalid export: {field}')
                    value = len(value)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f'Invalid count: {field}')
                metric['observed'] += 1
                metric['total'] += value
                metric['maximum'] = max(value, metric['maximum'] or 0)
    return {'scope': __doc__, 'primary_audit_sha256': bound['primary_audit_sha256'],
            'reporter_sha256': sha(Path(__file__)), 'record_hashes': bound['record_hashes'],
            'stages': stages,
            'interpretation': 'Bytes are exact serialized case-file sizes, excluding shared inputs, source snapshots, tasks, indexes and temporary files. Counts include all retained attempts; partial exports are not complete candidate sets. Missing counts are not zero. Representative counts are not labeled-map or ITS counts.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.directory)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result['stages'], indent=2))
