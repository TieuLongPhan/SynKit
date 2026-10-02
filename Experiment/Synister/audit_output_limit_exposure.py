"""Preserve the output-cap diagnostic and check exposure of existing studies."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.identifiability import Endpoint


ROOT = Path(__file__).resolve().parents[2]


def diagnose(directories):
    endpoint = Endpoint((6, 6), (0, 0), (0, 0), ())
    examples = []
    for target in (0, 'minimal'):
        for stream in (False, True):
            emitted = []
            result = enumerate_distance_mappings(
                [endpoint.graph(), endpoint.graph()], CD=target, binary=False,
                max_bijections=None, compute_minimum_cost=target == 'minimal',
                symmetry_pruning=True, expand_symmetry=True, max_mappings=1,
                collect_mappings=not stream,
                mapping_callback=(lambda mapping, cost: emitted.append(mapping)) if stream else None)
            examples.append({'target': target, 'stream': stream, 'output_cap': 1,
                             'literal_complete_set': [[0, 1], [1, 0]],
                             'reported_complete': result.complete, 'status': result.status,
                             'termination': result.truncation_reason,
                             'reported_mapping_count': result.selected_mapping_count,
                             'returned_maps': emitted if stream else result.mappings,
                             'false_completion': result.complete})
    studies = []
    for directory in directories:
        summary = json.loads((directory/'summary.json').read_text())
        hashes = summary['all_record_hashes']
        records = []
        for path in sorted((directory/'cases').glob('*.json')):
            if sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
                raise ValueError('Changed archived attempt')
            records.append(json.loads(path.read_text()))
        if len(records) != summary['attempts'] or len(records) != len(hashes):
            raise ValueError('Incomplete attempt inventory')
        reached = [r['task']['task_id'] for r in records
                   if r.get('mapping_count', 0) >= r['task']['max_maps']]
        studies.append({'directory': str(directory), 'attempts': len(records),
                        'summary_sha256': sha256((directory/'summary.json').read_bytes()).hexdigest(),
                        'maximum_saved_maps': max(r.get('mapping_count', 0) for r in records),
                        'attempts_reaching_output_cap': reached,
                        'scope': 'Exposure to this output-cap defect only; not a general correctness audit'})
    return {'schema': 'synister.output-cap-exposure.v1', 'diagnostic_cases': examples,
            'current_defect_reproduced': any(row['false_completion'] for row in examples),
            'studies': studies,
            'source_sha256': {path: sha256((ROOT/path).read_bytes()).hexdigest() for path in
                              ('synkit/Chem/Mapper/exact/distance.py',
                               'Experiment/Synister/audit_output_limit_exposure.py')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.studies)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
