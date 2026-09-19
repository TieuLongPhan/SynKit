#!/usr/bin/env python3
"""Run one explicit uncapped, disk-backed reference-CD shell."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def implementation_hash(source):
    digest = hashlib.sha256()
    for path in sorted((source / 'synkit').rglob('*')):
        if path.suffix not in {'.py', '.cpp'}:
            continue
        name = path.relative_to(source).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(name).to_bytes(4, 'little'))
        digest.update(name)
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--dataset-sha256')
    parser.add_argument('--source-line', required=True, type=int)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', required=True, type=float,
                        help='Explicit revised wall allowance; not the original capped protocol')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a fresh directory')
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.dataset_sha256 and digest != args.dataset_sha256:
        parser.error('dataset hash mismatch')
    opener = gzip.open if args.dataset.suffix == '.gz' else open
    with opener(args.dataset, 'rt', encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['source_line']) == args.source_line]
    if len(rows) != 1:
        parser.error('source line must identify exactly one dataset row')
    row = rows[0]
    reaction = row['mapped_reaction']
    if '|' in reaction:
        reaction, identifier = reaction.rsplit('|', 1)
        if identifier != row['reaction_id']:
            parser.error('reaction suffix provenance mismatch')
    source = args.source_root.resolve()
    if not (source / 'synkit/__init__.py').is_file():
        parser.error('source root must contain synkit')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['SYNKIT_NATIVE_PATTERN_CACHE'] = '0'
    sys.path.insert(0, str(source))
    from synkit.Chem.Mapper import blinded_mapped_reaction_problem
    from synkit.Chem.Mapper.exact.disk_frontier import analyze_disk_reference_shell

    initial_hash = implementation_hash(source)
    problem = blinded_mapped_reaction_problem(reaction, heavy_only=True, blind_seed='synister-global-v1')
    manifest = {'source_line': args.source_line, 'reaction_id': row['reaction_id'], 'dataset_sha256': digest,
                'source_root': str(source), 'implementation_sha256': initial_hash, 'started_unix': time.time()}
    try:
        result = analyze_disk_reference_shell(
            problem.lgp, problem.reference_mapping, library_path=args.library,
            output=args.output, seconds=args.seconds, workers=args.workers,
        )
        manifest.update(complete=result['complete'], structure_complete=result['structure']['complete'])
    except Exception as error:
        manifest['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        manifest.update(source_unchanged=implementation_hash(source) == initial_hash, finished_unix=time.time())
        if args.output.is_dir():
            (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    return 0 if manifest['complete'] and manifest['structure_complete'] and manifest['source_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
