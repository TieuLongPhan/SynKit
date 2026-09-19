"""Fresh full-dataset V20 campaign; retain both shell modes and final audit."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
V = R.parent / 'synister_enhancement_v20_20260909'
D = R.parents[2] / 'Synister/data/flower_test_10000_v252.csv.gz'

def write(name, value):
    temp = R / (name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(R / name)

def main():
    digest = hashlib.sha256(D.read_bytes()).hexdigest()
    assert digest == 'e40647847169a7fc98af1aab44fa81c7a64deb70b77085ae3deb3925e39642b0'
    with gzip.open(D, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r['source_line'] for r in rows}) == 10000
    tasks = [dict(source_line=int(r['source_line']), reaction_id=r['reaction_id'], mode=m)
             for r in rows for m in ('minimal', 'reference_cd')]
    env = dict(os.environ, SYNKIT_NATIVE_PATTERN_CACHE='0', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('SYNKIT_NATIVE_PROFILE', None)
    processes, logs, launches = [], [], []
    started = time.time()
    for part, cpus in enumerate(('0-15', '16-31')):
        selection = f'selection_{part}.json'
        write(selection, dict(dataset=str(D), dataset_sha256=digest, tasks=tasks[part*10000:(part+1)*10000]))
        command = ['taskset', '-c', cpus, sys.executable, str(V / 'benchmark_synister_native.py'),
                   '--selection', str(R / selection), '--source-root', str(V / 'source'),
                   '--library', (V / 'library.txt').read_text().strip(), '--output', str(R / f'part_{part}'),
                   '--seconds', '60', '--workers', '16', '--mapping-cap', '1000000',
                   '--wall-budget', '--compact-json', '--immutable-json']
        log = (R / f'part_{part}.log').open('w')
        logs.append(log)
        processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        launches.append(dict(pid=processes[-1].pid, command=command))
    write('launch.json', dict(started_unix=started, reactions=10000, shells=20000, launches=launches))
    while True:
        timings = []
        for part in range(2):
            try:
                timings += json.loads((R / f'part_{part}/case_timings.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        status = dict(processed_shells=len(timings), total_shells=20000,
                      search_complete_below_60_seconds=sum(t['complete_below_60_seconds'] for t in timings),
                      wall_seconds=time.time()-started, return_codes=[p.poll() for p in processes])
        write('progress.json', status)
        print(json.dumps(status), flush=True)
        if all(p.poll() is not None for p in processes):
            break
        time.sleep(30)
    counts = {}
    seen = set()
    for part in range(2):
        for path in (R / f'part_{part}').glob('*_*.json'):
            if path.name == 'case_timings.json':
                continue
            rec = json.loads(path.read_text())
            key = (rec['source_line'], rec['mode'])
            assert key not in seen
            seen.add(key)
            result = rec.get('result', {})
            label = rec['mode'] + ':' + ('error' if 'error' in rec else result.get('status', 'unknown'))
            counts[label] = counts.get(label, 0) + 1
    manifests = [json.loads((R / f'part_{p}/manifest.json').read_text()) for p in range(2)]
    valid = (seen == {(t['source_line'], t['mode']) for t in tasks}
             and all(p.returncode == 0 for p in processes)
             and all(m.get('source_unchanged') for m in manifests))
    write('summary.json', dict(status, counts=counts, coverage_and_source_verified=valid))
    if not valid:
        raise RuntimeError('Campaign incomplete or source verification failed; see summary')

if __name__ == '__main__':
    main()
