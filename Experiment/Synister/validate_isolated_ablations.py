"""Source-bound literal complete-set controls for the narrow E4 interventions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
from unittest.mock import patch

from Experiment.Synister import ablation_worker, validate_ablations
from Experiment.Synister.all_distance_oracle import binary_endpoint, weighted_cases
from Experiment.Synister.classify_enumeration import digest, encode
from Experiment.Synister.isolated_ablation import compile_variant, VARIANTS
from Experiment.Synister.worked_oracle import REACTION
from synkit.Chem.Mapper.exact import distance
from synkit.Chem.Mapper.identifiability import parse_reaction


def run(output, binary_pairs=32, weighted=16):
    output.mkdir(parents=True, exist_ok=False)
    functions = {v:compile_variant(v) for v in VARIANTS}
    sources = sorted(Path('synkit/Chem/Mapper').rglob('*.py'))
    sources += [Path('Experiment/Synister')/name for name in (
        'isolated_ablation.py','isolated_ablation_worker.py','ablation_observer.py','ablation_worker.py',
        'validate_isolated_ablations.py','validate_ablations.py','all_distance_oracle.py','global_milp.py','worked_oracle.py')]
    (output/'sources.json').write_text(encode({str(p):p.read_text() for p in sources}))
    (output/'variants.json').write_text(encode({v:identity for v,(_,identity) in functions.items()}))
    pairs = sorted(((a,b) for a in range(64) for b in range(64)),
                   key=lambda p:sha256(f'{p[0]},{p[1]}'.encode()).hexdigest())[:binary_pairs]
    (output/'manifest.json').write_text(encode({'schema':'synister.isolated-ablation-controls.v1',
        'binary_pairs':pairs,'weighted':weighted,'worked_reaction':REACTION,'variants':VARIANTS,
        'file_sha256':{name:digest(output/name) for name in ('sources.json','variants.json')}}))
    def enumerate_variant(r,p,*,variant,target):
        function,identity = functions[variant]
        with patch.object(distance,'enumerate_distance_mappings',function), patch.dict(
                ablation_worker.CONFIGURATIONS,{variant:identity['options']}):
            return ablation_worker.enumerate_variant(r,p,variant=variant,target=target)
    cases = [(f'binary_{a}_{b}',binary_endpoint(a),binary_endpoint(b),True) for a,b in pairs]
    cases += [(name,r,p,False) for name,r,p in weighted_cases(weighted)]
    cases.append(('worked',*parse_reaction(REACTION),False))
    counts = Counter()
    started = time.perf_counter()
    with (output/'cases.jsonl').open('x') as stream, patch.object(validate_ablations,'CONFIGURATIONS',VARIANTS), patch.object(
            validate_ablations,'enumerate_variant',enumerate_variant):
        for name,r,p,binary in cases:
            record = validate_ablations.check(name,r,p,binary=binary)
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            stream.flush()
            counts['cases'] += 1
            counts['queries'] += len(record['queries'])
            counts['failed_queries'] += sum(not q['passed'] for q in record['queries'])
            print(encode({'case':name,'passed':record['passed']}),flush=True)
    result = {'schema':'synister.isolated-ablation-controls.v1',**counts,
              'all_passed':counts['failed_queries']==0,'elapsed_seconds':time.perf_counter()-started,
              'files_sha256':{name:digest(output/name) for name in ('sources.json','variants.json','manifest.json','cases.jsonl')}}
    (output/'summary.json').write_text(encode(result))
    print(encode(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output)['all_passed'] else 1)
