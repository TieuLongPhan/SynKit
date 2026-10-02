"""E4 set/accounting audit plus intervention and observer identity checks."""

import argparse
import json
from pathlib import Path

from Experiment.Synister.audit_ablations import audit as audit_outputs
from Experiment.Synister.classify_enumeration import digest, encode


def audit(directory):
    result = audit_outputs(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    sources = json.loads((directory/'sources.json').read_text())
    variants = json.loads((directory/'variants.json').read_text())
    controls = Path(manifest['literal_controls'])
    if digest(controls/'summary.json') != manifest['literal_controls_sha256']:
        raise ValueError('Literal control provenance changed')
    control = json.loads((controls/'summary.json').read_text())
    if not control['all_passed']:
        raise ValueError('Unsuccessful literal controls')
    for name,expected in control['files_sha256'].items():
        if digest(controls/name)!=expected:
            raise ValueError('Changed literal control artifact')
    prior_sources = json.loads((controls/'sources.json').read_text())
    for name,content in prior_sources.items():
        if name!='Experiment/Synister/validate_isolated_ablations.py' and sources.get(name)!=content:
            raise ValueError('Executed runtime differs from controlled source')
    if variants != json.loads((controls/'variants.json').read_text()):
        raise ValueError('Experimental transformation differs from validated solver')
    if digest(Path(manifest['selection_source'])/'inputs.json')!=manifest['selection_sha256']:
        raise ValueError('Existing input selection changed')
    if json.loads((Path(manifest['selection_source'])/'inputs.json').read_text())!=json.loads((directory/'inputs.json').read_text()):
        raise ValueError('E4 differs from the forty fixed inputs')
    if manifest['instrumented']:
        ordinary = Path(manifest['ordinary'])
        if digest(ordinary/'audit.json')!=manifest['ordinary_audit_sha256']:
            raise ValueError('Ordinary-study gate changed')
    observed = 0
    for path in (directory/'cases').glob('*.json'):
        record = json.loads(path.read_text())
        if 'experimental_solver' not in record:
            if record['complete']:
                raise ValueError('Complete attempt lacks experimental source identity')
            continue
        variant = record['task']['variant']
        expected = {k:v for k,v in variants[variant].items() if k!='transformed_source'}
        if record['experimental_solver']!=expected or record['instrumented']!=manifest['instrumented']:
            raise ValueError('Executed intervention identity differs')
        observer = record['instrumentation']
        if not manifest['instrumented']:
            if observer is not None:
                raise ValueError('Ordinary headline attempt was instrumented')
            continue
        observed += 1
        phases = observer['phases']
        if record['complete']:
            if phases['enumeration']['counters']['visited_nodes'] != record['visited_nodes']:
                raise ValueError('Observer enumeration nodes disagree with solver')
            if record['task']['query']=='minimum' and phases['minimum_proof']['counters']['visited_nodes']!=record['minimum_proof_nodes']:
                raise ValueError('Observer proof nodes disagree with solver')
        if any(value<0 for phase in phases.values() for value in phase['timings'].values()):
            raise ValueError('Negative operation timing')
    result.update(schema='synister.isolated-ablation-audit.v1',intervention_sources_verified=True,
                  instrumented=manifest['instrumented'],observed_attempts=observed,
                  auditor_sha256=digest(Path(__file__)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.study)
    with args.output.open('x') as stream:
        stream.write(encode(result))
    print(encode({k:v for k,v in result.items() if k!='comparisons'}))
