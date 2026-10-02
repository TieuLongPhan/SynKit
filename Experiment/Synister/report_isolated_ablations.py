"""All-input E4 completion and separate, censored operation measurements."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from Experiment.Synister.audit_isolated_ablations import audit
from Experiment.Synister.classify_enumeration import digest, encode


def measured(values):
    known = [v for v in values if v is not None]
    return {'observed':len(known),'missing':len(values)-len(known),
            'median':median(known) if known else None,'maximum':max(known) if known else None}


def load(directory):
    checked = audit(directory)
    saved = json.loads((directory/'audit.json').read_text())
    if {k:v for k,v in saved.items() if k!='auditor_sha256'} != {k:v for k,v in checked.items() if k!='auditor_sha256'}:
        raise ValueError('Saved isolated-ablation audit differs from fresh accounting')
    if not checked['all_output_comparisons_consistent']:
        raise ValueError('Ablation output consistency failed')
    return (json.loads((directory/'manifest.json').read_text()),
            [json.loads(p.read_text()) for p in sorted((directory/'cases').glob('*.json'))],
            {'directory':str(directory),'manifest_sha256':digest(directory/'manifest.json'),
             'summary_sha256':digest(directory/'summary.json'),'audit_sha256':digest(directory/'audit.json')})


def summarize(records):
    return {'attempts':len(records),'complete':sum(r['complete'] for r in records),
            'minimum_proved':sum(r.get('minimum_proved',False) for r in records),
            'terminations':dict(Counter(r['termination'] for r in records)),
            'resources':{field:measured([r.get(field) for r in records]) for field in
                ('parent_seconds','worker_seconds','cpu_seconds','peak_rss_kib','visited_nodes',
                 'minimum_proof_nodes','seed_seconds','search_seconds','compile_seconds',
                 'mapping_count','output_bytes')}}


def build(ordinary, instrumented=None):
    manifest,records,provenance = load(ordinary)
    if manifest['instrumented']:
        raise ValueError('Headline study must be uninstrumented')
    rows = json.loads((ordinary/'inputs.json').read_text())
    variants = manifest['variants']
    table = {variant:{query:summarize([r for r in records if r['task']['variant']==variant
                                     and (query=='all' or r['task']['query']==query)])
                     for query in ('minimum','at_minimum','plus_2','all')} for variant in variants}
    original = {(r['task']['benchmark_id'],r['task']['query']):r for r in records if r['task']['variant']=='full'}
    pairs = []
    for record in records:
        task = record['task']
        if task['variant']=='full':
            continue
        baseline = original[task['benchmark_id'],task['query']]
        both = baseline['complete'] and record['complete']
        pairs.append({'benchmark_id':task['benchmark_id'],'query':task['query'],'variant':task['variant'],
            'full_complete':baseline['complete'],'variant_complete':record['complete'],
            'both_complete':both,'variant_over_full_parent_ratio':record['parent_seconds']/baseline['parent_seconds'] if both else None,
            'full_nodes':baseline.get('visited_nodes'),'variant_nodes':record.get('visited_nodes'),
            'node_scope':'Visited enumeration nodes; partial attempts are censored work, not completed-tree size.'})
    result = {'schema':'synister.isolated-ablation-report.v1','selected':len(rows),'variants':manifest['scope'],
        'ordinary_provenance':provenance,'completion':table,'paired_records':pairs,
        'unavailable_queries':json.loads((ordinary/'unavailable_queries.json').read_text()),
        'pair_summary':{variant:{'both_complete':sum(p['both_complete'] for p in pairs if p['variant']==variant),
            'full_only':sum(p['full_complete'] and not p['variant_complete'] for p in pairs if p['variant']==variant),
            'variant_only':sum(p['variant_complete'] and not p['full_complete'] for p in pairs if p['variant']==variant),
            'conditional_parent_ratio':measured([p['variant_over_full_parent_ratio'] for p in pairs if p['variant']==variant])}
            for variant in variants if variant!='full'},
        'scope':'Fixed forty inputs and original minimum-dependent target availability; all failures retained. Parent ratios condition on both complete outputs and are not population speed ratios. No independent-effect interpretation of ordered rejection counts.',
        'instrumentation':None}
    if instrumented is not None:
        observed_manifest,observed,observed_provenance = load(instrumented)
        if (not observed_manifest['instrumented'] or observed_manifest['ordinary_audit_sha256']!=provenance['audit_sha256']
                or json.loads((instrumented/'inputs.json').read_text())!=rows):
            raise ValueError('Instrumentation does not follow this ordinary study')
        measurements = {}
        for variant in variants:
            selected = [r for r in observed if r['task']['variant']==variant]
            available = [r for r in selected if r.get('instrumentation') is not None]
            phases = {}
            for phase in ('minimum_proof','enumeration'):
                values = [r['instrumentation']['phases'][phase] for r in available if phase in r['instrumentation']['phases']]
                phases[phase] = {'returned_phase_records':len(values)}
                for category in ('counters','timings'):
                    keys = sorted({key for value in values for key in value[category]})
                    phases[phase][category] = {key:{'sum':sum(value[category].get(key,0) for value in values),
                        **measured([value[category].get(key,0) for value in values])} for key in keys}
            measurements[variant] = {'attempts':len(selected),'returned_observations':len(available),
                'missing_observations':len(selected)-len(available),'complete':sum(r['complete'] for r in selected),
                'terminations':dict(Counter(r['termination'] for r in selected)),'phases':phases}
        result['instrumentation'] = {'provenance':observed_provenance,'measurements':measurements,
            'scope':'Separate five-second attempts. Counters describe visited work, including partial attempts. Inclusive operation times overlap and include tracing overhead; they cannot be summed into a time partition or used as headline runtimes.'}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ordinary',type=Path,required=True)
    parser.add_argument('--instrumented',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build(args.ordinary,args.instrumented)
    args.output.write_text(encode(result))
    print(encode({'selected':result['selected'],'completion':result['completion'],'pair_summary':result['pair_summary']}))
