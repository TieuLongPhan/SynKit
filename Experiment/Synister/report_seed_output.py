"""All-attempt E2 reporting, with prediction costs and C1/C2 task differences."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from Experiment.Synister.audit_seed_output import audit
from Experiment.Synister.classify_enumeration import digest, encode


def measured(records, field):
    values = [r[field] for r in records if r.get(field) is not None]
    return {'observed': len(values), 'missing': len(records)-len(values),
            'median': median(values) if values else None, 'max': max(values) if values else None}


def summarize(records):
    return {'attempts': len(records), 'complete': sum(r['complete'] for r in records),
        'minimum_proved': sum(r.get('minimum_proved',False) for r in records),
        'terminations': dict(Counter(r['termination'] for r in records)),
        'resources': {field: measured(records, field) for field in (
            'parent_seconds','worker_seconds','cpu_seconds','peak_rss_kib','proof_seconds','enumeration_seconds',
            'expansion_seconds','writing_seconds','checking_seconds','seed_inclusive_parent_seconds',
            'prediction_already_available_parent_seconds','classification_parent_seconds','analysis_end_to_end_parent_seconds')}}


def representative_report(directory, provenance, records):
    summary = json.loads((directory/'summary.json').read_text())
    if summary['study_summary_sha256'] not in {p['summary_sha256'] for p in provenance}:
        raise ValueError('Representative classification belongs to another study')
    for name, expected in summary['files_sha256'].items():
        if digest(directory/name) != expected:
            raise ValueError('Representative classification artifact changed')
    by_task = {r['task']['task_id']: r for r in records}
    results = []
    for task in json.loads((directory/'tasks.json').read_text()):
        original = by_task[task['task_id']]
        if (not original['complete'] or original.get('output_unit') != 'verified_cyclic_subgroup_representatives'
                or original['mapping_sha256'] != task['mapping_sha256']
                or original['group'] != task['source_group']):
            raise ValueError('Representative classification source differs')
        result = json.loads((directory/(task['task_id']+'.json')).read_text())
        if result['task'] != task or result.get('structural_audit', {}).get('consistent') is False:
            raise ValueError('Representative classification has inconsistent evidence')
        if 'detail_sha256' in result and digest(directory/'details'/(task['task_id']+'.json')) != result['detail_sha256']:
            raise ValueError('Representative classification details differ')
        results.append(result)
    return {'directory': str(directory), 'summary_sha256': digest(directory/'summary.json'),
            'summary': summary, 'records': results,
            'timing_scope': 'Classification and independent audit are separately timed inside one worker. Parent time includes both; it is not added to application end-to-end time.'}


def build(directories, controls, representative_directory=None):
    inputs, records, provenance, classifications, seeds = {}, [], [], {}, {}
    for directory in directories:
        checked = audit(directory, controls)
        saved = json.loads((directory/'audit.json').read_text())
        if {k:v for k,v in checked.items() if k != 'auditor_sha256'} != {k:v for k,v in saved.items() if k != 'auditor_sha256'}:
            raise ValueError('Saved E2 audit differs from fresh accounting')
        provenance.append({'directory': str(directory), 'audit_sha256': digest(directory/'audit.json'),
                           'summary_sha256': digest(directory/'summary.json'), 'sources_sha256': digest(directory/'sources.json')})
        for row in json.loads((directory/'inputs.json').read_text()):
            bid = row['benchmark_id']
            if bid in inputs:
                raise ValueError('Do not pool repeated inputs or follow-up budgets as new reactions')
            inputs[bid] = row
            seeds[bid] = json.loads((directory/'preparation'/f'{bid}.seed.json').read_text())
        for path in (directory/'classification').glob('*.json'):
            result = json.loads(path.read_text())
            for alias in result['task']['aliases']:
                classifications[alias] = result
        records += [json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
    for record in records:
        classified = classifications.get(record['task']['task_id'])
        record['classification_parent_seconds'] = classified.get('parent_seconds') if classified else None
        record['classification_status'] = classified.get('termination') if classified else 'no_complete_indexed_input'
        record['classification_verified'] = classified.get('structural_audit',{}).get('verified',False) if classified else False
        record['analysis_end_to_end_parent_seconds'] = (record['seed_inclusive_parent_seconds']+classified['parent_seconds']
            if classified and classified.get('complete') else None)
    groups = defaultdict(list)
    for record in records:
        task = record['task']
        groups['.'.join(task[name] for name in ('stage','seed_condition','method','output'))].append(record)
    comparisons = []
    indexed = {r['task']['task_id']: r for r in records}
    for bid in sorted(inputs):
        for method in ('synister','milp'):
            pair = [indexed[f'{bid}.{seed}.{method}.indexed'] for seed in ('none','slap')]
            both = all(r['complete'] for r in pair)
            comparisons.append({'benchmark_id': bid, 'method': method, 'both_complete': both,
                'none_complete': pair[0]['complete'], 'slap_complete': pair[1]['complete'],
                'same_complete_set': pair[0]['mapping_sha256'] == pair[1]['mapping_sha256'] if both else None,
                'unseeded_over_seed_inclusive_parent_ratio': pair[0]['parent_seconds']/pair[1]['seed_inclusive_parent_seconds'] if both else None})
    historical = []
    for bid, row in sorted(inputs.items()):
        source = 'c1' if row['source'] == 'FlowER' else 'c2'
        directory = Path(f'paper/synister/evidence/identifiability_{source}_primary_v1')
        resources = json.loads(Path(f'paper/synister/evidence/{source}_resources_v1.json').read_text())
        path = directory/'cases'/f"{row['case_id']}.exact.json"
        if digest(path) != resources['record_hashes'][path.name]:
            raise ValueError('C1/C2 historical attempt changed')
        old = json.loads(path.read_text())
        historical.append({'benchmark_id': bid, 'source': row['source'], 'case_id': row['case_id'],
            'original_complete': old.get('enumeration_complete',False), 'original_minimum_proved': old.get('minimum_proved',False),
            'original_representatives': old.get('emitted_representatives'), 'original_seed_cd': old.get('initial_cost'),
            'original_parent_seconds': old['parent_seconds'], 'original_record_sha256': digest(path),
            'fresh_slap_seed_cd': seeds[bid].get('seed_doubled_cd',0)/2 if seeds[bid].get('seed_doubled_cd') is not None else None,
            'fresh_slap_seed_equals_original': seeds[bid].get('prediction',{}).get('mapping') == old.get('initial_mapping'),
            'fresh_slap_preparation_parent_seconds': seeds[bid]['parent_seconds']})
    rows = []
    for bid, row in sorted(inputs.items()):
        selected = [r for r in records if r['task']['benchmark_id'] == bid]
        rows.append({**row, 'seed_preparation': seeds[bid], 'attempts': selected})
    return {'schema':'synister.seed-output-report.v1', 'selected':len(inputs),
        'representative_classification': representative_report(representative_directory, provenance, records)
            if representative_directory is not None else None,
        'provenance': provenance, 'scope':'Native matched indexed outputs; staged diagnostics separate. Every selected input and failure retained.',
        'timing_scope':'Per-query seed-inclusive scenarios charge measured preparation parent time; shared classification costs are explicit scenario costs, not additional executions. Independent audit time is validation overhead, excluded from application end-to-end. Incomplete work is not a completion time.',
        'summary': {name:summarize(values) for name,values in sorted(groups.items())},
        'by_source': {source:{name:summarize([r for r in values if inputs[r['task']['benchmark_id']]['source']==source])
                             for name,values in sorted(groups.items())} for source in ('FlowER','Rhea')},
        'by_size': {str(size):{name:summarize([r for r in values if inputs[r['task']['benchmark_id']]['size_bin']==size])
                             for name,values in sorted(groups.items())} for size in range(5)},
        'paired_seed_comparisons':comparisons, 'historical_c1_c2_subset': historical,
        'historical_scope':'Same inputs, different runs/contracts: C1/C2 used best of two previously available predictions and collected product-group representatives; E2 uses fresh SLAP only and full indexed output, different subgroup policy and process cap. Do not equate the historical and matched task percentages.',
        'rows':rows}


def plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style=json.loads(Path('paper/synister/figures/style_tokens.json').read_text())
    palette=style['colors']
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'font.size':8.5,'axes.labelsize':8,'axes.linewidth':.6,
        'text.color':palette['text'],'axes.labelcolor':palette['text'],
        'xtick.color':palette['muted'],'ytick.color':palette['muted'],
        'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(6.5,3.7))
    groups=[('matched.none.synister.indexed','Synister, no external seed'),
            ('matched.slap.synister.indexed','Synister, SLAP seed'),
            ('matched.none.milp.indexed','MILP, no external seed'),
            ('matched.slap.milp.indexed','MILP, SLAP cutoff')]
    colors=[palette['teal'],palette['teal'],palette['purple'],palette['purple']]
    records=[r for row in data['rows'] for r in row['attempts']]
    for ax, field, title in zip(axes, ('parent_seconds','seed_inclusive_parent_seconds'),
                               ('Prediction already available','Preparation time charged')):
        for (key,label),color in zip(groups,colors):
            selected=[r for r in records if '.'.join(r['task'][n] for n in ('stage','seed_condition','method','output'))==key]
            times=sorted(r[field] for r in selected if r['complete'])
            end=max([r[field] for r in selected]+[1])
            ax.step([0]+times+[end],[0]+list(range(1,len(times)+1))+[len(times)],where='post',label=label,color=color,
                    linestyle='--' if '.none.' in key else '-',linewidth=1.2)
            for r in selected:
                if not r['complete']:
                    ax.plot(r[field],0,marker='|',color=color,alpha=.45,markersize=4)
        ax.set(xlabel='Recorded seconds per query',ylim=(-.5,data['selected']+.5))
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Complete indexed minimum sets')
    axes[1].set_yticklabels([])
    handles,labels = axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=7.2,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.32,top=.81,wspace=.17)
    for ax,letter,title in zip(axes,'AB',('Prediction already available','Preparation time charged')):
        pos=ax.get_position()
        fig.text(pos.x0-.075,pos.y1+.12,letter,color='white',fontsize=7.6,fontweight='bold',
                 ha='center',va='center',bbox={'boxstyle':'circle,pad=.33','facecolor':palette['navy'],'edgecolor':'none'})
        fig.text(pos.x0-.05,pos.y1+.12,title,fontsize=8.6,fontweight='bold',va='center')
    fig.text(.5,.02,f"All {data['selected']} selected inputs; ticks show incomplete attempts. Fixed-budget observations.",
             ha='center',fontsize=7.2,color=palette['muted'])
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,nargs='+',required=True)
    parser.add_argument('--controls',type=Path,default=Path('paper/synister/evidence/seed_output_controls_v1.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path,required=True)
    parser.add_argument('--representative-classification',type=Path)
    args=parser.parse_args()
    data=build(args.study,args.controls,args.representative_classification)
    args.output.write_text(encode(data))
    plot(data,args.figure)
    print(encode({name:{k:row[k] for k in ('attempts','complete','minimum_proved')} for name,row in data['summary'].items()}))
