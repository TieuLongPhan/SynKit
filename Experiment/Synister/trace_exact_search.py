"""Observe actual collected search with Python's trace hook, then audit literally.

Instrumentation never modifies solver globals, arguments or branching. Source
line anchors and SHA-256 make this intentionally implementation-specific. Its
elapsed times are not benchmark measurements. No trace mixes execution modes.
"""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import permutations, product
import json
import math
from pathlib import Path
import sys

from Experiment.Synister.benchmark_inputs import read_record, RECORD
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.exact import distance


SOURCE = Path(distance.__file__).resolve()


def number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else ('infinity' if value > 0 else '-infinity')


class SearchObserver:
    def __init__(self):
        lines = SOURCE.read_text().splitlines()
        anchors = {
            'committed_reject': 'pruned_branches += 1',
            'lower_reject': 'lower_bound_pruned_branches += 1',
            'upper_reject': 'upper_bound_pruned_branches += 1',
            'leaf': 'visited_leaves += 1',
            'candidates': 'for product_atom in candidate_images:',
            'symmetry_reject': 'symmetry_pruned_branches += 1',
            'tight_profile_reject': 'tight_profile_pruned += 1',
            'root_domains': 'dynamic_row_order = profile_pair_costs is not None and not certify',
        }
        self.anchors = {}
        for key, anchor in anchors.items():
            matches = [i+1 for i, line in enumerate(lines) if line.strip() == anchor]
            if len(matches) != 1:
                raise ValueError(f'Search trace anchor changed: {key}')
            self.anchors[matches[0]] = key
        self.nodes, self.active, self.candidate_rejections = [], {}, []

    def __call__(self, frame, event, arg):
        if frame.f_code.co_filename != str(SOURCE):
            return None
        name, local = frame.f_code.co_name, frame.f_locals
        if name == 'enumerate_distance_mappings':
            if event == 'line' and self.anchors.get(frame.f_lineno) == 'root_domains':
                for atom, domain in local['domains'].items():
                    for j in local['product_by_element'][local['reactant_elements'][atom]]:
                        if j not in local['fixed_images'] and j not in domain:
                            self.candidate_rejections.append({'parent_id': None, 'atom': atom, 'image': j,
                                'reason': 'root_profile_domain', 'source_line': frame.f_lineno,
                                'limit': number(local['profile_domain_limit']),
                                'forced_lower_bound': float(local['profile_edge_bounds'][atom, j])})
            return self
        if name not in ('visit', 'remaining_assignment_interval'):
            return None
        if name == 'remaining_assignment_interval':
            if event == 'return' and id(frame.f_back) in self.active:
                node = self.nodes[self.active[id(frame.f_back)]]
                blocks = []
                remaining, images = local['remaining_atoms'], local['remaining_products']
                for element in sorted({local['reactant_elements'][i] for i in remaining}):
                    rows = [i for i in remaining if local['reactant_elements'][i] == element]
                    columns = [i for i in images if local['product_elements'][i] == element]
                    matrix = [[float(local['cross_costs'][i, j]) for j in columns] for i in rows]
                    # Literal block assignments are separate audit calculations.
                    minimum = min(sum(matrix[i][j] for i, j in enumerate(p))
                                  for p in permutations(range(len(columns))))
                    blocks.append({'element': element, 'reactant_atoms': rows, 'product_atoms': columns,
                                   'cross_cost_matrix': matrix, 'literal_assignment_minimum': minimum,
                                   'independent_row_minima': sum(min(row) for row in matrix)})
                node['bound'] = {key: number(local.get(key)) for key in
                                 ('cross_lower', 'internal_lower', 'profile_lower', 'conditioned_lower', 'lower')}
                node['bound'].update(blocks=blocks, remaining_lower=number(arg[0]), remaining_upper=number(arg[1]),
                                     base_pruned=bool(local['base_pruned']),
                                     profile_evaluated='profile_lower' in local,
                                     conditioned_evaluated='conditioned_lower' in local)
            return self
        if event == 'call':
            node_id = len(self.nodes)
            self.active[id(frame)] = node_id
            self.nodes.append({'node_id': node_id, 'parent_id': self.active.get(id(frame.f_back)),
                               'phase': 'collected_minimum' if local['target'] == 'minimal' else 'supplied_cd',
                               'depth': local['depth'], 'prefix': [[i, int(j)] for i, j in enumerate(local['mapping']) if j >= 0],
                               'committed_cost': float(local['committed_cost']),
                               'incumbent_or_target': number(local['best_cost'] if local['target'] == 'minimal' else local['target']),
                               'strict_improvement': bool(local['strict_improvement']),
                               'decision': 'exhausted_children',
                               'stabilizer_generators': [list(g) for g in local['stabilizer_generators']]})
        elif id(frame) in self.active:
            node = self.nodes[self.active[id(frame)]]
            action = self.anchors.get(frame.f_lineno) if event == 'line' else None
            if action in ('committed_reject', 'lower_reject', 'upper_reject', 'leaf'):
                node['decision'] = action
                node['decision_source_line'] = frame.f_lineno
                if action == 'lower_reject':
                    b = node['bound']
                    node['first_decisive_bound'] = ('assignment_plus_residual' if b['base_pruned'] else
                        'incident_profile' if b['profile_lower'] is not None and b['profile_lower'] > local['limit']
                        else 'conditioned_profile')
            elif action == 'candidates' and 'candidate_domain' not in node:
                atom = int(local['reactant_atom'])
                node.update(next_reactant_atom=atom, candidate_domain=list(map(int, local['candidate_images'])),
                            active_stabilizer_generators=[list(g) for g in local['stabilizer_generators']])
                compatible = [j for j in local['domains'][atom] if not local['used_products'][j]]
                node['domain_before_dynamic_filter'] = list(map(int, compatible))
                for j in compatible:
                    if j not in node['candidate_domain']:
                        reason = ('conditioned_assignment_domain' if local['allowed_images'] is not None
                                  and j not in local['allowed_images'] else 'tight_profile_domain')
                        forced = (float(local['forced_bounds'][local['selected_position']-local['depth'],
                                                             list(local['remaining_images']).index(j)])
                                  if reason == 'conditioned_assignment_domain' else None)
                        self.candidate_rejections.append({'parent_id': node['node_id'], 'atom': atom, 'image': int(j),
                                                         'reason': reason, 'source_line': frame.f_lineno,
                                                         'limit': number(local['limit']), 'forced_lower_bound': forced})
            elif action in ('symmetry_reject', 'tight_profile_reject'):
                self.candidate_rejections.append({'parent_id': node['node_id'], 'atom': int(local['reactant_atom']),
                                                 'image': int(local['product_atom']), 'reason': action,
                                                 'source_line': frame.f_lineno, 'limit': number(local['limit'])})
            elif event == 'return':
                node['incumbent_on_return'] = number(local['best_cost'])
                if node['decision'] == 'leaf':
                    node['selected_at_visit'] = bool(local.get('selected', False))
                if local.get('timed_out'):
                    node['interruption'] = local['truncation_reason']
                    if node['decision'] == 'exhausted_children':
                        node['decision'] = 'interrupted'
                del self.active[id(frame)]
        return self


def literal_maps(r, p):
    groups = []
    for element in sorted(set(r.atomic_numbers)):
        rows = [i for i, z in enumerate(r.atomic_numbers) if z == element]
        columns = [i for i, z in enumerate(p.atomic_numbers) if z == element]
        groups.append([list(zip(rows, images)) for images in permutations(columns)])
    a, b = ({(i, j): w for i, j, w in mol.bonds} for mol in (r, p))
    output = []
    for blocks in product(*groups):
        mapping = dict(pair for block in blocks for pair in block)
        images = tuple(mapping[i] for i in range(len(r.atomic_numbers)))
        cost = sum(abs(a.get((i, j), 0)-b.get(tuple(sorted((images[i], images[j]))), 0))
                   for i in range(len(images)) for j in range(i+1, len(images))) / 2
        output.append((images, cost))
    return output


def audit(observer, literal):
    for node in observer.nodes:
        subtree = [cost for mapping, cost in literal if all(mapping[i] == j for i, j in node['prefix'])]
        if not subtree:
            raise ValueError('Visited prefix has no compatible completion')
        node['literal_subtree'] = {'completions': len(subtree), 'minimum_cd': min(subtree), 'maximum_cd': max(subtree)}
        if node['committed_cost'] > min(subtree):
            raise ValueError('Committed cost exceeds a literal completion')
        bound = node.get('bound')
        if bound:
            if sum(b['literal_assignment_minimum'] for b in bound['blocks']) != bound['cross_lower']:
                raise ValueError('Assignment relaxation differs from literal block minimum')
            if node['committed_cost']+bound['remaining_lower'] > min(subtree):
                raise ValueError('Recorded bound exceeds a literal completion')
        if node['decision'] in ('committed_reject', 'lower_reject'):
            if min(subtree) <= node['incumbent_or_target']:
                raise ValueError('Rejected subtree contains a permitted completion')
        if node['decision'] == 'upper_reject' and max(subtree) >= node['incumbent_or_target']:
            raise ValueError('Upper rejection contains a permitted completion')
    for event in observer.candidate_rejections:
        node = observer.nodes[event['parent_id']] if event['parent_id'] is not None else None
        prefix = (node['prefix'] if node else [])+[[event['atom'], event['image']]]
        subtree = [cost for mapping, cost in literal if all(mapping[i] == j for i, j in prefix)]
        event['literal_completions'] = len(subtree)
        event['literal_minimum_cd'] = min(subtree) if subtree else None
        if event['reason'] != 'symmetry_reject' and subtree and min(subtree) <= event['limit']:
            raise ValueError('Candidate filtering removes a permitted completion')


def run(target='minimal', max_mappings=None, case=None):
    worked = read_record()
    source = RECORD
    reaction = '>>'.join(worked['unmapped_smiles'])
    if case is not None:
        source = RECORD.parents[1]/'enumeration_main_v1/inputs.json'
        cases = json.loads(source.read_text())
        reaction = next(row['reaction'] for row in cases if row['benchmark_id'] == case)
    r, p = parse_reaction(reaction)
    options = dict(CD=target, binary=False, max_bijections=None, max_mappings=max_mappings,
                   tolerance=0, symmetry_pruning=False, collect_mappings=True,
                   compute_minimum_cost=target == 'minimal')
    plain = distance.enumerate_distance_mappings([r.graph(), p.graph()], **options)
    observer = SearchObserver()
    previous = sys.gettrace()
    try:
        sys.settrace(observer)
        traced = distance.enumerate_distance_mappings([r.graph(), p.graph()], **options)
    finally:
        sys.settrace(previous)
    if (traced.mappings != plain.mappings or traced.complete != plain.complete
            or traced.visited_nodes != plain.visited_nodes or traced.cost != plain.cost):
        raise ValueError('Instrumentation changed the search result')
    literal = literal_maps(r, p)
    audit(observer, literal)
    best = min(cost for _, cost in literal)
    expected = {m for m, cost in literal if cost == (best if target == 'minimal' else target)}
    if traced.complete and set(map(tuple, traced.mappings)) != expected:
        raise ValueError('Complete production set differs from the literal oracle')
    highlight = [[2, 7], [3, 0], [10, 11], [11, 8], [12, 9]]
    matches = [n['node_id'] for n in observer.nodes if n['prefix'] == highlight]
    return {'schema': 'synister.production-trace.v1', 'mode': 'collected_minimum' if target == 'minimal' else 'supplied_cd',
            'input_source': str(source.relative_to(Path(__file__).resolve().parents[2])), 'case_id': case,
            'input_sha256': sha256(source.read_bytes()).hexdigest(), 'unmapped_smiles': reaction.split('>>'),
            'solver_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
            'solver_module_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in SOURCE.parent.glob('*.py')},
            'observer_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'options': options, 'subgroup': {'order': 1, 'scope': 'Symmetry disabled in this trace; identity subgroup only.'},
            'instrumentation': 'Python line/call/return observer on unchanged production code; no performance timing.',
            'nodes': observer.nodes, 'candidate_rejections': observer.candidate_rejections,
            'highlight_node_ids': matches,
            'summary': {'complete': traced.complete, 'status': traced.status, 'truncation_reason': traced.truncation_reason,
                        'minimum_cd': best, 'returned_maps': traced.mappings, 'visited_nodes': traced.visited_nodes,
                        'literal_maps': len(literal), 'traced_untraced_equal': True,
                        'all_displayable_subtrees_literally_checked': True,
                        'decisions': dict(Counter(n['decision'] for n in observer.nodes))}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'minimum': run(), 'supplied_six': run(6), 'supplied_seven': run(7),
              'interrupted_six': run(6, max_mappings=1),
              'larger_compatible_space': run(case='reaction_058')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {'nodes': data['summary']['visited_nodes'], 'highlight': data['highlight_node_ids']}
                      for name, data in result.items()}))
