"""Run the unmodified published AAM-Ising CPU route on small literal controls.

This is a compatibility audit, never a timing comparison. Run in the dedicated
external environment. No method is repaired or silently filtered into agreement.
"""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import distributions
import io
from itertools import combinations, permutations
import json
from pathlib import Path
import subprocess
import sys


COMMIT = '53f9c353564ea3a510f6310a4e53548430d588d9'
CONTROLS = [
    'C>>C', 'CC>>CC', 'CO>>CO', 'CCO>>CCO', 'CCO>>COC',
    'CCC>>CCC', 'CCCC>>CC(C)C', 'CCCO>>CCOC', 'CCN>>CNC',
    'COC>>CCO', 'CC.O>>CCO', 'C.C>>CC', 'C=C>>CC',
    'CC=O>>C=CO', 'CC(=O)O>>COC=O', 'C1CC1>>CCC',
    'CCCl>>C(C)Cl', 'NCCO>>CNCO', 'CC(=O)C>>CCC=O', 'O.O>>OO',
]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def literal(mapping):
    r, p = mapping.rct, mapping.prd
    if sorted(r.an.values()) != sorted(p.an.values()):
        raise ValueError('Unbalanced control')
    costs = {'binary': {}, 'bond_order': {}}
    for images in permutations(p.atoms):
        if any(r.an[i] != p.an[j] for i, j in zip(r.atoms, images)):
            continue
        mp = dict(zip(r.atoms, images))
        pairs = [(r.refer_bond_order(i, j), p.refer_bond_order(mp[i], mp[j]))
                 for i, j in combinations(r.atoms, 2)]
        costs['binary'][images] = sum(bool(a) != bool(b) for a, b in pairs)
        costs['bond_order'][images] = sum(abs(a-b) for a, b in pairs)
    return {name: {'minimum': min(values.values()),
                   'optimal_maps': sorted(m for m, cost in values.items() if cost == min(values.values())),
                   'cost_by_map': values} for name, values in costs.items()}


def check_maps(raw, mapping, oracle):
    valid, invalid = set(), []
    for row in raw:
        pairs = [tuple(x) for x in row]
        mp = dict(pairs)
        if (len(pairs) != len(mapping.rct.atoms) or len(mp) != len(pairs)
                or set(mp) != set(mapping.rct.atoms) or set(mp.values()) != set(mapping.prd.atoms)
                or any(mapping.rct.an[i] != mapping.prd.an[j] for i, j in pairs)):
            invalid.append(pairs)
        else:
            valid.add(tuple(mp[i] for i in mapping.rct.atoms))
    return {'raw_count': len(raw), 'unique_valid_count': len(valid), 'invalid_maps': invalid,
            'valid_maps': sorted(valid),
            'by_objective': {name: {'literal_minimum': data['minimum'],
                                    'literal_optimal_count': len(data['optimal_maps']),
                                    'complete_sets_equal': not invalid and valid == set(data['optimal_maps']),
                                    'missing_optimal_maps': sorted(set(data['optimal_maps'])-valid),
                                    'nonoptimal_maps': sorted(valid-set(data['optimal_maps'])),
                                    'returned_costs': sorted({data['cost_by_map'][m] for m in valid})}
                             for name, data in oracle.items()}}


def run(checkout):
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT or subprocess.check_output(['git', '-C', str(checkout), 'diff', 'HEAD'], text=True):
        raise ValueError('Comparator checkout is not the declared unmodified revision')
    sys.path.insert(0, str(checkout.resolve()))
    import MapIsing
    import optim_wrapper
    import networkx as nx
    rows = []
    for reaction in CONTROLS:
        record = {'reaction': reaction}
        capture = io.StringIO()
        with redirect_stdout(capture):
            try:
                mapping = MapIsing.Mapping(reaction)
                oracle = literal(mapping)
                record['endpoints'] = {name: {'atoms': obj.atoms, 'atomic_numbers': obj.an,
                                              'bonds': list(zip(obj.bonds, obj.order)), 'hydrogens': obj.dic_hyd}
                                       for name, obj in [('reactant', mapping.rct), ('product', mapping.prd)]}
                record['literal'] = {name: {k: v for k, v in data.items() if k != 'cost_by_map'}
                                     for name, data in oracle.items()}
                nodes, edges = mapping.modular_product()
                record.update(modular_nodes=nodes, modular_edges=edges)
                record['routes'] = {}
                # Published README constructs Graph(edges), which drops isolated
                # modular nodes. A separately named diagnostic retains them;
                # it is not substituted for the published route.
                for route in ('published_readme', 'retain_isolated_nodes_diagnostic'):
                    graph = nx.Graph(edges)
                    if route == 'retain_isolated_nodes_diagnostic':
                        graph.add_nodes_from(range(len(nodes)))
                    cliques, _ = optim_wrapper.MaxCliques(graph).find_maximum_cliques_cp()
                    known = list(nx.find_cliques(graph))
                    largest = max(map(len, known), default=0)
                    expected = {frozenset(c) for c in known if len(c) == largest}
                    item = {'graph_nodes': len(graph), 'maximum_cliques': [sorted(c) for c in cliques],
                            'clique_sets_agree_with_networkx': {frozenset(c) for c in cliques} == expected}
                    try:
                        raw = mapping.cliques_to_mappings(nodes, cliques)
                        item['unfiltered_indexed'] = check_maps(raw, mapping, oracle)
                        for option in ('filter1', 'filter2'):
                            try:
                                item[option] = check_maps(mapping.filtering(raw, option), mapping, oracle)
                            except Exception as exc:
                                item[option] = {'error': type(exc).__name__, 'message': str(exc)}
                    except Exception as exc:
                        item.update(error=type(exc).__name__, message=str(exc))
                    record['routes'][route] = item
            except Exception as exc:
                record.update(error=type(exc).__name__, message=str(exc))
        record['stdout'] = capture.getvalue()
        rows.append(record)
    files = {str(p.relative_to(checkout)): digest(p) for p in checkout.rglob('*.py')}
    return {'schema': 'synister.published-comparator-compatibility.v1',
            'repository': 'https://github.com/aki-27/AAM-Ising', 'commit': revision,
            'source_sha256': files, 'auditor_sha256': digest(Path(__file__)),
            'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in distributions()},
            'scope': 'Small compatibility controls on the unmodified published CPU/clique-to-map route; no performance measurement or superiority claim.',
            'controls_selected_before_execution': CONTROLS, 'rows': rows,
            'interpretation': {'objective': 'Default modular product uses bond presence and endpoint elements, not bond order. Binary and weighted literal optima are therefore reported separately.',
                               'filters': 'filter1 minimizes pendant-hydrogen discrepancy among returned candidates; filter2 then minimizes bond-order discrepancy. Neither proves unrestricted global weighted-CD optimality.',
                               'output': 'Indexed output before non_equivalent is checked to avoid conflating quotient conventions. Invalid or missing maps are retained as disagreements.',
                               'diagnostic': 'Adding isolated modular nodes tests graph construction only; it does not repair other conversion or completeness issues and is not a published-method result.'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    with args.output.open('x') as stream:
        stream.write(encoded)
    print(json.dumps({'controls': len(result['rows']), 'initialization_errors': sum('error' in r for r in result['rows'])}))
