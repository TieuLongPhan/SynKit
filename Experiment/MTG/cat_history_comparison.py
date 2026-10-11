#!/usr/bin/env python
"""Execute the published CAT kernels with equal supplied origin maps.

CAT's unrestricted transformation-word orbit and executable configurations of
one occurrence history are different domains. The adapter retains copy IDs;
it does not generate a native MØD network or infer atom correspondences.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.mcsa155_case import local_history  # noqa: E402
from Experiment.MTG.overlay_history_comparison import _projection  # noqa: E402
from Experiment.MTG.verified_cases import _process, _unpack_certificate  # noqa: E402
from synkit.Graph.MTG import OccurrenceMTG, cooccurrence_configuration  # noqa: E402


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def author_example(upstream):
    """Execute the original example's computation, omitting drawing only."""
    source = upstream / "examples/test.py"
    tree = ast.parse(source.read_text())
    retained = []
    stopped = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "scay" for t in node.targets
        ):
            stopped = True
            break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value.func
            if isinstance(call, ast.Name) and call.id == "print":
                continue
            if isinstance(call, ast.Attribute) and call.attr in (
                "print",
                "print_dg",
                "postSection",
            ):
                continue
        retained.append(node)
    assert stopped
    namespace = {"__file__": str(source)}
    exec(
        compile(ast.Module(body=retained, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    graph = namespace["pcay"].get_nxgraph()
    assert len(namespace["gens"]) == 6
    assert graph.number_of_nodes() == 8 and graph.number_of_edges() == 14
    return {
        "source_sha256": digest(source),
        "execution": "Original AST through pcay.calc; drawing calls omitted.",
        "generators": 6,
        "orbit_states": sorted(d["atom_state"] for _, d in graph.nodes(data=True)),
        "edges": graph.number_of_edges(),
    }


def material_graphs(mod, history):
    vertices, carrier_to_vertex, vertex_to_carrier = [], {}, {}
    for material in history.materials:
        projection = _projection(material.value)
        indices = {n: i for i, n in enumerate(sorted(projection, key=repr))}
        nodes = [
            f'node [ id {indices[n]} label "{d["label"]}" ]'
            for n, d in projection.nodes(data=True)
        ]
        edges = [
            f'edge [ source {indices[a]} target {indices[b]} label "{d["label"]}" ]'
            for a, b, d in projection.edges(data=True)
        ]
        graphs = mod.Graph.fromGMLStringMulti(
            "graph [ " + " ".join(nodes + edges) + " ]", add=False
        )
        assert len(graphs) == 1
        graph = graphs[0]
        graph.name = material.occurrence_id
        vertices.append(SimpleNamespace(graph=graph))
        for node in material.value.node_ids:
            point = graph.getVertexFromExternalId(indices[node])
            assert not point.isNull()
            carrier = (material.occurrence_id, node)
            carrier_to_vertex[carrier] = point
            vertex_to_carrier[point] = carrier
    return vertices, carrier_to_vertex, vertex_to_carrier


class EventView:
    def __init__(self, event, sources, targets):
        self.event = event
        self.sources = sources
        self.targets = targets


def endpoint_carrier(bindings, node):
    found = [
        (b.material_id, dict(b.mapping)[node])
        for b in bindings
        if node in dict(b.mapping)
    ]
    assert len(found) == 1
    return found[0]


def cat_inputs(mod, history, api):
    GeneratorSet, _, Transformation = api
    vertices, to_vertex, to_carrier = material_graphs(mod, history)
    by_id = {v.graph.name: v for v in vertices}
    edges = [
        EventView(
            event,
            tuple(by_id[b.material_id] for b in event.inputs),
            tuple(by_id[b.material_id] for b in event.outputs),
        )
        for event in history.events
    ]
    network = SimpleNamespace(vertices=vertices, edges=edges)
    # This original input API accepts supplied maps. Native graph values remain
    # chemical; the network view preserves occurrence IDs without an isomorphism
    # quotient or an invalid TrustMe duplicate-species database.
    point_map = GeneratorSet(
        network, transformations=[], fun_ignore=lambda v: v not in to_carrier
    )
    transformations = []
    for index, event in enumerate(history.events):
        pairs = [
            (
                endpoint_carrier(event.inputs, event.rule.left_arm.mapping[k]),
                endpoint_carrier(event.outputs, event.rule.right_arm.mapping[k]),
            )
            for k in event.rule.interface.node_ids
        ]
        assert (
            len(pairs)
            == len(event.rule.left.node_ids)
            == len(event.rule.right.node_ids)
        )
        left = [point_map.get_id(to_vertex[a]) for a, _ in pairs]
        right = [point_map.get_id(to_vertex[b]) for _, b in pairs]
        assert len(set(left)) == len(left) and len(set(right)) == len(right)
        transformations.append(Transformation(index, left, right))
    generators = GeneratorSet(
        network,
        transformations=transformations,
        fun_ignore=lambda v: v not in to_carrier,
    )
    carriers = {generators.get_id(v): c for v, c in to_carrier.items()}
    return generators, to_vertex, carriers


def cut_tuples(history, generators, to_vertex, roots):
    graph = history.to_process().causal_graph
    order = tuple(nx.lexicographical_topological_sort(graph))
    assert all(nx.has_path(graph, a, b) for a, b in zip(order, order[1:]))
    root_lineages = [history.node_lineage_by_carrier[c] for c in roots]
    tuples = {}
    for index in range(len(order) + 1):
        cut = history.configuration_cut(order[:index])
        states = {lineage.lineage_id: state for lineage, state in cut.node_states}
        points = tuple(
            generators.get_id(
                to_vertex[(states[lineage].material_id, states[lineage].carrier)]
            )
            for lineage in root_lineages
        )
        tuples[points] = sorted(cut.executed_events)
    return tuples


def structural_filter(history, carriers, states):
    graph = history.to_process().causal_graph
    producer = {
        b.material_id: e.occurrence_id for e in history.events for b in e.outputs
    }
    consumer = {
        b.material_id: e.occurrence_id for e in history.events for b in e.inputs
    }
    accepted, refused = {}, {}
    for state in states:
        copies = {carriers[p][0] for p in state}
        downset = set()
        for copy in copies:
            if copy in producer:
                event = producer[copy]
                downset.add(event)
                downset.update(nx.ancestors(graph, event))
        blockers = sorted({consumer[c] for c in copies if consumer.get(c) in downset})
        if blockers:
            refused[state] = {
                "required_events": sorted(downset),
                "blocking_consumers": blockers,
            }
        else:
            accepted[state] = sorted(downset)
    # Exercise the public validated query on a representative of each class.
    for table, possible in ((accepted, True), (refused, False)):
        if table:
            state = min(table)
            witness = cooccurrence_configuration(
                history, {carriers[p][0] for p in state}
            )
            assert (witness is not None) == possible
            if witness is not None:
                assert sorted(witness.executed_events) == accepted[state]
    return accepted, refused


def neighbors(vertex):
    return [
        (edge.target if edge.source == vertex else edge.source, edge.stringLabel)
        for edge in vertex.incidentEdges
    ]


def actual_query(generators, state, kind):
    if kind == "isotope":
        oxygen, carbon = [generators.get_vertex(p) for p in state]
        if int(oxygen.atomId) != 8 or int(oxygen.isotope) != 18:
            return "missing labelled origin"
        incident = neighbors(oxygen)
        if any(target == carbon and label == "=" for target, label in incident):
            return "carbonyl"
        if len(incident) == 2 and all(
            int(target.atomId) == 6 and label == "-" for target, label in incident
        ):
            return "ether"
        return "other"
    if kind == "transferase":
        carbon, sulfur, _ = [generators.get_vertex(p) for p in state]
        edges = neighbors(carbon)
        return any(
            target == sulfur and label == "-" for target, label in edges
        ) and any(int(target.atomId) == 8 and label == "=" for target, label in edges)
    return None


def history_case(mod, api, name, packed, origins, kind):
    started = perf_counter()
    certificates = [_unpack_certificate(c) for c in packed]
    assert all(c.replay().valid for c in certificates)
    worlds = [certificates[0].host.to_networkx()] + [
        c.result.to_networkx() for c in certificates
    ]
    if kind == "isotope":
        # Keep the source isotope schema: the electron-only fragment adapter
        # must not erase semantic mass. The supplied isotope events are global.
        values = [certificates[0].host] + [c.result for c in certificates]
        history = OccurrenceMTG.from_process(
            _process(values, [c.rule for c in certificates])
        )
    else:
        history, _ = local_history(worlds, process_id=name)
    history_seconds = perf_counter() - started
    started = perf_counter()
    generators, to_vertex, carriers = cat_inputs(mod, history, api)
    roots = []
    for origin in origins:
        found = [
            c
            for c in to_vertex
            if c[0] in history.initial_material_ids and c[1] == origin
        ]
        assert len(found) == 1
        roots.append(found[0])
    encoding_seconds = perf_counter() - started
    started = perf_counter()
    reference = cut_tuples(history, generators, to_vertex, roots)
    reference_seconds = perf_counter() - started
    _, ProjectedCayleyGraph, _ = api
    cayley = ProjectedCayleyGraph([to_vertex[c] for c in roots], generators)
    started = perf_counter()
    cayley.calc()
    graph = cayley.get_nxgraph()
    kernel_seconds = perf_counter() - started
    states = {d["atom_state"] for _, d in graph.nodes(data=True)}
    assert set(reference) <= states
    started = perf_counter()
    accepted, refused = structural_filter(history, carriers, states)
    filter_seconds = perf_counter() - started
    assert set(accepted) == set(reference)
    # Every original atom map preserves its actual supplied lineage.
    root_lineages = [history.node_lineage_by_carrier[c] for c in roots]
    assert all(
        [history.node_lineage_by_carrier[carriers[p]] for p in state] == root_lineages
        for state in states
    )
    last = tuple(generators.get_id(to_vertex[c]) for c in roots)
    for event in history.events:
        edge = next(e for e in generators.dg.edges if e.event is event)
        (transform,) = generators[edge]
        last = tuple(transform[p] for p in last)
    assert last in reference
    final_answer = actual_query(generators, last, kind)
    cut_answers = [actual_query(generators, s, kind) for s in sorted(reference)]
    orbit_answers = [actual_query(generators, s, kind) for s in sorted(states)]
    state_ids = {d["atom_state"]: n for n, d in graph.nodes(data=True)}
    witness = None
    if refused:
        selected = min(refused)
        path = nx.shortest_path(
            graph, state_ids[cayley.initialState._atom_state], state_ids[selected]
        )
        witness = {
            "state": selected,
            "carriers": [carriers[p] for p in selected],
            "cat_path_states": [graph.nodes[n]["atom_state"] for n in path],
            "generator_ids": [
                graph.edges[a, b]["label"] for a, b in zip(path, path[1:])
            ],
            **refused[selected],
        }
    return {
        "name": name,
        "supplied_steps": len(certificates),
        "origins": origins,
        "generators": len(generators),
        "material_copies": len(history.materials),
        "orbit_states": len(states),
        "orbit_edges": graph.number_of_edges(),
        "executable_projected_states": len(reference),
        "outside_prefix_domain": len(refused),
        "filtered_orbit_equals_all_cut_projections": True,
        "fixed_trace_final_answer": final_answer,
        "executable_cut_has_thioester": (
            any(cut_answers) if kind == "transferase" else None
        ),
        "unrestricted_orbit_has_thioester": (
            any(orbit_answers) if kind == "transferase" else None
        ),
        "material_semantics": (
            "Supplied global isotope events with their original semantic isotope schema."
            if kind == "isotope"
            else "Declared whole-component local fragments with unchanged identity frames."
        ),
        "out_of_domain_witness": witness,
        "state_positions": [
            {
                "state": s,
                "carriers": [carriers[p] for p in s],
                "least_configuration": accepted.get(s),
                "prefix_compatible": s in accepted,
            }
            for s in sorted(states)
        ],
        "operational_seconds": {
            "certificate_checks_and_history": history_seconds,
            "encoding": encoding_seconds,
            "reference_cut_projection": reference_seconds,
            "cat_kernel": kernel_seconds,
            "filter_including_public_checks": filter_seconds,
        },
    }


def run(upstream, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    assert commit == protocol["upstream_commit"]
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--"],
        cwd=upstream,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for name, expected in protocol["inputs"].items():
        assert digest(ROOT / name) == expected, name
    sys.path.insert(0, str(upstream))
    import mod
    from cat import GeneratorSet, ProjectedCayleyGraph
    from cat.transformation import Transformation

    api = (GeneratorSet, ProjectedCayleyGraph, Transformation)
    original = author_example(upstream)

    def read(name):
        return json.loads((ROOT / f"Experiment/MTG/evidence/{name}.json").read_text())

    cases, isotope, transferase = (
        read("verified_cases"),
        read("isotope_case"),
        read("mcsa155_case"),
    )
    rows = []
    for case in cases["cases"]:
        for index, item in enumerate(case["interpretations"]):
            first = _unpack_certificate(item["certificates"][0])
            origins = sorted(first.host.node_ids)[:3]
            rows.append(
                history_case(
                    mod,
                    api,
                    f"SynEPD {case['id']} interpretation {index+1}",
                    item["certificates"],
                    origins,
                    "construction",
                )
            )
    for case in isotope["cases"]:
        row = history_case(
            mod, api, case["name"], case["certificates"], [2, 1], "isotope"
        )
        assert row["fixed_trace_final_answer"] == case["mtg_oxygen_roles"][-1]
        rows.append(row)
    for case in transferase["candidates"]:
        row = history_case(
            mod,
            api,
            f"MCSA155 source ID {case['proposal']}",
            case["certificates"],
            [71, 48, 52],
            "transferase",
        )
        assert row["executable_cut_has_thioester"] == case["observation_supported"]
        assert row["unrestricted_orbit_has_thioester"] == case["observation_supported"]
        rows.append(row)
    return {
        "schema": "synkit.mtg-cat-history-comparison/1",
        "status": "PASS",
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(protocol_path),
        "inputs": protocol["inputs"],
        "adapter_sha256": {
            n: digest(ROOT / f"Experiment/MTG/{n}.py")
            for n in ("mcsa155_case", "overlay_history_comparison", "verified_cases")
        },
        "source_sha256": {
            n: digest(ROOT / f"synkit/Graph/MTG/{n}.py")
            for n in ("history", "process", "queries")
        },
        "upstream": {
            "repository": protocol["upstream_repository"],
            "commit": commit,
            "tracked_tree_unmodified": True,
            "module_sha256": {
                str(p.relative_to(upstream)): digest(p)
                for p in sorted((upstream / "cat").glob("*.py"))
            },
        },
        "environment": {
            "python": platform.python_version(),
            "mod": str(mod.version()),
            "networkx": nx.__version__,
        },
        "author_example": original,
        "histories": rows,
        "summary": {
            "histories": len(rows),
            "supplied_steps": sum(r["supplied_steps"] for r in rows),
            "orbit_states": sum(r["orbit_states"] for r in rows),
            "executable_projected_states": sum(
                r["executable_projected_states"] for r in rows
            ),
            "outside_prefix_domain": sum(r["outside_prefix_domain"] for r in rows),
            "isotope_roles": [r["fixed_trace_final_answer"] for r in rows[6:8]],
            "transferase_queries": [
                r["executable_cut_has_thioester"] for r in rows[8:]
            ],
        },
        "claim_limits": (
            "Actual unmodified CAT explicit-map kernels over an occurrence-network view; "
            "no automatic map inference or native chemical-network generation. "
            "Unrestricted monoid states outside finite prefix semantics are domain differences, "
            "not CAT errors or physically observed false predictions. "
            "CAT plus the declared occurrence control agrees; no complete-pipeline superiority."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "Experiment/MTG/cat_comparison_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.upstream, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
