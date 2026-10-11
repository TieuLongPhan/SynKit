#!/usr/bin/env python
"""Run the published catalytic-overlay code on shared supplied histories.

The common projection preserves elements, isotopes, charge, total bond order
and a fixed hydrogen inventory. Lone-pair/radical coordinates are not shared
query data. A formal trajectory pair tests aggregate information loss, not an
experimentally observed chemical mechanism or a performance advantage.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from Experiment.MTG.isotope_case import oxygen_role  # noqa: E402
from synkit.Graph.Morphism import (  # noqa: E402
    ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Graph.MTG import (  # noqa: E402
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceProcess,
    RuleOccurrence,
)
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402

UPSTREAM_COMMIT = "56683194d088e60ce033193f3eb9e84ef5a1f89d"
SYMBOLS = ("?", "-", "=", "#")


def _integer(value):
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(
            f"The chemical projection needs integer values, received {value!r}."
        )
    return int(value)


def _projection(value):
    graph = nx.Graph()
    for node in value.node_ids:
        attrs = value.node_labels(node)
        charge, mass = _integer(attrs["charge"]), _integer(attrs.get("isotope", 0))
        label = (
            (str(mass) if mass else "")
            + attrs["element"]
            + (("+" if charge > 0 else "-") * abs(charge))
        )
        graph.add_node(node, label=label)
        for index in range(_integer(attrs["hcount"])):
            hydrogen = ("implicit-h", node, index)
            graph.add_node(hydrogen, label="H")
            graph.add_edge(node, hydrogen, label="-")
    for edge in value.edge_keys:
        attrs = value.edge_labels(edge)
        order = _integer(attrs["sigma_order"] + attrs["pi_order"])
        if not 1 <= order <= 3:
            raise ValueError(
                "The common chemical projection supports single, double and triple bonds."
            )
        left, right = tuple(edge)
        graph.add_edge(left, right, label=SYMBOLS[order])
    return graph


def _rules(mod, certificates):
    keys = sorted(_projection(certificates[0].host), key=repr)
    indices = {key: index for index, key in enumerate(keys)}
    rules = []
    for certificate in certificates:
        before, after = _projection(certificate.host), _projection(certificate.result)
        if set(before) != set(keys) or set(after) != set(keys):
            raise ValueError(
                "This comparison requires fixed atoms and fixed per-atom unexpanded hydrogen counts."
            )
        parts = {"context": [], "left": [], "right": []}
        for node in keys:
            left, right = before.nodes[node]["label"], after.nodes[node]["label"]
            if (
                abs(
                    _integer(certificate.host.node_labels(node)["charge"])
                    - _integer(certificate.result.node_labels(node)["charge"])
                )
                > 1
                if node in certificate.host.node_ids
                else False
            ):
                raise ValueError(
                    "The published marking kernel's shared domain requires unit charge changes."
                )
            if left == right:
                parts["context"].append(f'node [ id {indices[node]} label "{left}" ]')
            else:
                parts["left"].append(f'node [ id {indices[node]} label "{left}" ]')
                parts["right"].append(f'node [ id {indices[node]} label "{right}" ]')
        edges = set(map(frozenset, before.edges)) | set(map(frozenset, after.edges))
        for edge in sorted(edges, key=repr):
            a, b = sorted(edge, key=repr)
            left = before.edges[a, b]["label"] if before.has_edge(a, b) else None
            right = after.edges[a, b]["label"] if after.has_edge(a, b) else None
            if (
                abs(SYMBOLS.index(left) if left else 0) > 3
                or abs(
                    (SYMBOLS.index(left) if left else 0)
                    - (SYMBOLS.index(right) if right else 0)
                )
                > 1
            ):
                raise ValueError("The shared domain requires unit bond-order changes.")

            def line(label):
                return (
                    f'edge [ source {indices[a]} target {indices[b]} label "{label}" ]'
                )

            if left == right:
                parts["context"].append(line(left))
            else:
                if left is not None:
                    parts["left"].append(line(left))
                if right is not None:
                    parts["right"].append(line(right))
        gml = (
            "rule [ "
            + " ".join(
                key + " [ " + " ".join(lines) + " ]" for key, lines in parts.items()
            )
            + " ]"
        )
        rules.append(mod.ruleGMLString(gml, add=False))
    return rules, indices


def _clean_product(overlay):
    graph = overlay.product_graph.copy()
    graph.remove_edges_from(
        [(a, b) for a, b, attrs in graph.edges(data=True) if attrs["label"] == "?"]
    )
    return graph


def _role_from_actual_output(mod, product, origin_to_vertex):
    from rdkit import Chem

    nodes = [
        f'node [ id {node} label "{attrs["label"]}" ]'
        for node, attrs in product.nodes(data=True)
    ]
    edges = [
        f'edge [ source {a} target {b} label "{attrs["label"]}" ]'
        for a, b, attrs in product.edges(data=True)
    ]
    graphs = mod.Graph.fromGMLStringMulti(
        "graph [ " + " ".join(nodes + edges) + " ]", add=False
    )
    periodic = Chem.GetPeriodicTable()
    decoded = nx.Graph()
    for origin, vertex in origin_to_vertex.items():
        if vertex not in product:
            continue
        actual = next(
            g.getVertexFromExternalId(vertex)
            for g in graphs
            if not g.getVertexFromExternalId(vertex).isNull()
        )
        decoded.add_node(
            origin,
            element=periodic.GetElementSymbol(int(actual.atomId)),
            isotope=max(0, int(actual.isotope)),
        )
    inverse = {vertex: origin for origin, vertex in origin_to_vertex.items()}
    for a, b, attrs in product.edges(data=True):
        if a in inverse and b in inverse:
            order = SYMBOLS.index(attrs["label"])
            decoded.add_edge(inverse[a], inverse[b], sigma_order=1, pi_order=order - 1)
    return oxygen_role(decoded) if 2 in decoded else "labelled origin not represented"


def _history(mod, api, name, certificates, observation=None):
    canonicaliser, cache, mechanism_class, step_class, compute = api
    assert all(certificate.replay().valid for certificate in certificates), name
    started = perf_counter()
    rules, indices = _rules(mod, certificates)
    encoding_seconds = perf_counter() - started
    origin_to_vertex = {
        n: rules[0].getVertexFromExternalId(indices[n]).id
        for n in certificates[0].host.node_ids
    }
    atom_maps = [
        {
            rule.getVertexFromExternalId(indices[n]).id: root
            for n, root in origin_to_vertex.items()
        }
        for rule in rules
    ]
    mechanism = mechanism_class(
        1, 1, [step_class(1, 1, index + 1, rule) for index, rule in enumerate(rules)]
    )
    started = perf_counter()
    overlays = list(compute(canonicaliser(), cache(), mechanism, atom_maps))
    execution_seconds = perf_counter() - started
    assert overlays, name
    expected = _projection(certificates[-1].result)
    root_all = {n: rules[0].getVertexFromExternalId(indices[n]).id for n in expected}
    expected = nx.relabel_nodes(expected, root_all)
    rows = []
    for overlay in overlays:
        product = _clean_product(overlay)
        # The published implementation intentionally removes completely inactive
        # components. Compare only its explicit retained-component domain.
        scoped_expected = expected.subgraph(overlay.host_graph.nodes).copy()
        matches = nx.utils.graphs_equal(scoped_expected, product)
        assert matches, name
        role = (
            _role_from_actual_output(mod, product, origin_to_vertex)
            if observation is not None
            else None
        )
        rows.append(
            {
                "overlay": overlay.serialise(),
                "retained_nodes": sorted(overlay.host_graph),
                "product_matches_shared_projection": matches,
                "labelled_oxygen_role": role,
                "observation_compatible": (
                    role == observation["labelled_product_role"]
                    if observation is not None
                    else None
                ),
            }
        )
    return {
        "name": name,
        "steps": len(certificates),
        "overlays": rows,
        "origin_to_upstream_vertex": {str(k): v for k, v in origin_to_vertex.items()},
        "encoding_seconds": encoding_seconds,
        "upstream_execution_seconds": execution_seconds,
    }


def _formal_trajectory(orders):
    values = []
    for order in orders:
        graph = nx.Graph()
        for atom, pairs in ((1, 3 - order), (2, 1)):
            graph.add_node(
                atom,
                element="C",
                aromatic=False,
                hcount=0,
                radical=0,
                lone_pairs=pairs,
                valence_electrons=4,
            )
        graph.add_edge(1, 2, sigma_order=1, pi_order=order - 1)
        values.append(
            LewisLabelledGraph.from_networkx(
                derive_electron_labeled_graph(graph), ELECTRON_LLG_SCHEMA
            )
        )
    materials = tuple(
        MaterialOccurrence(f"m{index}", value) for index, value in enumerate(values)
    )
    certificates, events = [], []
    for index, (before, after) in enumerate(zip(values, values[1:])):
        rule = RuleSpan.from_mapping(
            before, after, {1: 1, 2: 2}, boundary="closed", name=f"e{index}"
        )
        application = apply_dpo(rule, before, {1: 1, 2: 2})
        assert application.certificate.replay().valid
        certificates.append(application.certificate)
        events.append(
            RuleOccurrence(
                f"e{index}",
                rule,
                (MaterialBinding(f"m{index}", {1, 2}, {1: 1, 2: 2}),),
                (MaterialBinding(f"m{index+1}", {1, 2}, {1: 1, 2: 2}),),
            )
        )
    history = OccurrenceMTG.from_process(
        OccurrenceProcess("formal-bond-history", materials, tuple(events))
    )
    atom1 = history.node_lineage_by_carrier["m0", 1]
    atom2 = history.node_lineage_by_carrier["m0", 2]
    observed = []
    for length in range(len(events) + 1):
        cut = history.configuration_cut(
            event.occurrence_id for event in events[:length]
        )
        attrs = cut.world.edge_labels(frozenset((atom1, atom2)))
        observed.append(_integer(attrs["sigma_order"] + attrs["pi_order"]))
    assert observed == list(orders)
    return certificates, observed


def run(upstream, case_path, isotope_path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Use the pinned published overlay-graphs commit.")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--"],
        cwd=upstream,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    sys.path.insert(0, str(upstream))
    import mod
    from overlay_graphs.canonicalisation import GraphCanonicaliser
    from overlay_graphs.reaction_sequence_tracking import IsomorphismCache
    from overlay_graphs.mechanism import Mechanism, Step
    from overlay_graphs.make_overlay_graphs import compute_overlay_graphs

    api = (
        GraphCanonicaliser,
        IsomorphismCache,
        Mechanism,
        Step,
        compute_overlay_graphs,
    )
    cases = json.loads(case_path.read_text())
    isotope = json.loads(isotope_path.read_text())
    histories = []
    for case in cases["cases"]:
        for index, item in enumerate(case["interpretations"]):
            histories.append(
                _history(
                    mod,
                    api,
                    f"SynEPD {case['id']} interpretation {index+1}",
                    [_unpack_certificate(c) for c in item["certificates"]],
                )
            )
    for case in isotope["cases"]:
        histories.append(
            _history(
                mod,
                api,
                case["name"],
                [_unpack_certificate(c) for c in case["certificates"]],
                isotope["observation"],
            )
        )
    formal = []
    for index, orders in enumerate(((1, 2, 3, 2, 1), (1, 2, 1, 2, 1))):
        certificates, observed = _formal_trajectory(orders)
        row = _history(mod, api, f"formal aggregate witness {index+1}", certificates)
        row.update(
            mtg_bond_order_history=observed,
            ever_triple_bond=3 in observed,
            closed_certificates_replayed=len(certificates),
            source_states=[
                {
                    "nodes": [
                        {"id": node, "labels": value.node_labels(node)}
                        for node in sorted(value.node_ids)
                    ],
                    "edges": [
                        {"endpoints": sorted(edge), "labels": value.edge_labels(edge)}
                        for edge in value.edge_keys
                    ],
                }
                for value in [certificates[0].host]
                + [certificate.result for certificate in certificates]
            ],
        )
        formal.append(row)
    assert len(formal[0]["overlays"]) == len(formal[1]["overlays"]) == 1
    assert formal[0]["overlays"][0]["overlay"] == formal[1]["overlays"][0]["overlay"]
    assert formal[0]["ever_triple_bond"] != formal[1]["ever_triple_bond"]
    return {
        "schema": "synkit.mtg-published-overlay-comparison/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "mod": str(mod.version()),
            "networkx": nx.__version__,
        },
        "upstream": {
            "repository": "https://github.com/JuriKolcak/overlay-graphs",
            "commit": commit,
            "tracked_tree_unmodified": True,
            "module_sha256": {
                str(p.relative_to(upstream)): sha256(p.read_bytes()).hexdigest()
                for p in sorted((upstream / "overlay_graphs").glob("*.py"))
            },
        },
        "inputs": {
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in (case_path, isotope_path)
        },
        "shared_domain": (
            "Elements, isotopes, integer charge, total bond order, fixed per-atom hydrogen counts "
            "expanded as virtual identity H leaves, and pinned original atom correspondences. "
            "Completely inactive components are outside the upstream output domain."
        ),
        "not_shared_query_data": (
            "Radical/lone-pair coordinates, separate sigma/pi meanings, stereo, arbitrary guards "
            "and environment contracts; no advantage is claimed from their omission."
        ),
        "histories": histories,
        "formal_query_separation": formal,
        "summary": {
            "selected_histories": len(histories),
            "selected_steps": sum(h["steps"] for h in histories),
            "selected_overlays": sum(len(h["overlays"]) for h in histories),
            "formal_histories": 2,
            "formal_steps": 8,
            "aggregate_collision": True,
            "mtg_query_answers": [row["ever_triple_bond"] for row in formal],
            "selected_isotope_candidate_roles": [
                h["overlays"][0]["labelled_oxygen_role"] for h in histories[-2:]
            ],
        },
        "limits": [
            "Actual published compute_overlay_graphs implementation, not a locally reimplemented endpoint baseline.",
            "Both reconstructed isotope candidates retain the same role answers under the supplied mappings; "
            "no isotope advantage.",
            "The aggregate collision is a formal closed-electron C2-state example, "
            "not a demonstrated chemical mechanism or kinetics result.",
            "A full mapped trajectory also answers the distinguishing query; the result concerns the aggregate "
            "alone, not the full upstream pipeline with inputs retained.",
            "No performance superiority, population coverage, catalytic-substrate abstraction, "
            "semigroup or causal-reduction benchmark is established.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("Experiment/MTG/evidence/verified_cases.json"),
    )
    parser.add_argument(
        "--isotope-cases",
        type=Path,
        default=Path("Experiment/MTG/evidence/isotope_case.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.upstream, args.cases, args.isotope_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
