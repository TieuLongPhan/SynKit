"""Typed unresolved phosphoenzyme ports, positional isotope exchange and cuts.

This connectivity projection does not supply Lewis/electron certificates or a
mapped enzyme residue. Native comparisons retain the same observed coordinates.
"""

from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.conditioned_coordinate_history import construct  # noqa: E402
from Experiment.MTG.frame_condition_compiler import AbsenceCondition  # noqa: E402
from Experiment.MTG.mod_encoding import (  # noqa: E402
    decode_graphs,
    expected_signature,
    graph_gml,
    rule_gml,
)
from Experiment.MTG.mod_history_comparison import _mod_signature  # noqa: E402
from Experiment.MTG.selenophosphate_isotope_case import (  # noqa: E402
    native_apply_with_frame,
)
from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph  # noqa: E402
from synkit.Graph.MTG.local import GroundedLocalEvent  # noqa: E402
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/phosphoenzyme_refinement_protocol.json"
SCHEMA = LabelSchema(
    node_identity=("carrier_kind", "element", "isotope"),
    node_state=("occupancy",),
    edge_state=("order", "bond_kind"),
    name="isotope-connectivity-unresolved-enzyme-port/1",
)
SITE = "unresolved-enzyme-port"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def value(graph):
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def same(left, right):
    return (
        left.schema == right.schema
        and left.nodes == right.nodes
        and left.edges == right.edges
    )


def initial(data, labelled=(), waters=None):
    graph = nx.Graph()
    heavy = {a["id"]: a for a in data["atoms"] if a["element"] != "H"}
    for node, atom in heavy.items():
        graph.add_node(
            node,
            carrier_kind="atom",
            element=atom["element"],
            isotope=18 if node in labelled else 0,
            occupancy="n/a",
        )
    for bond in data["bonds"]:
        a, b = bond["ends"]
        if a in heavy and b in heavy:
            phosphate = "P" in (heavy[a]["element"], heavy[b]["element"])
            graph.add_edge(
                a, b, order=1 if phosphate else bond["order"], bond_kind="chemical"
            )
    graph.add_node(
        SITE,
        carrier_kind="enzyme_port",
        element="unresolved",
        isotope=0,
        occupancy="free",
    )
    if waters is not None:
        for link, mass in zip(("ab", "bg"), waters):
            graph.add_node(
                f"water-{link}",
                carrier_kind="atom",
                element="O",
                isotope=mass,
                occupancy="n/a",
            )
    return graph


def bridge_state(graph, family, bridge):
    result = graph.copy()
    result.remove_edge(family["target"], family["initial_bridge"])
    result.add_edge(family["target"], bridge, order=1, bond_kind="chemical")
    return result


def changed(graph, deleted=(), added=(), occupancy=None):
    result = graph.copy()
    for a, b in deleted:
        assert result.has_edge(a, b), (a, b)
        result.remove_edge(a, b)
    for a, b, kind in added:
        assert not result.has_edge(a, b), (a, b)
        result.add_edge(a, b, order=1, bond_kind=kind)
    if occupancy is not None:
        result.nodes[SITE]["occupancy"] = occupancy
    return result


def event(name, before, after, nodes, absence=()):
    left, right = (value(g.subgraph(nodes).copy()) for g in (before, after))
    rule = RuleSpan.from_mapping(
        left, right, {n: n for n in nodes}, boundary="abstract", name=name
    )
    if absence:
        rule = replace(
            rule, condition=AbsenceCondition(tuple(frozenset(p) for p in absence))
        )
    return GroundedLocalEvent(name, rule, {n: n for n in nodes})


def observation(graph):
    def positions(a, b):
        bridge = [
            n
            for n in graph[a]
            if graph.nodes[n]["element"] == "O" and graph.has_edge(n, b)
        ]
        terminal = [
            n
            for n in graph[a]
            if graph.nodes[n]["element"] == "O"
            and len([p for p in graph[n] if graph.nodes[p]["element"] == "P"]) == 1
        ]
        return {
            "bridge": sorted(bridge),
            "labelled_terminal": sorted(
                n for n in terminal if graph.nodes[n]["isotope"] == 18
            ),
        }

    return {
        "alpha": positions("PA", "PB"),
        "beta": positions("PB", "PG"),
        "enzyme_occupancy": graph.nodes[SITE]["occupancy"],
    }


def free_phosphate(graph):
    components = [
        nodes
        for nodes in nx.connected_components(graph)
        if sum(graph.nodes[n]["element"] == "P" for n in nodes) == 1
        and not any(
            graph.nodes[n]["element"] == "C"
            or graph.nodes[n]["carrier_kind"] == "enzyme_port"
            for n in nodes
        )
    ]
    return {
        "total": len(components),
        "labelled": sum(
            any(
                graph.nodes[n]["element"] == "O" and graph.nodes[n]["isotope"] == 18
                for n in nodes
            )
            for nodes in components
        ),
    }


def native(mod, initial_world, events, expected, *, compose=False):
    hosts = mod.Graph.fromGMLStringMulti(graph_gml(initial_world), add=False)
    assert _mod_signature(hosts) == expected_signature(initial_world)
    first, maps, current, keep = [], [], hosts, []
    rows = []
    for e, target in zip(events, expected[1:]):
        actual_before = decode_graphs(current)
        if e.rule.condition is not None:
            assert not any(
                actual_before.has_edge(*tuple(p)) for p in e.rule.condition.pairs
            )
        gml, left, right = rule_gml(e.rule)
        rule = mod.ruleGMLString(gml, add=False)
        first.append(rule)
        maps.append((left, right))
        dg, current = native_apply_with_frame(mod, current, rule)
        keep.append(dg)
        assert _mod_signature(current) == expected_signature(target)
        graph = decode_graphs(current)
        assert observation(graph) == observation(target.to_networkx())
        assert free_phosphate(graph) == free_phosphate(target.to_networkx())
        rows.append(
            {"observation": observation(graph), "free_phosphate": free_phosphate(graph)}
        )
    composition = None
    if compose:
        assert len(first) == 2 and all(e.rule.condition is None for e in events)
        match = mod.RCMatch(*first)
        overlap = set(maps[0][1]) & set(maps[1][0])
        for key in sorted(overlap, key=repr):
            match.push(
                first[0].getVertexFromExternalId(maps[0][1][key]).right,
                first[1].getVertexFromExternalId(maps[1][0][key]).left,
            )
        outer = match.compose()
        assert outer is not None
        dg, outputs = native_apply_with_frame(mod, hosts, outer)
        assert _mod_signature(outputs) == expected_signature(expected[-1])
        composition = {
            "observation": observation(decode_graphs(outputs)),
            "all_coordinates_equal": True,
        }
    return {
        "worlds": rows,
        "composition": composition,
        "native_guard_adapter": any(e.rule.condition for e in events),
    }


def replay(initial_world, events):
    current, worlds = initial_world, [initial_world]
    for e in events:
        app = apply_dpo(e.rule, current, dict(e.match))
        assert app.certificate.replay().valid and app.certificate.resource_delta is None
        current = app.result
        worlds.append(current)
    return worlds


def check_cuts(initial_graph, events, history, expected_cuts):
    results = []
    for ideal, graph in expected_cuts:
        cut = history.configuration(ideal)
        assert same(cut.world, value(graph))
        composite = history.compile_ideal(ideal).rule
        app = apply_dpo(
            composite, value(initial_graph), {n: n for n in composite.left.node_ids}
        )
        assert same(app.result, cut.world) and app.certificate.replay().valid
        assert composite.boundary.value == "abstract"
        results.append(
            {
                "ideal": list(ideal),
                "observation": observation(graph),
                "free_phosphate": free_phosphate(graph),
                "direct_composite_agrees": True,
            }
        )
    return results


def pix_panel(mod, data, protocol):
    rows = []
    for label, labelled in protocol["labels"].items():
        for name, family in protocol["pix_families"].items():
            p, neighbour, occupancy = (
                family["target"],
                family["neighbour"],
                family["occupancy"],
            )
            for start, finish in product(family["return_positions"], repeat=2):
                before = bridge_state(initial(data, labelled), family, start)
                middle = changed(
                    before, [(p, start)], [(p, SITE, "binding")], occupancy
                )
                after = changed(middle, [(p, SITE)], [(p, finish, "chemical")], "free")
                events = (
                    event("capture", before, middle, (p, start, SITE)),
                    event(
                        "return",
                        middle,
                        after,
                        tuple({p, neighbour, start, finish, SITE}),
                    ),
                )
                history = construct(value(before), events)
                assert set(history.causal_graph.edges) == {("capture", "return")}
                cuts = check_cuts(
                    before,
                    events,
                    history,
                    [
                        ((), before),
                        (("capture",), middle),
                        (("capture", "return"), after),
                    ],
                )
                worlds = replay(value(before), events)
                assert same(worlds[-1], value(after))
                baseline = native(mod, value(before), events, worlds, compose=True)
                assert nx.is_isomorphic(
                    before,
                    after,
                    node_match=nx.algorithms.isomorphism.categorical_node_match(
                        ("carrier_kind", "element", "occupancy"), (None,) * 3
                    ),
                    edge_match=nx.algorithms.isomorphism.categorical_edge_match(
                        ("order", "bond_kind"), (None, None)
                    ),
                )
                rows.append(
                    {
                        "label": label,
                        "family": name,
                        "start_bridge": start,
                        "return_bridge": finish,
                        "cuts": cuts,
                        "native": baseline,
                        "isotope_masked_boundaries_isomorphic": True,
                    }
                )
    return rows


def fork_graphs(data, labels, *, retained_beta=False):
    before = initial(data, waters=labels)
    if retained_beta:
        middle = changed(
            before,
            [("PB", "O3B")],
            [("PB", "water-bg", "chemical"), ("PG", SITE, "binding")],
            "retained_gamma_phosphate",
        )
        nodes = ("PB", "O3B", "water-bg", "PG", SITE)
        absence = (("PB", "O3B"),)
    else:
        middle = changed(
            before, [("PG", "O3B")], [("PG", SITE, "binding")], "mono_gamma"
        )
        nodes = ("PG", "O3B", SITE)
        absence = (("PG", "O3B"),)
    beta = changed(middle, [("PB", "O3A")], [("PB", "water-ab", "chemical")])
    release_added = () if retained_beta else (("PG", "water-bg", "chemical"),)
    gamma = changed(middle, [("PG", SITE)], release_added, "free")
    final = changed(beta, [("PG", SITE)], release_added, "free")
    beta_nodes = {"PB", "O3A", "water-ab", "O3B", "PG"}
    if retained_beta:
        beta_nodes.add("water-bg")
    gamma_nodes = ("PG", SITE) if retained_beta else ("PG", SITE, "water-bg")
    events = (
        event("capture", before, middle, nodes),
        event("beta", middle, beta, tuple(beta_nodes), absence),
        event("gamma", middle, gamma, gamma_nodes),
    )
    cuts = [
        ((), before),
        (("capture",), middle),
        (("capture", "beta"), beta),
        (("capture", "gamma"), gamma),
        (("capture", "beta", "gamma"), final),
    ]
    return before, events, cuts


def fork_panel(mod, data, *, retained_beta=False):
    rows = []
    for labels in ((18, 18), (18, 0), (0, 18), (0, 0)):
        before, events, expected_cuts = fork_graphs(
            data, labels, retained_beta=retained_beta
        )
        history = construct(value(before), events)
        assert set(history.causal_graph.edges) == {
            ("capture", "beta"),
            ("capture", "gamma"),
        }
        cuts = check_cuts(before, events, history, expected_cuts)
        schedules = []
        for order in (events, (events[0], events[2], events[1])):
            worlds = replay(value(before), order)
            assert same(worlds[-1], value(expected_cuts[-1][1]))
            schedules.append(
                {
                    "order": [e.event_id for e in order],
                    "native": native(mod, value(before), order, worlds),
                }
            )
        row = {
            "labels": list(labels),
            "cuts": cuts,
            "schedules": schedules,
            "capture_precedes_two_branches": True,
        }
        if retained_beta:
            hidden = expected_cuts[2][1].copy()
            hidden.remove_node(SITE)
            full = cuts[2]["free_phosphate"]
            erased = free_phosphate(hidden)
            row["after_beta_retained_port"] = full
            row["after_beta_erased_port"] = erased
            row["old_beta_class_bound_fails"] = 2 * full["labelled"] > full["total"]
            assert erased["total"] == 2 and full["total"] == 1
        rows.append(row)
    return rows


def run():
    import mod

    protocol = json.loads(PROTOCOL.read_text())
    path = ROOT / protocol["input"]
    assert digest(path) == protocol["input_sha256"]
    data = json.loads(path.read_text())
    pix = pix_panel(mod, data, protocol)
    fork = fork_panel(mod, data)
    retained = fork_panel(mod, data, retained_beta=True)
    assert len(pix) == 54 and len(fork) == len(retained) == 4
    gamma_start = [
        r for r in pix if r["label"] == "gamma18" and r["start_bridge"] == "O3B"
    ]
    mono = [r for r in gamma_start if r["family"] == "mono_gamma"]
    pyro_gamma = [
        r for r in pix if r["label"] == "gamma18" and r["family"] == "pyro_beta_gamma"
    ]
    assert (
        len(mono) == 3
        and sum(
            "O3B" in r["cuts"][-1]["observation"]["beta"]["labelled_terminal"]
            for r in mono
        )
        == 2
    )
    assert all(
        r["cuts"][-1]["observation"]["beta"]["bridge"] == ["O3B"] for r in pyro_gamma
    )
    assert sum(r["old_beta_class_bound_fails"] for r in retained) == 3
    refused = False
    before, events, _ = fork_graphs(data, (18, 18))
    try:
        apply_dpo(events[1].rule, value(before), dict(events[1].match))
    except ValueError:
        refused = True
    assert refused
    pinned = [
        PROTOCOL,
        path,
        Path(__file__),
        ROOT / "Experiment/MTG/conditioned_coordinate_history.py",
        ROOT / "Experiment/MTG/compositional_condition_compiler.py",
        ROOT / "Experiment/MTG/frame_condition_compiler.py",
        ROOT / "Experiment/MTG/mod_encoding.py",
        ROOT / "Experiment/MTG/mod_history_comparison.py",
        ROOT / "Experiment/MTG/selenophosphate_isotope_case.py",
        ROOT / "synkit/Graph/MTG/local.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
    ]
    native_paths = [Path(mod.libpymod.__file__).resolve()]
    linked = native_paths[0].parents[1] / "libmod.so"
    if linked.is_file():
        native_paths.append(linked.resolve())
    return {
        "schema": "mtg.phosphoenzyme-refinement-evidence/1",
        "status": "PASS",
        "protocol": protocol,
        "pix": pix,
        "published_fork": fork,
        "formal_retention_control": retained,
        "counts": {
            "pix_cycles": 54,
            "pix_cuts": 162,
            "pix_native_applications": 108,
            "pix_native_compositions": 54,
            "fork_histories": 4,
            "fork_cuts": 20,
            "fork_schedules": 8,
            "fork_native_applications": 24,
            "retention_histories": 4,
            "retention_cuts": 20,
            "retention_schedules": 8,
            "retention_native_applications": 24,
        },
        "beta_before_gamma_capture_refused": refused,
        "source_boundaries": "ABSTRACT",
        "gamma_exchange_reachable": True,
        "pyro_only_gamma_exchange_reachable": False,
        "negative_pix_alone_excludes_intermediate": False,
        "reachability_implies_uniform_2_over_3_fraction": False,
        "old_coarse_bound_applies_to_expanded_family": False,
        "retention_control_is_published_covalent_intermediate": False,
        "complete_lewis_certificates": False,
        "physical_parallelism_established": False,
        "measured_kinetics_fitted": False,
        "comparative_advantage_established": False,
        "ncs_readiness_established": False,
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "mod": str(mod.version()),
            "native_sha256": {str(p): digest(p) for p in native_paths},
        },
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in pinned},
    }


if __name__ == "__main__":
    report = run()
    (ROOT / "Experiment/MTG/evidence/phosphoenzyme_refinement.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "old_coarse_bound_applies_to_expanded_family": False,
            }
        )
    )
