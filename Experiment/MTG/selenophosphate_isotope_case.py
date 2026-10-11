"""Independent solvent-oxygen observation on supplied ATP hydrolysis histories.

Net macroevents are author reconstructions. The isotope experiment constrains
their product positions; it does not measure their order or physical concurrency.
"""

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
from Experiment.MTG.mod_encoding import (  # noqa: E402
    decode_graphs,
    expected_signature,
    graph_gml,
    rule_gml,
)
from Experiment.MTG.mod_history_comparison import _mod_signature  # noqa: E402
from synkit.Graph.Morphism import (  # noqa: E402
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Graph.MTG.local import GroundedLocalEvent  # noqa: E402
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/selenophosphate_isotope_protocol.json"
VALENCE = {"H": 1, "C": 4, "N": 5, "O": 6, "P": 5}
ORDERS = (("ab", "bg"), ("bg", "ab"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def llg(graph):
    value = graph.copy()
    for _, attrs in value.nodes(data=True):
        for key in ISOTOPE_ELECTRON_LLG_SCHEMA.node_derived:
            attrs.pop(key, None)
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(value), ISOTOPE_ELECTRON_LLG_SCHEMA
    )


def initial_graph(data, isotopes):
    graph = nx.Graph()
    for atom in data["atoms"]:
        graph.add_node(atom["id"], element=atom["element"], isotope=0)
    for bond in data["bonds"]:
        graph.add_edge(*bond["ends"], sigma_order=1, pi_order=bond["order"] - 1)
    for atom in data["atoms"]:
        node = atom["id"]
        valence = VALENCE[atom["element"]]
        order = sum(e["sigma_order"] + e["pi_order"] for e in graph[node].values())
        residual = valence - atom["charge"] - order
        assert residual >= 0 and residual % 2 == 0, node
        graph.nodes[node].update(
            aromatic=False,
            hcount=0,
            radical=0,
            lone_pairs=residual // 2,
            valence_electrons=valence,
        )
    for link, mass in zip(("ab", "bg"), isotopes):
        oxygen, h1, h2 = water_nodes(link)
        graph.add_node(
            oxygen,
            element="O",
            isotope=mass,
            aromatic=False,
            hcount=0,
            radical=0,
            lone_pairs=2,
            valence_electrons=6,
        )
        for hydrogen in (h1, h2):
            graph.add_node(
                hydrogen,
                element="H",
                isotope=0,
                aromatic=False,
                hcount=0,
                radical=0,
                lone_pairs=0,
                valence_electrons=1,
            )
            graph.add_edge(oxygen, hydrogen, sigma_order=1, pi_order=0)
    value = llg(graph)
    assert all(value.node_labels(n)["charge"] == 0 for n in value.node_ids)
    return value


def water_nodes(link):
    return tuple(f"water-{link}-{suffix}" for suffix in ("O", "H1", "H2"))


def edit(graph, link, target, links):
    """Independent connectivity oracle: two deleted and two created bonds."""
    result = graph.copy()
    _, bridge, _ = links[link]
    oxygen, proton, _ = water_nodes(link)
    assert result.has_edge(target, bridge) and result.has_edge(oxygen, proton)
    assert not result.has_edge(target, oxygen) and not result.has_edge(bridge, proton)
    result.remove_edge(target, bridge)
    result.remove_edge(oxygen, proton)
    result.add_edge(target, oxygen, sigma_order=1, pi_order=0)
    result.add_edge(bridge, proton, sigma_order=1, pi_order=0)
    return result


def macro(initial, link, target, links):
    _, bridge, _ = links[link]
    oxygen, proton, _ = water_nodes(link)
    nodes = (target, bridge, oxygen, proton)
    before = initial.to_networkx()
    after = edit(before, link, target, links)
    left, right = (llg(g.subgraph(nodes).copy()) for g in (before, after))
    # Pattern-derived charges describe incomplete local patterns. Source matching
    # uses the authoritative Lewis coordinates; complete host charges are derived.
    rule = RuleSpan.from_mapping(
        left,
        right,
        {n: n for n in nodes},
        boundary="closed",
        name=link,
    )
    return GroundedLocalEvent(link, rule, {n: n for n in nodes})


def same(left, right):
    return (
        left.schema == right.schema
        and left.nodes == right.nodes
        and left.edges == right.edges
    )


def roles(graph):
    """Classify actual anchored solvent oxygens by product component chemistry."""
    result = {}
    for link in ("ab", "bg"):
        oxygen = water_nodes(link)[0]
        nodes = nx.node_connected_component(graph, oxygen)
        elements = [graph.nodes[n]["element"] for n in nodes]
        p_count = elements.count("P")
        if "C" in elements and p_count == 1:
            role = "AMP"
        elif "C" not in elements and p_count == 1:
            role = "orthophosphate"
        elif "C" in elements and p_count == 2:
            role = "ADP"
        elif "C" not in elements and p_count == 2:
            role = "pyrophosphate"
        elif p_count == 0:
            role = "water"
        else:
            role = "other"
        result[oxygen] = {
            "role": role,
            "isotope": graph.nodes[oxygen]["isotope"],
            "phosphorus_anchors": sorted(
                n for n in nodes if graph.nodes[n]["element"] == "P"
            ),
            "component_18O_count": sum(
                graph.nodes[n]["element"] == "O" and graph.nodes[n]["isotope"] == 18
                for n in nodes
            ),
        }
    return result


def contains_adp(graph):
    return any(
        sum(graph.nodes[n]["element"] == "P" for n in nodes) == 2
        and any(graph.nodes[n]["element"] == "C" for n in nodes)
        for nodes in nx.connected_components(graph)
    )


def phosphate_inventory(graph):
    components = [
        nodes
        for nodes in nx.connected_components(graph)
        if sum(graph.nodes[n]["element"] == "P" for n in nodes) == 1
        and not any(graph.nodes[n]["element"] == "C" for n in nodes)
    ]
    labelled = sum(
        any(
            graph.nodes[n]["element"] == "O" and graph.nodes[n]["isotope"] == 18
            for n in nodes
        )
        for nodes in components
    )
    return {"total": len(components), "labelled": labelled}


def mixture_bounds(cases, observed_percent):
    rows = []
    for ab_target, bg_target in product(("PA", "PB"), ("PB", "PG")):
        selected = [
            c for c in cases if c["targets"] == {"ab": ab_target, "bg": bg_target}
        ]
        assert len(selected) == 4
        inventories = [
            cut["phosphate_inventory"] for c in selected for cut in c["cuts"]
        ]
        ratio = max(
            (v["labelled"] / v["total"] for v in inventories if v["total"]),
            default=0,
        )
        denominator = 2 if ratio == 0.5 else 1
        numerator = 1 if ratio else 0
        assert ratio in (0, 0.5, 1)
        assert all(
            denominator * v["labelled"] <= numerator * v["total"] for v in inventories
        )
        position_ok = next(
            c["qualitative_observation_compatible"]
            for c in selected
            if c["isotopes"] == [18, 18]
        )
        estimate_ok = observed_percent * denominator <= 100 * numerator
        rows.append(
            {
                "targets": {"ab": ab_target, "bg": bg_target},
                "cuts_checked": len(inventories),
                "bound_numerator": numerator,
                "bound_denominator": denominator,
                "maximum_fraction": ratio,
                "reported_point_estimate_exceeds_bound": not estimate_ok,
                "amp_position_and_bound_consistent": position_ok and estimate_ok,
            }
        )
    return rows


def native_apply_with_frame(mod, hosts, rule):
    dg = mod.DG(graphDatabase=hosts)
    with dg.build() as builder:
        derivations = builder.apply(hosts, rule, onlyProper=False, verbosity=0)
    assert len(derivations) == 1, len(derivations)
    edge = derivations[0]
    used = {vertex.graph.id for vertex in edge.sources}
    # Native derivations list participating components. The complete supplied
    # world also retains every untouched component, including the other water.
    outputs = [vertex.graph for vertex in edge.targets]
    outputs.extend(graph for graph in hosts if graph.id not in used)
    return dg, outputs


def native_word(mod, initial, ordered, expected):
    hosts = mod.Graph.fromGMLStringMulti(graph_gml(initial), add=False)
    assert _mod_signature(hosts) == expected_signature(initial)
    native_rules, maps = [], []
    for event in ordered:
        gml, left, right = rule_gml(event.rule)
        native_rules.append(mod.ruleGMLString(gml, add=False))
        maps.append((left, right))
    current, owners = hosts, []
    signatures, decoded_roles, inventories = [], [], []
    for rule, world in zip(native_rules, expected[1:]):
        dg, current = native_apply_with_frame(mod, current, rule)
        owners.append(dg)
        actual = _mod_signature(current)
        assert actual == expected_signature(world)
        decoded = decode_graphs(current)
        assert roles(decoded) == roles(world.to_networkx())
        inventory = phosphate_inventory(decoded)
        assert inventory == phosphate_inventory(world.to_networkx())
        inventories.append(inventory)
        signatures.append(sha256(repr(actual).encode()).hexdigest())
        decoded_roles.append(roles(decoded))
    match = mod.RCMatch(*native_rules)
    first_right, second_left = maps[0][1], maps[1][0]
    overlap = set(first_right) & set(second_left)
    for key in sorted(overlap, key=repr):
        match.push(
            native_rules[0].getVertexFromExternalId(first_right[key]).right,
            native_rules[1].getVertexFromExternalId(second_left[key]).left,
        )
    composed = match.compose()
    assert composed is not None
    dg, outputs = native_apply_with_frame(mod, hosts, composed)
    assert _mod_signature(outputs) == expected_signature(expected[-1])
    composite_roles = roles(decode_graphs(outputs))
    assert composite_roles == decoded_roles[-1]
    composite_inventory = phosphate_inventory(decode_graphs(outputs))
    assert composite_inventory == inventories[-1]
    return {
        "primitive_world_signatures": signatures,
        "primitive_roles": decoded_roles,
        "supplied_overlap_carriers": len(overlap),
        "composite_roles": composite_roles,
        "primitive_phosphate_inventories": inventories,
        "composite_phosphate_inventory": composite_inventory,
        "primitive_and_composite_agree": True,
    }


def run():
    import mod

    protocol = json.loads(PROTOCOL.read_text())
    amendment_path = ROOT / "Experiment/MTG/solvent_mixture_analysis_amendment.json"
    amendment = json.loads(amendment_path.read_text())
    assert amendment["original_protocol_sha256"] == digest(PROTOCOL)
    path = ROOT / protocol["input"]
    assert digest(path) == protocol["input_sha256"]
    data = json.loads(path.read_text())
    links = protocol["family"]["links"]
    cases, final_graphs, scored = [], [], []
    count = {
        "histories": 0,
        "ideal_comparisons": 0,
        "direct_composites": 0,
        "source_orders": 0,
        "source_applications": 0,
        "native_primitive_applications": 0,
        "native_compositions": 0,
    }
    for isotopes in protocol["family"]["isotope_controls"]:
        initial = initial_graph(data, isotopes)
        for ab_target, bg_target in product(("PA", "PB"), ("PB", "PG")):
            targets = {"ab": ab_target, "bg": bg_target}
            events = tuple(macro(initial, e, targets[e], links) for e in ("ab", "bg"))
            history = construct(initial, events)
            assert history.replay() and not history.causal_pairs
            cuts = []
            for ideal in ((), ("ab",), ("bg",), ("ab", "bg")):
                oracle = initial.to_networkx()
                for event in ideal:
                    oracle = edit(oracle, event, targets[event], links)
                expected = llg(oracle)
                cut = history.configuration(ideal)
                assert same(cut.world, expected)
                composite = history.compile_ideal(ideal).rule
                app = apply_dpo(
                    composite, initial, {n: n for n in composite.left.node_ids}
                )
                assert app.certificate.replay().valid and same(app.result, expected)
                cuts.append(
                    {
                        "ideal": list(ideal),
                        "roles": roles(oracle),
                        "contains_adp": contains_adp(oracle),
                        "phosphate_inventory": phosphate_inventory(oracle),
                        "direct_composite_agrees": True,
                    }
                )
                count["ideal_comparisons"] += 1
                count["direct_composites"] += 1
            schedules = []
            for order in ORDERS:
                current, oracle, worlds = initial, initial.to_networkx(), [initial]
                for name in order:
                    event = next(e for e in events if e.event_id == name)
                    app = apply_dpo(event.rule, current, dict(event.match))
                    assert app.certificate.replay().valid
                    oracle = edit(oracle, name, targets[name], links)
                    assert same(app.result, llg(oracle))
                    current = app.result
                    worlds.append(current)
                    count["source_applications"] += 1
                native = native_word(
                    mod,
                    initial,
                    tuple(next(e for e in events if e.event_id == n) for n in order),
                    worlds,
                )
                schedules.append({"order": list(order), "native": native})
                count["source_orders"] += 1
                count["native_primitive_applications"] += 2
                count["native_compositions"] += 1
            final = history.configuration(("ab", "bg")).world.to_networkx()
            final_roles = roles(final)
            row = {
                "isotopes": isotopes,
                "targets": targets,
                "cuts": cuts,
                "schedules": schedules,
                "final_roles": final_roles,
                "model_parallelism_only": True,
            }
            if isotopes == [18, 18]:
                compatible = not any(v["role"] == "AMP" for v in final_roles.values())
                row["qualitative_observation_compatible"] = compatible
                scored.append(row)
            else:
                row["qualitative_observation_compatible"] = None
            cases.append(row)
            final_graphs.append(final)
            count["histories"] += 1
    # Boundary control removes isotope/anchor identity only for the diagnostic.
    node_equal = nx.algorithms.isomorphism.categorical_node_match(
        ("element", "lone_pairs", "valence_electrons", "radical", "hcount"),
        (None,) * 5,
    )
    edge_equal = nx.algorithms.isomorphism.categorical_edge_match(
        ("sigma_order", "pi_order"), (None, None)
    )
    assert all(
        nx.is_isomorphic(
            final_graphs[0], g, node_match=node_equal, edge_match=edge_equal
        )
        for g in final_graphs[1:]
    )
    refusals = {}
    initial = initial_graph(data, (18, 18))
    event = macro(initial, "ab", "PB", links)
    bad_match = dict(event.match)
    bad_match[water_nodes("ab")[0]] = water_nodes("bg")[0]
    try:
        apply_dpo(event.rule, initial, bad_match)
    except ValueError:
        refusals["wrong_solvent_match"] = True
    else:
        raise AssertionError("Incorrect supplied water binding admitted")
    assert count == {
        "histories": 16,
        "ideal_comparisons": 64,
        "direct_composites": 64,
        "source_orders": 32,
        "source_applications": 64,
        "native_primitive_applications": 64,
        "native_compositions": 32,
    }
    assert sum(r["qualitative_observation_compatible"] for r in scored) == 2
    bounds = mixture_bounds(cases, amendment["observation_percent"])
    assert sum(r["amp_position_and_bound_consistent"] for r in bounds) == 1
    inequalities = 0
    for case in cases:
        for cut in case["cuts"]:
            value = cut["phosphate_inventory"]
            gamma_pool = value["total"] if case["targets"]["bg"] == "PG" else 0
            assert 2 * value["labelled"] <= value["total"] + gamma_pool
            inequalities += 1
    assert inequalities == 64
    # An integer-copy mixture attains the 63% point estimate and the 26% lower
    # bound in the allowed inventory cone: 37 beta-complete, 26 gamma-single cuts.
    beta = next(
        c
        for c in cases
        if c["isotopes"] == [18, 18] and c["targets"] == {"ab": "PB", "bg": "PB"}
    )
    gamma = next(
        c
        for c in cases
        if c["isotopes"] == [18, 18] and c["targets"] == {"ab": "PB", "bg": "PG"}
    )
    b = next(
        c["phosphate_inventory"] for c in beta["cuts"] if c["ideal"] == ["ab", "bg"]
    )
    g = next(c["phosphate_inventory"] for c in gamma["cuts"] if c["ideal"] == ["bg"])
    witness = {
        "beta_complete_copies": 37,
        "gamma_single_event_copies": 26,
        "total": 37 * b["total"] + 26 * g["total"],
        "labelled": 37 * b["labelled"] + 26 * g["labelled"],
        "gamma_pool": 26 * g["total"],
    }
    assert (
        witness["total"] == 100
        and witness["labelled"] == 63
        and witness["gamma_pool"] == 26
    )
    pinned = [
        PROTOCOL,
        amendment_path,
        path,
        Path(__file__),
        ROOT / "Experiment/MTG/conditioned_coordinate_history.py",
        ROOT / "Experiment/MTG/compositional_condition_compiler.py",
        ROOT / "Experiment/MTG/frame_condition_compiler.py",
        ROOT / "Experiment/MTG/mod_encoding.py",
        ROOT / "Experiment/MTG/mod_history_comparison.py",
        ROOT / "synkit/Graph/MTG/local.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
    ]
    native_paths = [Path(mod.libpymod.__file__).resolve()]
    linked = native_paths[0].parents[1] / "libmod.so"
    if linked.is_file():
        native_paths.append(linked.resolve())
    return {
        "schema": "mtg.selenophosphate-isotope-evidence/1",
        "status": "PASS",
        "protocol": protocol,
        "counts": count,
        "cases": cases,
        "refusals": refusals,
        "scored_binding_alternatives": 4,
        "compatible_bindings": 2,
        "contradicted_bindings": 2,
        "compatible_ordered_candidates": 4,
        "contradicted_ordered_candidates": 4,
        "secondary_mixture_analysis": {
            "data_informed_amendment": amendment,
            "binding_bounds": bounds,
            "inventory_inequalities_checked": inequalities,
            "reported_point_estimate_percent": amendment["observation_percent"],
            "minimum_gamma_attack_phosphate_pool_percent": 2
            * amendment["observation_percent"]
            - 100,
            "bound_attaining_copy_witness": witness,
            "all_binding_mixtures_may_retain_beta_attack": True,
            "turnover_fraction_or_probability_inferred": False,
            "full_spectral_agreement_established": False,
        },
        "compatibility_scope": (
            "Only the predeclared AMP oxygen-position condition; "
            "full spectral/kinetic agreement is not established."
        ),
        "phosphate_isotopologue_counts": (
            "Component 18O counts are retained. The two AMP-position-compatible "
            "bindings give (2,0) or (1,1) for beta/gamma Pi under [18,18]. "
            "Detection limits and a mixed-solvent isotope model were not fitted."
        ),
        "all_isotope_masked_boundaries_isomorphic": True,
        "native_outputs_answer_the_same_queries": True,
        "original_isotope_electron_schema_closed_certificates": True,
        "physical_parallelism_established": False,
        "kinetic_mixture_fitted": False,
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
    result = run()
    output = ROOT / "Experiment/MTG/evidence/selenophosphate_isotope_case.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "compatible_bindings": result["compatible_bindings"],
                "contradicted_bindings": result["contradicted_bindings"],
            }
        )
    )
