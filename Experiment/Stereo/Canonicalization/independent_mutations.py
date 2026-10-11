"""Seven executable defects which independent identity/action gates must reject."""

from dataclasses import replace
import time

import networkx as nx

from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
from Experiment.Stereo.Canonicalization.independent_model import (
    formal_states, mirror, reflect_frame, to_descriptor, transport,
)
from Experiment.Stereo.Canonicalization.independent_orbits import (
    _automorphisms, _states_and_lookup,
    composition_check, partition_agreement, reference_partition,
)


def _certificate(graph, states, *, pin_carriers=False):
    """Production-side control; it supplies no reference truth."""
    from synkit.Graph.Stereo import canonicalize_configured_stereograph, expand_configured_stereograph
    from synkit.Graph.Stereo.canonical import StereoLocusVertex
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

    descriptors = tuple(to_descriptor(value) for value in states)
    if not pin_carriers:
        return canonicalize_configured_stereograph(
            graph, descriptors, atom_color="color", bond_color="color",
            enumerate_automorphism_group=False,
        ).canonical_code
    expansion = expand_configured_stereograph(graph, descriptors, atom_color="color", bond_color="color")
    for node in expansion:
        if isinstance(node, StereoLocusVertex):
            expansion.nodes[node]["color"] = (*expansion.nodes[node]["color"], "pinned", node.index)
    return ExactColoredGraphCanonicalizer(
        expansion, node_color="color", edge_color=None, enumerate_automorphism_group=False,
    ).canonicalize().canonical_code


def _basis(fixture, method="exhaustive_constitutional_automorphisms"):
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    reference = reference_partition(fixture.graph, choices, atom_color="color", bond_color="color", method=method)
    if reference["status"] != "complete":
        raise RuntimeError(f"Mutation basis is incomplete: {reference}")
    assignments = reference["assignments"]
    states = tuple(tuple(local[index] for local, index in zip(choices, assignment, strict=True))
                   for assignment in assignments)
    labels = tuple(_certificate(fixture.graph, values) for values in states)
    if not partition_agreement(reference, assignments, labels):
        raise RuntimeError("Unmutated production control does not agree with the reference basis.")
    return choices, reference, assignments, states, labels


def run_mutations(single_carrier_method="exhaustive_constitutional_automorphisms"):
    """Inject each defect into real certificates, graph gadgets or slot actions."""
    catalog = {fixture.name: fixture for fixture in fixtures()}
    atom = catalog["tetrahedral:distinct"]
    choices, reference, assignments, states, labels = _basis(atom, single_carrier_method)
    constant = ("constant-certificate",) * len(assignments)
    erased = tuple(_certificate(atom.graph, (choices[0][0],)) for _ in assignments)
    omitted = tuple(_certificate(atom.graph, ()) for _ in assignments)
    records = [
        {"mutation": "constant_certificate", "rejected": not partition_agreement(reference, assignments, constant),
         "mechanism": "Replace every production full certificate by one constant."},
        {"mutation": "erase_one_parity", "rejected": not partition_agreement(reference, assignments, erased),
         "mechanism": "Normalize away one tetrahedral configuration by replacing it with a fixed orientation."},
        {"mutation": "omit_one_locus", "rejected": not partition_agreement(reference, assignments, omitted),
         "mechanism": "Canonicalize the constitution after dropping its configured locus."},
    ]
    mapping = {node: 1000 + node for node in atom.graph}
    renamed = nx.relabel_nodes(atom.graph, mapping)
    original_literal = tuple(label + repr(tuple(atom.graph)) for label in labels)
    relabeled = tuple(_certificate(renamed, tuple(transport(value, mapping) for value in row)) for row in states)
    relabeled_literal = tuple(label + repr(tuple(renamed)) for label in relabeled)
    correct_renumbering = relabeled == labels
    records.append({"mutation": "literal_atom_indices", "rejected": correct_renumbering
                    and original_literal != relabeled_literal,
                    "mechanism": "Append literal atom indices to otherwise correct full certificates."})

    planar = catalog["planar_bond:distinct"]
    _, planar_reference, _, planar_states, planar_labels = _basis(planar, single_carrier_method)
    wrong_states = tuple((replace(row[0], atoms=reflect_frame("planar_bond", row[0].atoms)),)
                         for row in planar_states)
    wrong_mirrors = tuple(_certificate(planar.graph, row) for row in wrong_states)
    correct_mirrors = tuple(_certificate(planar.graph, tuple(mirror(value) for value in row)) for row in planar_states)
    records.append({"mutation": "wrong_EZ_mirror", "rejected": correct_mirrors == planar_labels
                    and wrong_mirrors != planar_labels,
                    "reference_mirror_fixed_classes": len(planar_reference["mirror_classes"]["mirror_fixed_classes"]),
                    "mechanism": "Use E/Z reconfiguration as the molecular mirror operation."})

    meso = catalog["tetrahedral:two:meso"]
    _, meso_reference, meso_assignments, meso_states, _ = _basis(meso)
    pinned = tuple(_certificate(meso.graph, row, pin_carriers=True) for row in meso_states)
    records.append({"mutation": "prohibit_carrier_exchange",
                    "rejected": not partition_agreement(meso_reference, meso_assignments, pinned),
                    "mechanism": "Color auxiliary stereo loci by input indices to prevent valid carrier exchange."})

    geometry = catalog["trigonal_bipyramidal:symmetric"]
    choices = tuple(formal_states(seed) for seed in geometry.seeds)
    prepared, lookup, inventory = _states_and_lookup(choices)
    maps, tables, _ = _automorphisms(
        geometry.graph, prepared, lookup, inventory, "color", "color", 50000, time.monotonic() + 10, 1024,
    )
    correct = composition_check(tuple(geometry.graph), maps, tables)
    reversed_action = composition_check(tuple(geometry.graph), maps, tables, reverse=True)
    records.append({"mutation": "reverse_permutation_composition",
                    "rejected": correct["failures"] == 0 and reversed_action["failures"] > 0,
                    "correct_action_checks": correct, "mutated_action_checks": reversed_action,
                    "mechanism": "Reverse noncommuting graph-map composition while retaining sequential transport."})
    return {"records": records, "single_carrier_reference_method": single_carrier_method,
            "passed": len(records) == 7 and all(record["rejected"] for record in records)}
