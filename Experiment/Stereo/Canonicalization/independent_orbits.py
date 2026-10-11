"""Bounded, independent assignment-orbit oracle and executable audit entry point.

Reference truth uses a constitutional GraphMatcher and geometry-derived local
actions. Production full certificates are read only by the comparison runner.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from itertools import product
from math import prod
from pathlib import Path
import sys
import time

from Experiment.Stereo.Canonicalization.independent_graph import typed_automorphisms
from Experiment.Stereo.Canonicalization.runtime import peak_rss_mib, wall_time_limit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Canonicalization.independent_model import (  # noqa: E402
    State, formal_states, from_descriptor, inventory_key, mirror, state_key, transport,
)

# Explicit declared attributed-graph contract, not a production color helper.
ATOM_KEYS = ("element", "isotope", "charge", "formal_charge", "radical", "radical_electrons",
             "aromatic", "hcount", "lone_pairs")
BOND_KEYS = ("order", "standard_order", "bond_type", "aromatic")


class OracleIncomplete(RuntimeError):
    """A resource or input boundary prevented an exhaustive reference result."""


def partition_from_labels(assignments, labels):
    """Compare exact label equality; hashes are never equality evidence."""
    if len(assignments) != len(labels) or any(label is None for label in labels):
        raise ValueError("Every assignment needs one completed exact label.")
    grouped = defaultdict(list)
    for assignment, label in zip(assignments, labels, strict=True):
        grouped[label].append(tuple(assignment))
    return tuple(sorted(tuple(sorted(values)) for values in grouped.values()))


def _selector(keys):
    if keys is None:
        return lambda _data: None
    if callable(keys):
        return keys
    if isinstance(keys, str):
        return lambda data: data.get(keys)
    return lambda data: tuple(data.get(key) for key in keys)


def color_signature(value):
    """Preserve declared attribute types without using the production encoder."""
    if value is None:
        return "null",
    if isinstance(value, Enum):
        return "enumeration", type(value).__module__, type(value).__qualname__, color_signature(value.value)
    if type(value) in (bool, int, str, bytes):
        return type(value).__name__, value
    if type(value) is float:
        return "real", value.hex()
    if isinstance(value, Mapping):
        return "dictionary", frozenset((color_signature(key), color_signature(item)) for key, item in value.items())
    if type(value) in (list, tuple):
        return type(value).__name__, tuple(color_signature(item) for item in value)
    if type(value) in (set, frozenset):
        return "set_members", frozenset(color_signature(item) for item in value)
    raise OracleIncomplete(f"undeclared_attribute_color_type:{type(value).__name__}")


def _limits(deadline, max_rss_mb):
    if time.monotonic() > deadline:
        raise OracleIncomplete("oracle_timeout")
    if peak_rss_mib() > max_rss_mb:
        raise OracleIncomplete("oracle_process_peak_rss_limit")


def _states_and_lookup(choices):
    states = tuple(tuple(value if isinstance(value, State) else from_descriptor(value) for value in carrier)
                   for carrier in choices)
    lookup = {}
    inventory = []
    for carrier, local in enumerate(states):
        if not local or any(value.parity is None for value in local):
            raise OracleIncomplete("empty_or_unspecified_configuration_choices")
        expected = {repr(state_key(value)) for value in formal_states(local[0])}
        actual = [repr(state_key(value)) for value in local]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise OracleIncomplete(f"formal_choices_not_independently_exhaustive:{carrier}")
        if len({repr(inventory_key(value)) for value in local}) != 1:
            raise OracleIncomplete(f"incompatible_local_inventory:{carrier}")
        inventory.append(repr(inventory_key(local[0])))
        for index, value in enumerate(local):
            key = repr(state_key(value))
            if key in lookup:
                raise OracleIncomplete("duplicate_or_ambiguous_carrier_key")
            lookup[key] = carrier, index
    if len(set(inventory)) != len(inventory):
        raise OracleIncomplete("duplicate_carrier_inventory_key")
    return states, lookup, tuple(sorted(inventory))


def _action_table(states, lookup, mapping):
    table = []
    for local in states:
        images = []
        for value in local:
            key = repr(state_key(transport(value, mapping)))
            if key not in lookup:
                raise OracleIncomplete("transport_not_closed_on_declared_formal_choices")
            images.append(lookup[key])
        if len({carrier for carrier, _ in images}) != 1:
            raise OracleIncomplete("configuration_dependent_carrier_transport")
        table.append(tuple(images))
    return tuple(table)


def apply_table(assignment, table):
    result = [None] * len(assignment)
    for source, index in enumerate(assignment):
        target, image = table[source][index]
        if result[target] is not None:
            raise OracleIncomplete("carrier_transport_not_bijective")
        result[target] = image
    if any(value is None for value in result):
        raise OracleIncomplete("carrier_transport_not_surjective")
    return tuple(result)


def _automorphisms(graph, states, lookup, inventory, atom_color, bond_color, max_automorphisms,
                   deadline, max_rss_mb):
    node_color, edge_color = _selector(atom_color), _selector(bond_color)
    if graph.is_directed() or graph.is_multigraph():
        raise OracleIncomplete("reference_domain_requires_simple_undirected_graph")
    node_tokens = {node: color_signature(node_color(data)) for node, data in graph.nodes(data=True)}
    edge_tokens = {frozenset((left, right)): color_signature(edge_color(data))
                   for left, right, data in graph.edges(data=True)}
    candidates = typed_automorphisms(graph, node_tokens, edge_tokens, lambda: _limits(deadline, max_rss_mb))
    maps, tables = [], []
    enumerated = 0
    for mapping in candidates:
        enumerated += 1
        _limits(deadline, max_rss_mb)
        if enumerated > max_automorphisms:
            raise OracleIncomplete(f"constitutional_automorphism_limit:{max_automorphisms}")
        image_inventory = tuple(sorted(repr(inventory_key(transport(local[0], mapping))) for local in states))
        if image_inventory != inventory:
            continue
        maps.append(mapping)
        tables.append(_action_table(states, lookup, mapping))
    if not maps:
        raise OracleIncomplete("missing_identity_automorphism")
    order = sorted(range(len(maps)), key=lambda index: tuple(repr(maps[index][node]) for node in graph))
    return tuple(maps[index] for index in order), tuple(tables[index] for index in order), enumerated


def composition_check(nodes, maps, tables, *, reverse=False, sample_limit=32):
    """Check noncommuting graph-map composition on every local formal state.

    All states are checked, with a declared deterministic subset of map pairs
    for large groups. The reversed argument order is a deliberate mutation.
    """
    indices = {tuple(mapping[node] for node in nodes): index for index, mapping in enumerate(maps)}
    chosen = tuple(range(min(len(maps), sample_limit)))
    checks, failures = 0, 0
    for left_index, right_index in product(chosen, repeat=2):
        left, right = maps[left_index], maps[right_index]
        composed = tuple(left[right[node]] if reverse else right[left[node]] for node in nodes)
        direct = tables[indices[composed]]
        for source, local in enumerate(tables[left_index]):
            for index, (middle, middle_index) in enumerate(local):
                sequential = tables[right_index][middle][middle_index]
                checks += 1
                failures += sequential != direct[source][index]
    return {"map_pairs": len(chosen) ** 2, "state_checks": checks, "failures": failures,
            "map_pair_selection": f"first {sample_limit} automorphisms in deterministic enumeration order"}


def _mirror_table(states, lookup):
    return tuple(tuple(lookup[repr(state_key(mirror(value)))] for value in local) for local in states)


def _mirror_commutation(tables, mirror_table):
    checks, failures = 0, 0
    for table in tables:
        for carrier, local in enumerate(table):
            for index, (target, image) in enumerate(local):
                mirror_source, mirror_index = mirror_table[carrier][index]
                before = table[mirror_source][mirror_index]
                after = mirror_table[target][image]
                checks += 1
                failures += before != after
    return {"state_checks": checks, "failures": failures}


def _orbit_partition(assignments, maps, tables, deadline, max_rss_mb):
    remaining = set(assignments)
    partition, witnesses = [], []
    while remaining:
        _limits(deadline, max_rss_mb)
        representative = min(remaining)
        orbit = {}
        for mapping, table in zip(maps, tables, strict=True):
            image = apply_table(representative, table)
            orbit.setdefault(image, (mapping, table))
        members = tuple(sorted(orbit))
        if not set(members) <= remaining:
            raise OracleIncomplete("reference_action_orbits_overlap_inconsistently")
        for member in members:
            mapping, table = orbit[member]
            witnesses.append({
                "source_assignment": representative, "target_assignment": member,
                "atom_mapping": tuple(sorted(mapping.items())),
                "carrier_permutation": tuple(local[0][0] for local in table),
            })
        partition.append(members)
        remaining.difference_update(members)
    return tuple(sorted(partition)), witnesses


def _mirror_classes(partition, mirror_table):
    class_of = {member: index for index, members in enumerate(partition) for member in members}
    images = []
    for members in partition:
        targets = {class_of[apply_table(member, mirror_table)] for member in members}
        if len(targets) != 1:
            raise OracleIncomplete("mirror_not_well_defined_on_reference_orbits")
        images.append(targets.pop())
    if any(images[images[index]] != index for index in range(len(images))):
        raise OracleIncomplete("reference_mirror_not_involutive")
    return {
        "class_images": images,
        "mirror_fixed_classes": [index for index, image in enumerate(images) if index == image],
        "enantiomer_pairs": [(index, image) for index, image in enumerate(images) if index < image],
    }


def _reference_partition(graph, choices, *, atom_color=ATOM_KEYS, bond_color=BOND_KEYS,
                         max_assignments=1024, max_automorphisms=50000, timeout_seconds=10.0,
                         max_rss_mb=1024):
    """Enumerate exact reference partitions or explicitly refuse an unfinished oracle."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    count = prod(len(local) for local in choices)
    budget = {"max_assignments": max_assignments, "max_constitutional_automorphisms": max_automorphisms,
              "reference_timeout_seconds": timeout_seconds, "max_process_peak_rss_mb": max_rss_mb}
    try:
        if not choices or count > max_assignments:
            raise OracleIncomplete(f"assignment_limit_or_empty_inventory:{count}")
        _limits(deadline, max_rss_mb)
        states, lookup, inventory = _states_and_lookup(choices)
        maps, tables, enumerated = _automorphisms(
            graph, states, lookup, inventory, atom_color, bond_color, max_automorphisms, deadline, max_rss_mb,
        )
        _limits(deadline, max_rss_mb)
        assignments = tuple(product(*(range(len(local)) for local in states)))
        mirror_table = _mirror_table(states, lookup)
        composition = composition_check(tuple(graph), maps, tables)
        commutation = _mirror_commutation(tables, mirror_table)
        if composition["failures"] or commutation["failures"]:
            raise OracleIncomplete("reference_action_or_mirror_law_failure")
        partition, witnesses = _orbit_partition(assignments, maps, tables, deadline, max_rss_mb)
        mirrors = _mirror_classes(partition, mirror_table)
        _limits(deadline, max_rss_mb)
        return {
            "schema": "synkit.independent-assignment-orbits/1", "status": "complete",
            "assignments": assignments, "partition": partition, "merge_witnesses": witnesses,
            "constitutional_automorphisms": enumerated, "inventory_preserving_automorphisms": len(maps),
            "formal_local_class_counts": [len(local) for local in states],
            "mirror_assignment_images": [apply_table(assignment, mirror_table) for assignment in assignments],
            "mirror_classes": mirrors, "composition": composition, "mirror_commutation": commutation,
            "budget": budget, "seconds": time.monotonic() - started,
        }
    except (OracleIncomplete, ValueError, KeyError) as error:
        return {"schema": "synkit.independent-assignment-orbits/1", "status": "oracle_incomplete",
                "reason": str(error), "budget": budget, "seconds": time.monotonic() - started}


def reference_partition(graph, choices, **options):
    """Apply cooperative deadlines and platform-supported wall interruption."""
    method = options.pop("method", "exhaustive_constitutional_automorphisms")
    if method == "direct_single_carrier_transport":
        from Experiment.Stereo.Canonicalization.independent_transport import reference_partition as implementation
    elif method == "exhaustive_constitutional_automorphisms":
        implementation = _reference_partition
    else:
        raise ValueError(f"Unknown reference method: {method}")
    seconds = options.get("timeout_seconds", 10.0)
    if seconds <= 0:
        return {"schema": "synkit.independent-assignment-orbits/1", "status": "oracle_incomplete",
                "method": method, "reason": "oracle_timeout", "seconds": 0,
                "budget": {"reference_timeout_seconds": seconds}}

    started = time.monotonic()
    try:
        with wall_time_limit(seconds, OracleIncomplete("oracle_timeout")):
            result = implementation(graph, choices, **options)
        return {"method": method, **result}
    except OracleIncomplete as error:
        return {"schema": "synkit.independent-assignment-orbits/1", "status": "oracle_incomplete",
                "method": method, "reason": str(error), "seconds": time.monotonic() - started,
                "budget": {"reference_timeout_seconds": seconds}}


def partition_agreement(reference, assignments, labels):
    """An incomplete oracle never supports a passing correctness decision."""
    if reference["status"] != "complete":
        return False
    try:
        observed = partition_from_labels(assignments, labels)
    except ValueError:
        return False
    reference_assignments = tuple(tuple(assignment) for assignment in reference["assignments"])
    expected_partition = tuple(sorted(tuple(sorted(tuple(member) for member in orbit))
                                      for orbit in reference["partition"]))
    assignments_agree = tuple(tuple(assignment) for assignment in assignments) == reference_assignments
    return assignments_agree and observed == expected_partition


if __name__ == "__main__":
    from Experiment.Stereo.Canonicalization.independent_runner import main

    raise SystemExit(main())
