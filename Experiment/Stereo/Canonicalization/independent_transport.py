"""Exact one-carrier orbits by rooted constitutional transport queries.

The local frame actions come from the independent geometry model. Rooted
queries concern the original attributed constitution; no stereo gadget,
production certificate or production equality predicate supplies truth.
"""

from collections import Counter
from dataclasses import replace
from itertools import product
import time

import networkx as nx

from Experiment.Stereo.Canonicalization.independent_graph import COLOR_KEY, _class_ids
from Experiment.Stereo.Canonicalization.independent_model import (
    ATOM_FAMILIES, BOND_FAMILIES, _effective_atoms, _reframe, _rotations,
    inventory_key, proper_group, state_key, transport,
)
from Experiment.Stereo.Canonicalization.independent_orbits import (
    ATOM_KEYS, BOND_KEYS, OracleIncomplete, _action_table, _limits,
    _mirror_classes, _mirror_commutation, _mirror_table, _selector,
    _states_and_lookup, apply_table, color_signature,
)

METHOD = "direct_single_carrier_transport"


def _normal(state):
    if state.family in ATOM_FAMILIES | BOND_FAMILIES:
        parity = 0 if state.family in {"square_planar", "planar_bond", "extended_cis_trans"} else 1
        return replace(state, atoms=_effective_atoms(state), parity=parity)
    if state.family == "planar_chirality" and state.parity == -1:
        return replace(state, path=state.path[::-1], parity=1)
    return state


def _poses(state):
    """Exhaust exactly the representations equivalent to a normalized target."""
    if state.family in ATOM_FAMILIES | BOND_FAMILIES:
        return tuple(_reframe(state, tuple(state.atoms[index] for index in image))
                     for image in proper_group(state.family))
    if state.family == "planar_chirality":
        return tuple(replace(state, path=path) for path in _rotations(state.path))
    if state.family == "helical":
        paths = (state.path, state.path[::-1])
        if state.cyclic:
            paths = (*_rotations(state.path), *_rotations(state.path[::-1]))
        return tuple(replace(state, path=path) for path in paths)
    raise OracleIncomplete(f"direct_transport_family_not_supported:{state.family}")


def _reference(reference):
    if type(reference) is int:
        return "material", reference
    if isinstance(reference, str) and reference.startswith(("@H:", "@LP:")):
        kind, owner = reference.split(":", 1)
        return kind, int(owner)
    raise OracleIncomplete(f"undeclared_reference_type:{reference!r}")


def _bindings(source, target):
    """Bind material vertices and virtual-port owners, preserving port kinds."""
    pairs = [*zip(source.atoms, target.atoms, strict=True), *zip(source.path, target.path, strict=True)]
    if source.pilot is not None:
        pairs.append((source.pilot, target.pilot))
    forward, reverse = {}, {}
    for left, right in pairs:
        left_kind, left_owner = _reference(left)
        right_kind, right_owner = _reference(right)
        if left_kind != right_kind:
            return None
        if left_owner in forward and forward[left_owner] != right_owner:
            return None
        if right_owner in reverse and reverse[right_owner] != left_owner:
            return None
        forward[left_owner], reverse[right_owner] = right_owner, left_owner
    return forward


def _joint_colors(graph, node_tokens, edge_tokens, bindings, check):
    """Refine both rooted copies with one palette, so IDs are comparable."""
    anchors = ({node: index for index, node in enumerate(bindings)},
               {node: index for index, node in enumerate(bindings.values())})
    colors = _class_ids({(side, node): (node_tokens[node], anchors[side].get(node))
                         for side in range(2) for node in graph})
    while True:
        check()
        signatures = {}
        for side, node in colors:
            neighbors = Counter((edge_tokens[frozenset((node, other))], colors[side, other])
                                for other in graph[node])
            signatures[side, node] = colors[side, node], frozenset(neighbors.items())
        refined = _class_ids(signatures)
        if len(set(refined.values())) == len(set(colors.values())):
            return tuple({node: refined[side, node] for node in graph} for side in range(2))
        colors = refined


class _Queries:
    def __init__(self, graph, atom_color, bond_color, deadline, max_rss_mb, limit):
        if graph.is_directed() or graph.is_multigraph():
            raise OracleIncomplete("reference_domain_requires_simple_undirected_graph")
        self.graph, self.deadline, self.max_rss_mb, self.limit = graph, deadline, max_rss_mb, limit
        node_color, edge_color = _selector(atom_color), _selector(bond_color)
        self.nodes = {node: color_signature(node_color(data)) for node, data in graph.nodes(data=True)}
        self.edges = {frozenset((left, right)): color_signature(edge_color(data))
                      for left, right, data in graph.edges(data=True)}
        self.counts = {"frame_bindings": 0, "isomorphism_queries": 0, "isomorphism_map_proposals": 0}
        self.cache = {}

    def check(self):
        _limits(self.deadline, self.max_rss_mb)

    def charge(self, field):
        self.check()
        self.counts[field] += 1
        if self.counts[field] > self.limit:
            raise OracleIncomplete(f"direct_transport_{field}_limit:{self.limit}")

    def query(self, bindings):
        key = frozenset(bindings.items())
        if key in self.cache:
            return self.cache[key]
        self.charge("isomorphism_queries")
        if not set(bindings) <= set(self.graph) or not set(bindings.values()) <= set(self.graph):
            raise OracleIncomplete("carrier_reference_owner_absent_from_constitution")
        if any(self.nodes[left] != self.nodes[right] for left, right in bindings.items()):
            self.cache[key] = None
            return None
        colors = _joint_colors(self.graph, self.nodes, self.edges, bindings, self.check)
        left, right = self.graph.copy(), self.graph.copy()
        nx.set_node_attributes(left, colors[0], COLOR_KEY)
        nx.set_node_attributes(right, colors[1], COLOR_KEY)
        result = None
        for mapping in nx.vf2pp_all_isomorphisms(left, right, node_label=COLOR_KEY):
            self.charge("isomorphism_map_proposals")
            if (all(mapping[node] == image for node, image in bindings.items())
                    and all(self.nodes[node] == self.nodes[image] for node, image in mapping.items())
                    and all(self.edges[frozenset((a, b))] == self.edges[frozenset((mapping[a], mapping[b]))]
                            for a, b in self.graph.edges)):
                result = mapping
                break
        self.cache[key] = result
        return result

    def relation(self, source, target):
        source, target = _normal(source), _normal(target)
        if source.family == "helical" and source.parity != target.parity:
            return None
        for pose in _poses(target):
            self.charge("frame_bindings")
            bindings = _bindings(source, pose)
            if bindings is None:
                continue
            mapping = self.query(bindings)
            if mapping is None:
                continue
            image = transport(source, mapping)
            if state_key(image) != state_key(target) or inventory_key(image) != inventory_key(target):
                raise OracleIncomplete("direct_transport_witness_replay_failure")
            return mapping
        return None


def _composition(nodes, maps, tables, states, lookup, check):
    chosen = tuple(range(min(len(maps), 32)))
    checks, failures = 0, 0
    for left_index, right_index in product(chosen, repeat=2):
        check()
        composed = {node: maps[right_index][maps[left_index][node]] for node in nodes}
        direct = _action_table(states, lookup, composed)
        for carrier, local in enumerate(tables[left_index]):
            for index, (middle, middle_index) in enumerate(local):
                checks += 1
                failures += tables[right_index][middle][middle_index] != direct[carrier][index]
    return {"map_pairs": len(chosen) ** 2, "state_checks": checks, "failures": failures,
            "map_pair_selection": "first 32 distinct positive witness maps; not an exhaustive symmetry group"}


def reference_partition(graph, choices, *, atom_color=ATOM_KEYS, bond_color=BOND_KEYS,
                        max_assignments=1024, timeout_seconds=10.0, max_rss_mb=1024,
                        max_queries=50000):
    """Decide the entire single-carrier orbit relation or return an explicit refusal."""
    started = time.monotonic()
    budget = {"max_assignments": max_assignments, "reference_timeout_seconds": timeout_seconds,
              "max_process_peak_rss_mb": max_rss_mb, "max_frame_bindings": max_queries,
              "max_isomorphism_queries": max_queries, "max_isomorphism_map_proposals": max_queries}
    base = {"schema": "synkit.independent-assignment-orbits/1", "method": METHOD, "budget": budget}
    queries = None
    try:
        if len(choices) != 1 or not choices[0] or len(choices[0]) > max_assignments:
            raise OracleIncomplete("direct_transport_requires_one_bounded_carrier")
        states, lookup, _inventory = _states_and_lookup(choices)
        queries = _Queries(graph, atom_color, bond_color, started + timeout_seconds, max_rss_mb, max_queries)
        queries.check()
        assignments = tuple((index,) for index in range(len(states[0])))
        remaining = set(range(len(states[0])))
        partition, witnesses, maps = [], [], {}
        while remaining:
            representative = min(remaining)
            members = []
            for target in sorted(remaining):
                mapping = queries.relation(states[0][representative], states[0][target])
                if mapping is None:
                    continue
                members.append((target,))
                maps.setdefault(tuple(mapping[node] for node in graph), mapping)
                witnesses.append({"source_assignment": (representative,), "target_assignment": (target,),
                                  "atom_mapping": tuple(sorted(mapping.items())), "carrier_permutation": (0,)})
            if (representative,) not in members:
                raise OracleIncomplete("missing_identity_transport")
            partition.append(tuple(members))
            remaining.difference_update(member[0] for member in members)
        partition = tuple(sorted(partition))
        maps = tuple(maps.values())
        tables = tuple(_action_table(states, lookup, mapping) for mapping in maps)
        mirror_table = _mirror_table(states, lookup)
        composition = _composition(tuple(graph), maps, tables, states, lookup, queries.check)
        commutation = _mirror_commutation(tables, mirror_table)
        if composition["failures"] or commutation["failures"]:
            raise OracleIncomplete("reference_action_or_mirror_law_failure")
        mirrors = _mirror_classes(partition, mirror_table)
        queries.check()
        return {**base, "status": "complete", "assignments": assignments, "partition": partition,
                "merge_witnesses": witnesses, "constitutional_automorphisms": None,
                "inventory_preserving_automorphisms": None, "automorphism_group_enumerated": False,
                "formal_local_class_counts": [len(states[0])], "query_counts": queries.counts,
                "retained_positive_witness_maps": len(maps), "composition": composition,
                "mirror_commutation": commutation, "mirror_classes": mirrors,
                "mirror_assignment_images": [apply_table(value, mirror_table) for value in assignments],
                "seconds": time.monotonic() - started}
    except (OracleIncomplete, ValueError, KeyError) as error:
        return {**base, "status": "oracle_incomplete", "reason": str(error),
                "query_counts": {} if queries is None else queries.counts,
                "seconds": time.monotonic() - started}
