"""Hydrogen projection and reaction serialization helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx
from rdkit import Chem

from synkit.Graph import remove_wildcard_nodes
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Mech.electron_accounting import (
    graph_to_sanitized_kekule_mol,
    refresh_electron_fields,
)
from synkit.IO.chem_converter import graph_to_smi
from synkit.Synthesis.Reactor import product_state as _product_state


def _explicit_h(rc: nx.Graph) -> nx.Graph:
    if bool(rc.graph.get("electron_aware_rewrite", False)):
        return _explicit_h_tuple(rc)

    next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1
    orig_delta: Dict[int, int] = {}
    pair_to_nodes: Dict[int, List[int]] = defaultdict(list)

    for n, d in rc.nodes(data=True):
        h_pairs = d.get("h_pairs", [])
        hl, hr = d["typesGH"][0][2], d["typesGH"][1][2]
        orig_delta[n] = hl - hr
        for pid in h_pairs:
            if n not in pair_to_nodes[pid]:
                pair_to_nodes[pid].append(n)

    conn = nx.Graph()
    for nodes in pair_to_nodes.values():
        conn.add_nodes_from(nodes)
        # fmt: off
        conn.add_edges_from(
            (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]
        )
        # fmt: on

    migrations: List[Tuple[int, int]] = []
    for comp in nx.connected_components(conn):
        donors = [(n, orig_delta[n]) for n in comp if orig_delta[n] > 0]
        recips = [(n, -orig_delta[n]) for n in comp if orig_delta[n] < 0]
        for donor, count in donors:
            for _ in range(count):
                recv_idx = next(i for i, r in enumerate(recips) if r[1] > 0)
                recv, rcap = recips[recv_idx]
                recips[recv_idx] = (recv, rcap - 1)
                migrations.append((donor, recv))

    for src, dst in migrations:
        h = next_id
        next_id += 1
        rc.add_node(
            h,
            element="H",
            aromatic=False,
            charge=0,
            atom_map=0,
            hcount=0,
            typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
        )
        rc.add_edge(src, h, order=(1, 0), standard_order=1)
        rc.add_edge(h, dst, order=(0, 1), standard_order=-1)

    affected = [n for nodes in pair_to_nodes.values() for n in nodes]
    for n in affected:
        t0, t1 = rc.nodes[n]["typesGH"]
        delta_h = t0[2] - t1[2]
        if delta_h >= 0:
            t0_h, t1_h = t0[2] - 1, t1[2]
        else:
            t0_h, t1_h = t0[2], t1[2] - 1
        rc.nodes[n]["typesGH"] = (
            t0[:2] + (t0_h,) + t0[3:],
            t1[:2] + (t1_h,) + t1[3:],
        )
    return rc


def _explicit_h_tuple(rc: nx.Graph) -> nx.Graph:
    """Materialize only hydrogens that were explicit in the template."""
    next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1
    pair_left: Dict[int, int] = {}
    pair_right: Dict[int, int] = {}
    for n, data in rc.nodes(data=True):
        for pair_id in data.get("h_pairs_left", []):
            pair_left[pair_id] = n
        for pair_id in data.get("h_pairs_right", []):
            pair_right[pair_id] = n

    explicit_pairs = sorted(set(pair_left) & set(pair_right))
    if explicit_pairs:
        rc.graph["_product_electron_fields_current"] = False
    used_maps = {
        value
        for _, data in rc.nodes(data=True)
        for atom_map in [data.get("atom_map")]
        for value in (
            atom_map
            if isinstance(atom_map, tuple)
            else (() if atom_map in (None, 0) else (atom_map,))
        )
    }
    for pair_id in explicit_pairs:
        src = pair_left[pair_id]
        dst = pair_right[pair_id]
        h = next_id
        next_id += 1
        preferred_map = rc.nodes[src].get("h_pair_atom_maps", {}).get(
            pair_id
        ) or rc.nodes[dst].get("h_pair_atom_maps", {}).get(pair_id)
        atom_map = preferred_map if preferred_map not in used_maps else h
        while atom_map in used_maps:
            atom_map += 1
        used_maps.add(atom_map)
        rc.add_node(
            h,
            element=("H", "H"),
            aromatic=(False, False),
            charge=(0, 0),
            atom_map=(atom_map, atom_map),
            hcount=(0, 0),
            radical=(0, 0),
            lone_pairs=(0, 0),
            valence_electrons=(1, 1),
            neighbors=([], []),
            present=(True, True),
            typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
        )
        if src == dst:
            rc.add_edge(
                src,
                h,
                order=(1.0, 1.0),
                kekule_order=(1.0, 1.0),
                sigma_order=(1.0, 1.0),
                pi_order=(0.0, 0.0),
                standard_order=0.0,
            )
            continue
        rc.add_edge(
            src,
            h,
            order=(1.0, 0.0),
            kekule_order=(1.0, 0.0),
            sigma_order=(1.0, 0.0),
            pi_order=(0.0, 0.0),
            standard_order=1.0,
        )
        rc.add_edge(
            h,
            dst,
            order=(0.0, 1.0),
            kekule_order=(0.0, 1.0),
            sigma_order=(0.0, 1.0),
            pi_order=(0.0, 0.0),
            standard_order=-1.0,
        )

    for pair_id in explicit_pairs:
        src = pair_left[pair_id]
        dst = pair_right[pair_id]
        if src == dst:
            h0, h1 = rc.nodes[src]["hcount"]
            rc.nodes[src]["hcount"] = (h0 - 1, h1 - 1)
            continue
        src_h0, src_h1 = rc.nodes[src]["hcount"]
        dst_h0, dst_h1 = rc.nodes[dst]["hcount"]
        rc.nodes[src]["hcount"] = (src_h0 - 1, src_h1)
        rc.nodes[dst]["hcount"] = (dst_h0, dst_h1 - 1)

    for n in set(pair_left.values()) | set(pair_right.values()):
        if "typesGH" in rc.nodes[n]:
            t0, t1 = rc.nodes[n]["typesGH"]
            rc.nodes[n]["typesGH"] = (
                t0[:2] + (rc.nodes[n]["hcount"][0],) + t0[3:],
                t1[:2] + (rc.nodes[n]["hcount"][1],) + t1[3:],
            )

    _ensure_tuple_atom_maps(rc)
    return rc


def _ensure_tuple_atom_maps(graph: nx.Graph) -> None:
    """Assign stable paired atom maps to tuple nodes lacking visible maps."""
    used: set[int] = set()
    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map")
        values = atom_map if isinstance(atom_map, tuple) else (atom_map,)
        for value in {item for item in values if isinstance(item, int) and item > 0}:
            if value in used:
                raise ValueError(f"Duplicate atom map {value} in tuple graph.")
            used.add(value)
    fresh = max(used, default=0) + 1
    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map")
        if atom_map in (None, 0) or atom_map == (0, 0):
            while fresh in used:
                fresh += 1
            attrs["atom_map"] = (fresh, fresh)
            used.add(fresh)
            fresh += 1


# --------------------- SMARTS serialisation -----------------------
def _tuple_preserved_hydrogen_maps(its: nx.Graph) -> List[int]:
    """Collect reaction-centre H maps without extracting an RC subgraph."""

    def pair_changed(value: Any) -> bool:
        return (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and value[0] != value[1]
        )

    reaction_center_nodes = set()
    for node, attrs in its.nodes(data=True):
        lone_pairs = attrs.get("lone_pairs", attrs.get("lp"))
        if any(
            pair_changed(value)
            for value in (
                attrs.get("element"),
                attrs.get("hcount"),
                attrs.get("charge"),
                lone_pairs,
                attrs.get("radical"),
                attrs.get("valence_electrons"),
            )
        ):
            reaction_center_nodes.add(node)
    for left, right, attrs in its.edges(data=True):
        standard_order = attrs.get("standard_order", 0.0)
        if standard_order != 0 and standard_order != 0.0:
            reaction_center_nodes.update((left, right))

    if its.graph.get("stereo_changes"):
        from synkit.Graph.Stereo import stereo_complete_reaction_center_nodes

        reaction_center_nodes.update(stereo_complete_reaction_center_nodes(its))

    atom_maps = set()
    for node in reaction_center_nodes:
        attrs = its.nodes[node]
        element = attrs.get("element")
        elements = (
            element
            if isinstance(element, (tuple, list)) and len(element) == 2
            else (element,)
        )
        if "H" not in elements:
            continue
        atom_map = attrs.get("atom_map")
        if isinstance(atom_map, (tuple, list)) and len(atom_map) == 2:
            atom_map = atom_map[0]
        try:
            atom_maps.add(int(atom_map))
        except (TypeError, ValueError):
            continue
    return sorted(atom_maps)


def _tuple_endpoint_graphs(its: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    """Project both tuple endpoints in one ITS traversal.

    ``ITSReverter`` exposes one side at a time.  Serialization always needs
    both, so projecting them together avoids scanning and decoding every
    tuple attribute twice while preserving the same endpoint schema.
    """
    endpoints = (nx.Graph(), nx.Graph())
    node_keys = ITSReverter.DEFAULT_NODE_ATTRS
    edge_keys = ITSReverter.DEFAULT_EDGE_ATTRS

    def side_values(value: Any) -> Tuple[Any, Any]:
        if isinstance(value, tuple) and len(value) == 2:
            return value
        return value, value

    for node, attrs in its.nodes(data=True):
        present = attrs.get("present")
        if isinstance(present, tuple) and len(present) == 2:
            exists = bool(present[0]), bool(present[1])
        else:
            elements = side_values(attrs.get("element"))
            exists = tuple(value not in (None, "") for value in elements)
        projected = ({}, {})
        for key in node_keys:
            if key not in attrs:
                continue
            left_value, right_value = side_values(attrs[key])
            projected[0][key] = left_value
            projected[1][key] = right_value
        for side in (0, 1):
            if exists[side]:
                endpoints[side].add_node(node, **projected[side])

    for left, right, attrs in its.edges(data=True):
        orders = side_values(attrs.get("order"))
        kekule_orders = side_values(attrs.get("kekule_order"))
        bond_types = side_values(attrs.get("bond_type"))
        exists = tuple(
            orders[side] not in (None, "", 0, 0.0)
            or kekule_orders[side] not in (None, "", 0, 0.0)
            or bond_types[side] not in (None, "")
            for side in (0, 1)
        )
        projected = ({}, {})
        for key in edge_keys:
            if key not in attrs:
                continue
            left_value, right_value = side_values(attrs[key])
            projected[0][key] = left_value
            projected[1][key] = right_value
        for side in (0, 1):
            endpoint = endpoints[side]
            if left not in endpoint or right not in endpoint or not exists[side]:
                continue
            endpoint.add_edge(left, right, **projected[side])

    stereo = its.graph.get("stereo_descriptors", {})
    if isinstance(stereo, dict) and ("reactant" in stereo or "product" in stereo):
        for side_name, endpoint in zip(("reactant", "product"), endpoints):
            endpoint.graph["stereo_descriptors"] = dict(stereo.get(side_name, {}))
            endpoint.graph["stereo_projection"] = side_name
    elif isinstance(stereo, dict):
        for endpoint in endpoints:
            endpoint.graph["stereo_descriptors"] = dict(stereo)
    return endpoints


def _to_smarts(its: nx.Graph) -> str:
    electron_aware = bool(its.graph.get("electron_aware_rewrite", False))
    if electron_aware:
        left, right = _tuple_endpoint_graphs(its)
        preserved_hydrogens = _tuple_preserved_hydrogen_maps(its)
    else:
        left, right = its_decompose(its)
        preserved_hydrogens = []
    left = remove_wildcard_nodes(left)
    right = remove_wildcard_nodes(right)
    r_smi = graph_to_smi(left, preserve_atom_maps=preserved_hydrogens)
    if electron_aware:
        p_smi = None
        for candidate_index in range(2):
            product = (
                right
                if candidate_index == 0
                else _product_state._prepared_electron_product_graph(its)
            )
            if candidate_index or not its.graph.get(
                "_product_electron_fields_current", False
            ):
                product = refresh_electron_fields(product)
            for node, attrs in product.nodes(data=True):
                product_charge = _product_state._electron_product_charge(
                    its,
                    node,
                    attrs,
                )
                if product_charge is not None:
                    attrs["charge"] = product_charge
            if any(
                attrs.get("order") == 1.5 for _, _, attrs in product.edges(data=True)
            ):
                # Product connectivity may have changed while its retained
                # Kekule phase still reflects the substrate. Re-perceive
                # from the authoritative aromatic presentation here.
                p_smi = graph_to_smi(product, prefer_kekule_order=False)
                if p_smi is not None:
                    break
            try:
                p_smi = Chem.MolToSmiles(graph_to_sanitized_kekule_mol(product))
                break
            except Exception:
                p_smi = None
    else:
        p_smi = graph_to_smi(right)
    if r_smi is None or p_smi is None:
        return None
    return f"{r_smi}>>{p_smi}"
