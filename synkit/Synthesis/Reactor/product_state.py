"""Electron-aware product-state helpers for the reactor facade.

This module is intentionally independent of SynReactor so the reactor remains
an orchestration facade rather than a dependency of its leaf helpers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Tuple

import networkx as nx

from synkit.Graph.Hyrogen._misc import implicit_hydrogen
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Mech.electron_accounting import refresh_electron_fields
from synkit.IO.chem_converter import _get_preserved_hydrogen_maps
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph

ITS_STRUCTURAL_EDGE_ATTRS = ["order", "kekule_order", "sigma_order", "pi_order"]


def _pair_electron_aware_node_attrs(
    host_n: Dict[str, Any],
    rc_n: Dict[str, Any],
    *,
    preserve_unchanged_state: bool = False,
) -> None:
    """Store paired attrs, preserving generic relative-query state locally."""
    _, product_types = host_n["typesGH"]
    rc_present = rc_n.get("present")
    reactant_is_absent = (
        isinstance(rc_present, tuple) and len(rc_present) == 2 and not rc_present[0]
    )
    legacy_product_values = {
        "element": product_types[0],
        "aromatic": product_types[1],
        "hcount": product_types[2],
        "neighbors": product_types[4],
    }

    for key, product_value in legacy_product_values.items():
        left_value = host_n.get(key)
        product_is_absent = (
            isinstance(rc_present, tuple) and len(rc_present) == 2 and not rc_present[1]
        )
        rc_value = rc_n.get(key)
        if reactant_is_absent and isinstance(rc_value, tuple) and len(rc_value) == 2:
            product_value = rc_value[1]
        if key == "element" and product_value == "*" and not product_is_absent:
            product_value = left_value
        host_n[key] = (left_value, product_value)

    for key in ("radical", "lone_pairs", "valence_electrons"):
        rc_value = rc_n.get(key)
        if isinstance(rc_value, tuple) and len(rc_value) == 2:
            left_value = host_n.get(key)
            if left_value is None:
                left_value = rc_value[0]
            product_value = (
                left_value
                if preserve_unchanged_state and rc_value[0] == rc_value[1]
                else rc_value[1]
            )
            host_n[key] = (left_value, product_value)

    host_n["template_charge"] = (host_n.get("charge"), product_types[3])

    # Electron-authoritative RCs derive charge at the product boundary.
    # Keep the reactant-side value temporarily so mutation does not copy
    # the RC's product charge label.
    host_n["charge"] = (host_n.get("charge"), host_n.get("charge"))

    if "atom_map" in host_n:
        host_n["atom_map"] = (host_n["atom_map"], host_n["atom_map"])
    if isinstance(rc_present, tuple) and len(rc_present) == 2:
        host_n["present"] = (bool(host_n.get("present", True)), rc_present[1])
    else:
        host_n["present"] = (True, True)


def _ensure_host_atom_maps(host: nx.Graph) -> None:
    """Assign stable fresh atom maps to unmapped host atoms."""
    used: set[int] = set()
    for node, attrs in host.nodes(data=True):
        atom_map = attrs.get("atom_map")
        if not isinstance(atom_map, int) or atom_map <= 0:
            continue
        if atom_map in used:
            raise ValueError(f"Duplicate atom map {atom_map} on host graph.")
        used.add(atom_map)
    fresh = max(used, default=0) + 1
    for node, attrs in host.nodes(data=True):
        if not isinstance(attrs.get("atom_map"), int) or attrs["atom_map"] <= 0:
            while fresh in used:
                fresh += 1
            attrs["atom_map"] = fresh
            used.add(fresh)
            fresh += 1


def _refresh_product_electron_fields(
    its: nx.Graph,
) -> None:
    """Refresh product-side electron fields from the scalar product graph."""
    # In the common case the product Kekule phase is already valid.  The
    # scalar projection previously built here only supplied values that
    # are directly derivable from the product half of the ITS tuples.
    # Computing them in place avoids two graph copies, one ITS traversal,
    # and explicit-H collapse for every candidate.
    if not its.graph.get("_product_kekule_phase_dirty", True):
        _refresh_product_electron_fields_direct(its)
        return

    product = _prepared_electron_product_graph(its)
    refreshed = refresh_electron_fields(product)
    for node, attrs in refreshed.nodes(data=True):
        current_charge = its.nodes[node].get("charge")
        left_charge = (
            current_charge[0]
            if isinstance(current_charge, tuple) and len(current_charge) == 2
            else current_charge
        )
        product_charge = _electron_product_charge(its, node, attrs)
        if product_charge is not None:
            its.nodes[node]["charge"] = (left_charge, product_charge)

        template_charge = its.nodes[node].get("template_charge")
        if isinstance(template_charge, tuple) and len(template_charge) == 2:
            attrs["charge_mismatch"] = template_charge[1] != attrs.get(
                "recomputed_charge"
            )

        for key in ("bond_order_sum", "recomputed_charge", "charge_mismatch"):
            if key in attrs:
                current = its.nodes[node].get(key)
                left_value = (
                    current[0]
                    if isinstance(current, tuple) and len(current) == 2
                    else current
                )
                its.nodes[node][key] = (left_value, attrs[key])
    for u, v, attrs in refreshed.edges(data=True):
        for key in ("kekule_order", "sigma_order", "pi_order"):
            if key not in attrs:
                continue
            current = its.edges[u, v].get(key)
            left_value = (
                current[0]
                if isinstance(current, tuple) and len(current) == 2
                else current
            )
            its.edges[u, v][key] = (left_value, attrs[key])
    its.graph["_product_electron_fields_current"] = True


def _refresh_product_electron_fields_direct(its: nx.Graph) -> None:
    """Refresh derived product fields without materialising a side graph.

    This path is exact while the aromatic Kekule phase is unchanged.
    Explicit hydrogen collapse preserves the heavy atom's electron count:
    a removed H--X sigma bond becomes one unit of ``hcount``.
    """

    def product_value(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 2:
            return value[1]
        return value

    def product_node_exists(attrs: Mapping[str, Any]) -> bool:
        present = attrs.get("present")
        if isinstance(present, tuple) and len(present) == 2:
            return bool(present[1])
        return product_value(attrs.get("element")) not in (None, "")

    def product_edge_exists(attrs: Mapping[str, Any]) -> bool:
        for name in ("order", "kekule_order", "bond_type"):
            value = product_value(attrs.get(name))
            if value not in (None, "", 0, 0.0):
                return True
        return False

    product_nodes = {
        node for node, attrs in its.nodes(data=True) if product_node_exists(attrs)
    }
    product_edges = [
        (left, right, attrs)
        for left, right, attrs in its.edges(data=True)
        if left in product_nodes
        and right in product_nodes
        and product_edge_exists(attrs)
    ]

    bond_sums: Dict[Any, float] = defaultdict(float)
    for left, right, attrs in product_edges:
        sigma = float(product_value(attrs.get("sigma_order", 0.0)) or 0.0)
        pi = float(product_value(attrs.get("pi_order", 0.0)) or 0.0)
        bond_order = sigma + pi
        bond_sums[left] += bond_order
        bond_sums[right] += bond_order

        current = attrs.get("kekule_order")
        left_value = (
            current[0] if isinstance(current, tuple) and len(current) == 2 else current
        )
        attrs["kekule_order"] = (left_value, bond_order)

    for node in product_nodes:
        attrs = its.nodes[node]
        bond_sum = bond_sums[node]
        current_bond_sum = attrs.get("bond_order_sum")
        left_bond_sum = (
            current_bond_sum[0]
            if isinstance(current_bond_sum, tuple) and len(current_bond_sum) == 2
            else current_bond_sum
        )
        attrs["bond_order_sum"] = (left_bond_sum, bond_sum)

        valence_electrons = product_value(attrs.get("valence_electrons"))
        if valence_electrons is None:
            continue
        lone_pairs = float(product_value(attrs.get("lone_pairs", 0)) or 0)
        radical = float(product_value(attrs.get("radical", 0)) or 0)
        hcount = float(product_value(attrs.get("hcount", 0)) or 0)
        recomputed_charge = (
            float(valence_electrons) - 2.0 * lone_pairs - radical - hcount - bond_sum
        )
        if recomputed_charge.is_integer():
            recomputed_charge = int(recomputed_charge)

        current_recomputed = attrs.get("recomputed_charge")
        left_recomputed = (
            current_recomputed[0]
            if isinstance(current_recomputed, tuple) and len(current_recomputed) == 2
            else current_recomputed
        )
        attrs["recomputed_charge"] = (left_recomputed, recomputed_charge)

        template_charge = attrs.get("template_charge")
        represented_charge = product_value(attrs.get("charge", 0))
        mismatch = float(represented_charge or 0) != recomputed_charge
        if isinstance(template_charge, tuple) and len(template_charge) == 2:
            mismatch = template_charge[1] != recomputed_charge
        current_mismatch = attrs.get("charge_mismatch")
        left_mismatch = (
            current_mismatch[0]
            if isinstance(current_mismatch, tuple) and len(current_mismatch) == 2
            else current_mismatch
        )
        attrs["charge_mismatch"] = (left_mismatch, mismatch)

        current_charge = attrs.get("charge")
        left_charge = (
            current_charge[0]
            if isinstance(current_charge, tuple) and len(current_charge) == 2
            else current_charge
        )
        aromatic = bool(product_value(attrs.get("aromatic", False)))
        product_charge = (
            template_charge[1]
            if aromatic
            and isinstance(template_charge, tuple)
            and len(template_charge) == 2
            else recomputed_charge
        )
        attrs["charge"] = (left_charge, product_charge)

    its.graph["_product_electron_fields_current"] = True


def _product_kekule_phase_is_dirty(its: nx.Graph) -> bool:
    """Return whether a rewrite can invalidate an aromatic Kekule phase.

    Substituent and hydrogen-count edits do not alter the alternating phase
    inside an aromatic system.  Electronic changes on aromatic atoms and
    edits to bonds within that system do, and therefore still require full
    RDKit re-perception.
    """

    def side_values(value: Any) -> Tuple[Any, Any]:
        if isinstance(value, tuple) and len(value) == 2:
            return value
        return value, value

    aromatic_nodes = {
        node
        for node, attrs in its.nodes(data=True)
        if any(bool(value) for value in side_values(attrs.get("aromatic", False)))
    }
    if not aromatic_nodes:
        return False

    for node in aromatic_nodes:
        attrs = its.nodes[node]
        for name in (
            "element",
            "aromatic",
            "radical",
            "lone_pairs",
            "valence_electrons",
            "present",
            "template_charge",
        ):
            left, right = side_values(attrs.get(name))
            if left != right:
                return True

    for left, right, attrs in its.edges(data=True):
        if left not in aromatic_nodes or right not in aromatic_nodes:
            continue
        for name in ITS_STRUCTURAL_EDGE_ATTRS:
            before, after = side_values(attrs.get(name))
            if before != after:
                return True
    return False


def _prepared_electron_product_graph(its: nx.Graph) -> nx.Graph:
    """Build the scalar product graph used for electron recomputation."""
    product = ITSReverter(its).to_product_graph()
    preserved_hydrogens = _get_preserved_hydrogen_maps(its, "tuple")
    product = implicit_hydrogen(product, set(preserved_hydrogens))
    return _reperceive_product_kekule_phase(product, its)


def _electron_product_charge(
    its: nx.Graph,
    node: Any,
    product_attrs: Mapping[str, Any],
) -> Any:
    """Choose the product charge used for electron-aware serialization.

    Non-aromatic tuple products are electron-authoritative and use the
    recomputed formal charge. Aromatic tuple products are still an open
    representation boundary: if the template explicitly carries a product
    charge, preserve it instead of inventing cationic aromatic carbons from
    an incomplete Kekule phase.
    """
    if node in its:
        template_charge = its.nodes[node].get("template_charge")
        aromatic = product_attrs.get("aromatic", its.nodes[node].get("aromatic"))
        if (
            aromatic is True
            and isinstance(template_charge, tuple)
            and len(template_charge) == 2
        ):
            return template_charge[1]
    return product_attrs.get("recomputed_charge")


def _reperceive_product_kekule_phase(product: nx.Graph, its: nx.Graph) -> nx.Graph:
    """Refresh aromatic sigma/pi phase from full product presentation bonds."""
    if not any(data.get("order") == 1.5 for _, _, data in product.edges(data=True)):
        return product
    if not its.graph.get("_product_kekule_phase_dirty", True):
        return product

    probe = product.copy()
    for node, attrs in probe.nodes(data=True):
        template_charge = its.nodes[node].get("template_charge")
        if isinstance(template_charge, tuple) and len(template_charge) == 2:
            attrs["charge"] = template_charge[1]

    try:
        mol = GraphToMol(edge_attributes={"order": "order"}).graph_to_mol(
            probe,
            sanitize=True,
            use_h_count=True,
        )
        reperceived = MolToGraph(attr_profile="minimal").transform(
            mol,
            use_index_as_atom_map=True,
        )
    except Exception:
        return product

    refreshed = product.copy()
    for u, v in refreshed.edges():
        if not reperceived.has_edge(u, v):
            continue
        for key in ("kekule_order", "sigma_order", "pi_order"):
            if key in reperceived[u][v]:
                refreshed[u][v][key] = reperceived[u][v][key]
    return refreshed


def _product_graph_for_diagnostics(its: nx.Graph) -> nx.Graph:
    """Return the product graph matching the rewrite representation."""
    if its.graph.get("electron_aware_rewrite", False):
        return ITSReverter(its).to_product_graph()
    return its_decompose(its)[1]
