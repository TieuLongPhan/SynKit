"""Electron-aware product-state helpers for the reactor facade.

This module is intentionally independent of SynReactor so the reactor remains
an orchestration facade rather than a dependency of its leaf helpers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Tuple

import networkx as nx
from rdkit import Chem

from synkit.Graph.Hyrogen._misc import implicit_hydrogen
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Mech.electron_accounting import refresh_electron_fields
from synkit.IO.chem_converter import _get_preserved_hydrogen_maps
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph

ITS_STRUCTURAL_EDGE_ATTRS = ["order", "kekule_order", "sigma_order", "pi_order"]


class ProductStatePerceptionError(RuntimeError):
    """Raised when a dirty aromatic product cannot be safely reperceived."""


def _product_value(value: Any) -> Any:
    """Return the product member of a scalar or endpoint pair."""
    if isinstance(value, tuple) and len(value) == 2:
        return value[1]
    return value


def _electron_refresh_support(  # noqa: C901
    its: nx.Graph,
    rewrite_nodes: Iterable[Any],
) -> frozenset[Any]:
    """Return the exact one-local support of product electron recomputation.

    The derived state at a vertex depends only on its intrinsic node fields
    and on the electron orders of incident edges.  The rewrite implementation
    writes intrinsic fields only at ``rewrite_nodes`` and writes edges only
    between such nodes.  A presence change can additionally remove every
    external incident edge, so neighbors across that boundary are included
    whenever the actual endpoint edge state differs.  This is the complete
    support of the local formal-charge functional, not a corpus assumption.
    """

    def side_values(value: Any) -> Tuple[Any, Any]:
        if isinstance(value, tuple) and len(value) == 2:
            return value
        return value, value

    def node_exists(node: Any, side: int) -> bool:
        attrs = its.nodes[node]
        present = attrs.get("present")
        if isinstance(present, tuple) and len(present) == 2:
            return bool(present[side])
        return side_values(attrs.get("element"))[side] not in (None, "")

    def edge_state(left: Any, right: Any, attrs: Mapping[str, Any], side: int) -> Any:
        if not node_exists(left, side) or not node_exists(right, side):
            return None
        values = tuple(
            side_values(attrs.get(name))[side]
            for name in (
                "order",
                "kekule_order",
                "sigma_order",
                "pi_order",
                "bond_type",
            )
        )
        if all(value in (None, "", 0, 0.0) for value in values):
            return None
        return values

    support = {node for node in rewrite_nodes if node in its}

    # Every edge written by the rewrite has both endpoints in ``support``.
    # Those endpoints are already scheduled, regardless of the edge delta.
    # An edge crossing out of the mapped locus can therefore change endpoint
    # state only when its mapped endpoint changes presence.  Test only those
    # boundary edges instead of rescanning every incident edge of every
    # application.  This is the same one-local support, derived directly from
    # the rewrite locality invariant above.
    presence_changes = []
    for node in support:
        if node_exists(node, 0) != node_exists(node, 1):
            presence_changes.append(node)
    for node in presence_changes:
        for left, right, attrs in its.edges(node, data=True):
            if left in support and right in support:
                continue
            if edge_state(left, right, attrs, 0) != edge_state(left, right, attrs, 1):
                support.update((left, right))
    return frozenset(support)


def _template_charge_is_authoritative(
    its: nx.Graph,
    node: Any,
    product_attrs: Mapping[str, Any],
) -> bool:
    """Return whether the source state lies outside the local Lewis model.

    Product charge is derived only when both parsed template endpoints satisfy
    the same ``VE - NBE - B`` identity.  An inconsistent endpoint is evidence
    that the ordinary graph omits electronic context, so its exact charge is
    retained in either replay direction.  This is a representation invariant
    and does not classify atoms by element.
    """
    if node not in its:
        return True
    attrs = its.nodes[node]
    model_consistent = attrs.get("charge_model_consistent")
    if (
        isinstance(model_consistent, tuple)
        and len(model_consistent) == 2
        and all(isinstance(value, bool) for value in model_consistent)
    ):
        return not all(model_consistent)
    present = attrs.get("present")
    if isinstance(present, tuple) and len(present) == 2 and not present[0]:
        return True

    def reactant_value(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 2:
            return value[0]
        return value

    required = ("valence_electrons", "lone_pairs", "radical", "hcount", "charge")
    values = {name: reactant_value(attrs.get(name)) for name in required}
    if any(value is None for value in values.values()):
        return True

    bond_sum = 0.0
    for _, _, edge_attrs in its.edges(node, data=True):
        sigma = reactant_value(edge_attrs.get("sigma_order"))
        pi = reactant_value(edge_attrs.get("pi_order"))
        if sigma is None or pi is None:
            order = reactant_value(edge_attrs.get("kekule_order"))
            if order is None:
                order = reactant_value(edge_attrs.get("order"))
            if order is None:
                return True
            bond_sum += float(order)
        else:
            bond_sum += float(sigma) + float(pi)

    modeled_charge = (
        float(values["valence_electrons"])
        - 2.0 * float(values["lone_pairs"])
        - float(values["radical"])
        - float(values["hcount"])
        - bond_sum
    )
    return modeled_charge != float(values["charge"])


def _pair_electron_aware_node_attrs(
    host_n: Dict[str, Any],
    rc_n: Dict[str, Any],
    *,
    preserve_unchanged_state: bool = False,
    relative_resources: frozenset[str] = frozenset(),
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
    model_consistent = rc_n.get("charge_model_consistent")
    if isinstance(model_consistent, tuple) and len(model_consistent) == 2:
        host_n["charge_model_consistent"] = (
            bool(model_consistent[0]),
            bool(model_consistent[1]),
        )

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
            if key in relative_resources:
                product_value = left_value - rc_value[0] + rc_value[1]
            else:
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
    # Direct refresh is valid while the product Kekule phase is unchanged.
    if not its.graph.get("_product_kekule_phase_dirty", True):
        _refresh_product_electron_fields_direct(its)
        return

    product = _prepared_electron_product_graph(its)
    refreshed = refresh_electron_fields(product)
    structural_dirty_nodes = its.graph.get("_structural_exact_dirty_nodes")
    structural_dirty_edges = its.graph.get("_structural_exact_dirty_edges")
    for node, attrs in refreshed.nodes(data=True):
        its.nodes[node].pop("_structural_exact_node_sig", None)
        if isinstance(structural_dirty_nodes, set):
            structural_dirty_nodes.add(node)
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
        its.edges[u, v].pop("_structural_exact_edge_sig", None)
        if isinstance(structural_dirty_edges, set):
            structural_dirty_edges.add((u, v))
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


def _refresh_product_electron_fields_direct(its: nx.Graph) -> None:  # noqa: C901
    """Refresh derived product fields without materialising a side graph.

    This path is exact while the aromatic Kekule phase is unchanged.
    Explicit hydrogen collapse preserves the heavy atom's electron count:
    a removed H--X sigma bond becomes one unit of ``hcount``.
    """

    def product_node_exists(attrs: Mapping[str, Any]) -> bool:
        present = attrs.get("present")
        if isinstance(present, tuple) and len(present) == 2:
            return bool(present[1])
        return _product_value(attrs.get("element")) not in (None, "")

    def product_edge_exists(attrs: Mapping[str, Any]) -> bool:
        for name in ("order", "kekule_order", "bond_type"):
            value = _product_value(attrs.get(name))
            if value not in (None, "", 0, 0.0):
                return True
        return False

    refresh_hint = its.graph.get("_product_refresh_nodes")
    candidate_nodes = (
        tuple(its)
        if refresh_hint is None
        else tuple(node for node in refresh_hint if node in its)
    )
    product_nodes = {
        node for node in candidate_nodes if product_node_exists(its.nodes[node])
    }

    # Every changed edge is incident to a mapped rule node.  Build the exact
    # local edge population once; unchanged edges elsewhere already carry
    # valid scalar host fields and need no tuple-side refresh.
    product_edges = []
    structural_dirty_nodes = its.graph.get("_structural_exact_dirty_nodes")
    structural_dirty_edges = its.graph.get("_structural_exact_dirty_edges")
    seen_edges: set[frozenset[Any]] = set()
    for node in product_nodes:
        for left, right, attrs in its.edges(node, data=True):
            edge_key = frozenset((left, right))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            if (
                left in its
                and right in its
                and product_node_exists(its.nodes[left])
                and product_node_exists(its.nodes[right])
                and product_edge_exists(attrs)
            ):
                product_edges.append((left, right, attrs))

    bond_sums: Dict[Any, float] = defaultdict(float)
    for left, right, attrs in product_edges:
        attrs.pop("_structural_exact_edge_sig", None)
        if isinstance(structural_dirty_edges, set):
            structural_dirty_edges.add((left, right))
        sigma = float(_product_value(attrs.get("sigma_order", 0.0)) or 0.0)
        pi = float(_product_value(attrs.get("pi_order", 0.0)) or 0.0)
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
        attrs.pop("_structural_exact_node_sig", None)
        if isinstance(structural_dirty_nodes, set):
            structural_dirty_nodes.add(node)
        bond_sum = bond_sums[node]
        current_bond_sum = attrs.get("bond_order_sum")
        left_bond_sum = (
            current_bond_sum[0]
            if isinstance(current_bond_sum, tuple) and len(current_bond_sum) == 2
            else current_bond_sum
        )
        attrs["bond_order_sum"] = (left_bond_sum, bond_sum)

        valence_electrons = _product_value(attrs.get("valence_electrons"))
        if valence_electrons is None:
            continue
        lone_pairs = float(_product_value(attrs.get("lone_pairs", 0)) or 0)
        radical = float(_product_value(attrs.get("radical", 0)) or 0)
        hcount = float(_product_value(attrs.get("hcount", 0)) or 0)
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
        represented_charge = _product_value(attrs.get("charge", 0))
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
        product_charge = _electron_product_charge(
            its,
            node,
            {
                "aromatic": bool(_product_value(attrs.get("aromatic", False))),
                "element": _product_value(attrs.get("element")),
                "recomputed_charge": recomputed_charge,
            },
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

    refresh_hint = its.graph.get("_product_refresh_nodes")
    candidate_nodes = (
        tuple(its)
        if refresh_hint is None
        else tuple(node for node in refresh_hint if node in its)
    )
    aromatic_nodes = {
        node
        for node in candidate_nodes
        if any(
            bool(value) for value in side_values(its.nodes[node].get("aromatic", False))
        )
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

    seen_edges: set[frozenset[Any]] = set()
    for node in aromatic_nodes:
        for left, right, attrs in its.edges(node, data=True):
            edge_key = frozenset((left, right))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
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

    Recompute charge when the template endpoints satisfy the local Lewis
    identity. Preserve explicit template charge for aromatic or
    model-inconsistent endpoints.
    """
    if node in its:
        template_charge = its.nodes[node].get("template_charge")
        aromatic = product_attrs.get("aromatic", its.nodes[node].get("aromatic"))
        if isinstance(template_charge, tuple) and len(template_charge) == 2:
            if aromatic is True or _template_charge_is_authoritative(
                its,
                node,
                product_attrs,
            ):
                return template_charge[1]
        else:
            return _product_value(its.nodes[node].get("charge"))
    return product_attrs.get("recomputed_charge")


def _reperceive_product_kekule_phase(  # noqa: C901
    product: nx.Graph,
    its: nx.Graph,
) -> nx.Graph:
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
            prefer_kekule_order=False,
        )
    except Exception as exc:
        raise ProductStatePerceptionError(
            "Could not reperceive the dirty aromatic product state."
        ) from exc

    refreshed = product.copy()
    try:
        # Only three edge fields are needed here.  Building a complete
        # MolToGraph representation also derives every atom descriptor and
        # stereo registry, which is substantially more work.  Read the
        # Kekule bond phase directly from a sanitized RDKit copy instead.
        kekule = Chem.Mol(mol)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        node_by_index = tuple(
            atom.GetAtomMapNum() or atom.GetIdx() + 1 for atom in kekule.GetAtoms()
        )
        if len(node_by_index) != len(set(node_by_index)) or set(node_by_index) != set(
            refreshed
        ):
            raise ValueError("Sanitized product atom identities are not bijective.")
        for bond in kekule.GetBonds():
            left = node_by_index[bond.GetBeginAtomIdx()]
            right = node_by_index[bond.GetEndAtomIdx()]
            if not refreshed.has_edge(left, right):
                raise ValueError("Sanitized product bond is absent from the graph.")
            order = float(bond.GetBondTypeAsDouble())
            refreshed[left][right]["kekule_order"] = order
            refreshed[left][right]["sigma_order"] = 1.0 if order > 0 else 0.0
            refreshed[left][right]["pi_order"] = max(0.0, order - 1.0)
    except Exception:
        # Preserve the established conservative path for unusual third-party
        # graphs whose atom maps cannot identify the sanitized RDKit bonds.
        try:
            reperceived = MolToGraph(attr_profile="minimal").transform(
                mol,
                use_index_as_atom_map=True,
            )
        except Exception as exc:
            raise ProductStatePerceptionError(
                "Could not reperceive the dirty aromatic product state."
            ) from exc
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
