"""Graph preparation and ITS gluing helpers for SynReactor."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx

from synkit.Graph import add_wildcard_subgraph_for_unmapped
from synkit.Graph.Hyrogen._misc import h_to_explicit
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.IO.chem_converter import ITSFormat, detect_its_format
from synkit.Synthesis.Reactor import product_state as _product_state
from synkit.Synthesis.Reactor.strategy import Strategy

MappingDict = Dict[Any, Any]


def _implicit_heavy_hydrogens(
    graph: nx.Graph,
    *,
    preserve_mapped_hydrogens: bool = False,
) -> nx.Graph:
    """Convert ordinary heavy-atom-bound explicit H nodes into hcount."""
    normalized = graph.copy()
    removable = []
    for node, attrs in normalized.nodes(data=True):
        if attrs.get("element") != "H":
            continue
        neighbors = list(normalized.neighbors(node))
        heavy_neighbors = [
            nbr for nbr in neighbors if normalized.nodes[nbr].get("element") != "H"
        ]
        if heavy_neighbors and len(heavy_neighbors) == len(neighbors):
            removable.append((node, heavy_neighbors))

    for h, heavy_neighbors in removable:
        if not normalized.has_node(h):
            continue
        for heavy in heavy_neighbors:
            atom_map = normalized.nodes[h].get("atom_map")
            if preserve_mapped_hydrogens and isinstance(atom_map, int) and atom_map > 0:
                stored = list(normalized.nodes[heavy].get("_implicit_h_atom_maps", ()))
                stored.append(atom_map)
                normalized.nodes[heavy]["_implicit_h_atom_maps"] = tuple(stored)
            normalized.nodes[heavy]["hcount"] = (
                normalized.nodes[heavy].get("hcount", 0) + 1
            )
        normalized.remove_node(h)
    return normalized


def _invert_template(
    tpl: nx.Graph,
    balance_its: bool = True,
    format: ITSFormat | None = None,
) -> nx.Graph:
    resolved_format = format or detect_its_format(tpl)
    if resolved_format == "tuple":
        reverter = ITSReverter(tpl)
        l, r = reverter.to_reactant_graph(), reverter.to_product_graph()
        return ITSConstruction().construct(
            r,
            l,
            balance_its=balance_its,
        )
    l, r = its_decompose(tpl)
    return ITSConstruction().ITSGraph(r, l, balance_its=balance_its)


# ==================================================================
# Aux – glue, explicit‑H, SMARTS
# ==================================================================
def _node_glue(
    host_n: Dict[str, Any], pat_n: Dict[str, Any], key: str = "typesGH"
) -> None:
    host_r, host_p = host_n[key]
    pat_r, pat_p = pat_n[key]
    delta = pat_r[2] - pat_p[2]
    if pat_r[0] == "*":
        new_r = host_r
    else:
        new_r = host_r[:2] + (host_r[2],) + host_r[3:]
    if pat_p[0] == "*":
        new_p = host_p[:2] + (host_r[2] - delta,) + (host_p[3],) + host_p[4:]
    else:
        new_p = host_p[:2] + (host_r[2] - delta,) + (pat_p[3],) + host_p[4:]
    # if pat_r[0] == '*':
    #     host_r[0] = '*'
    # if pat_p[0] == '*':
    #     host_p[0] = '*'
    host_n[key] = (new_r, new_p)

    for key in ("h_pairs", "h_pairs_left", "h_pairs_right", "h_pair_atom_maps"):
        if key in pat_n:
            host_n[key] = pat_n[key]


def _get_explicit_map(
    host: nx.Graph,
    mapping: MappingDict,
    pattern_explicit: nx.Graph | None = None,
    strategy: Strategy = Strategy.ALL,
    embed_threshold: float = None,
    embed_pre_filter: bool = False,
):
    expand_nodes = [v for _, v in mapping.items()]
    original_nodes = set(host)
    host_explicit = h_to_explicit(host, expand_nodes)
    for node in set(host_explicit) - original_nodes:
        if not host_explicit.nodes[node].get("_restored_mapped_h"):
            host_explicit.nodes[node]["_pattern_expanded_h"] = True
    mappings = SubgraphSearchEngine.find_subgraph_mappings(
        host=host_explicit,
        pattern=pattern_explicit or nx.Graph(),
        node_attrs=["element", "charge"],
        edge_attrs=["order"],
        strategy=strategy,
        threshold=embed_threshold,
        pre_filter=embed_pre_filter,
    )

    # Atom maps are not chemical matching constraints, but when applying a
    # reviewed mapped rule back to its mapped host they are the deterministic
    # tie-breaker between symmetry-equivalent explicit hydrogens.
    mappings.sort(
        key=lambda candidate: _mapping_atom_map_alignment(
            pattern_explicit,
            host_explicit,
            candidate,
        ),
        reverse=True,
    )
    return mappings, host_explicit


def _mapping_atom_map_alignment(
    pattern: nx.Graph,
    host: nx.Graph,
    mapping: MappingDict,
) -> int:
    """Count positive reactant-side AAM identities in one candidate map."""

    def reactant_map(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 2:
            return value[0]
        return value

    score = 0
    for pattern_node, host_node in mapping.items():
        pattern_map = reactant_map(pattern.nodes[pattern_node].get("atom_map"))
        host_map = reactant_map(host.nodes[host_node].get("atom_map"))
        if isinstance(pattern_map, int) and pattern_map > 0:
            score += pattern_map == host_map
    return score


def _restore_unmatched_pattern_hydrogens(
    graph: nx.Graph,
    mapping: MappingDict,
) -> None:
    """Fold unmatched temporary H expansions back into heavy-atom hcount."""
    matched_nodes = set(mapping.values())
    removable = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("_pattern_expanded_h") and node not in matched_nodes
    ]
    for hydrogen in removable:
        neighbors = list(graph.neighbors(hydrogen))
        if len(neighbors) != 1:
            continue
        heavy = neighbors[0]
        graph.nodes[heavy]["hcount"] = graph.nodes[heavy].get("hcount", 0) + 1
        types = graph.nodes[heavy].get("typesGH")
        if (
            isinstance(types, tuple)
            and len(types) == 2
            and isinstance(types[0], tuple)
            and len(types[0]) >= 3
        ):
            graph.nodes[heavy]["typesGH"] = (
                types[0][:2] + (types[0][2] + 1,) + types[0][3:],
                types[1],
            )
        graph.remove_node(hydrogen)


def _glue_graph(
    host: nx.Graph,
    rc: nx.Graph,
    mapping: MappingDict,
    pattern_has_explicit_H: bool = False,
    pattern_explicit: nx.Graph | None = None,
    strategy: Strategy = Strategy.ALL,
    embed_threshold: float = None,
    embed_pre_filter: bool = False,
    relative_pi_edges: set[frozenset[Any]] | None = None,
    restore_unmatched_explicit_h: bool = True,
    refresh_electrons: bool = True,
    electron_aware: bool | None = None,
) -> List[nx.Graph]:
    list_its: List[nx.Graph] = []
    # NetworkX copies node/edge attribute dictionaries.  Rewrite values
    # are replaced rather than mutated in place, so recursively copying
    # every tuple and stereo descriptor only adds mapping-sized overhead.
    host_g = host.copy()
    if electron_aware is None:
        electron_aware = _is_electron_aware_template(rc)

    def _default_tg(a: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        tpl = (
            a.get("element", "*"),
            a.get("aromatic", False),
            a.get("hcount", 0),
            a.get("charge", 0),
            a.get("neighbors", []),
        )
        return tpl, tpl

    for _, data in host_g.nodes(data=True):
        data.setdefault("typesGH", _default_tg(data))
    if electron_aware:
        _product_state._ensure_host_atom_maps(host_g)

    if pattern_has_explicit_H:
        mappings, host_g = _get_explicit_map(
            host_g,
            mapping,
            pattern_explicit,
            strategy,
            embed_threshold,
            embed_pre_filter,
        )
        if electron_aware:
            _product_state._ensure_host_atom_maps(host_g)
    else:
        mappings = [mapping]

    relative_pi_centers = {
        node for edge in (relative_pi_edges or set()) for node in edge
    }

    # Iterate over remappings --------------------------------------
    reuse_prepared_host = len(mappings) == 1
    for m in mappings:

        its = host_g if reuse_prepared_host else host_g.copy()
        if pattern_has_explicit_H and restore_unmatched_explicit_h:
            _restore_unmatched_pattern_hydrogens(its, m)
        # This should only work for implict cases
        if len(m.keys()) < rc.number_of_nodes():
            its, m = add_wildcard_subgraph_for_unmapped(
                its,
                rc,
                m,
                inplace=True,
                tuple_mode=electron_aware,
            )

        for _, _, data in its.edges(data=True):
            o = data.get("order", 1.0)
            data["order"] = (o, o)
            if electron_aware:
                sigma = data.get("sigma_order", 1.0 if o else 0.0)
                pi = data.get("pi_order", max(0.0, float(o) - 1.0))
                data["sigma_order"] = (sigma, sigma)
                data["pi_order"] = (pi, pi)
            data.setdefault("standard_order", 0.0)

        for _, data in rc.nodes(data=True):
            data.setdefault("typesGH", _default_tg(data))

        # merge nodes -------------------------------------------
        for rc_n, host_n in m.items():
            if its.has_node(host_n):
                _node_glue(its.nodes[host_n], rc.nodes[rc_n])
                if electron_aware:
                    rc_element = rc.nodes[rc_n].get("element")
                    wildcard_context = (
                        isinstance(rc_element, tuple)
                        and len(rc_element) == 2
                        and rc_element[0] == rc_element[1] == "*"
                    )
                    _product_state._pair_electron_aware_node_attrs(
                        its.nodes[host_n],
                        rc.nodes[rc_n],
                        preserve_unchanged_state=(
                            rc_n in relative_pi_centers or wildcard_context
                        ),
                    )

        # merge edges (additive order) ---------------------------
        for u, v, rc_attr in rc.edges(data=True):
            hu, hv = m.get(u), m.get(v)
            if hu is None or hv is None:
                continue
            if not its.has_edge(hu, hv):
                its.add_edge(hu, hv, **dict(rc_attr))
            else:
                host_attr = its[hu][hv]
                rc_order = rc_attr.get("order", (0, 0))
                if relative_pi_edges and frozenset((u, v)) in relative_pi_edges:
                    # Apply the rule delta to the matched host edge. Thus
                    # C=C -> C-C naturally becomes C#C -> C=C on an
                    # alkyne, while both still remove exactly one pi bond.
                    for key in (
                        "order",
                        "kekule_order",
                        "sigma_order",
                        "pi_order",
                    ):
                        host_value = host_attr.get(key)
                        rule_value = rc_attr.get(key)
                        if not (
                            isinstance(host_value, tuple)
                            and len(host_value) == 2
                            and isinstance(rule_value, tuple)
                            and len(rule_value) == 2
                        ):
                            continue
                        delta = float(rule_value[0]) - float(rule_value[1])
                        product_value = float(host_value[0]) - delta
                        if product_value.is_integer():
                            product_value = int(product_value)
                        host_attr[key] = (host_value[0], product_value)
                    host_attr["standard_order"] = rc_attr.get("standard_order", 0.0)
                elif rc_order[0] == 0:  # additive only on product side
                    ho = host_attr["order"]
                    host_attr["order"] = (ho[0], round(ho[1] + rc_order[1]))
                    if electron_aware:
                        host_sigma = host_attr.get("sigma_order", (0.0, 0.0))
                        host_pi = host_attr.get("pi_order", (0.0, 0.0))
                        rc_sigma = rc_attr.get("sigma_order", (0.0, 0.0))
                        rc_pi = rc_attr.get("pi_order", (0.0, 0.0))
                        host_attr["sigma_order"] = (
                            host_sigma[0],
                            host_sigma[1] + rc_sigma[1],
                        )
                        host_attr["pi_order"] = (
                            host_pi[0],
                            host_pi[1] + rc_pi[1],
                        )
                    host_attr["standard_order"] += rc_attr.get("standard_order", 0.0)
                else:
                    # The host is the authoritative reactant endpoint.
                    # Independently parsed aromatic graphs may have an
                    # equivalent but different Kekule phase from the rule.
                    # Replacing the complete rule tuple here used to copy
                    # that phase onto the host half and could serialize the
                    # unchanged substrate as spurious [C]/[CH] radicals.
                    for key, rule_value in rc_attr.items():
                        if isinstance(rule_value, tuple) and len(rule_value) == 2:
                            host_value = host_attr.get(key)
                            host_left = (
                                host_value[0]
                                if isinstance(host_value, tuple)
                                and len(host_value) == 2
                                else host_value
                            )
                            host_attr[key] = (host_left, rule_value[1])
                        else:
                            host_attr[key] = rule_value
        its.graph["electron_aware_rewrite"] = electron_aware
        if electron_aware:
            its.graph["_product_electron_fields_current"] = False
            its.graph["_product_kekule_phase_dirty"] = (
                _product_state._product_kekule_phase_is_dirty(its)
            )
            if refresh_electrons:
                _product_state._refresh_product_electron_fields(its)
        list_its.append(its)
    return list_its


def _is_electron_aware_template(rc: nx.Graph) -> bool:
    """Return whether an RC carries paired Lewis-state rewrite data.

    Explicit hydrogen transfer edges may be folded into ``hcount`` by
    :class:`SynRule`.  Radical or lone-pair changes on the remaining nodes
    still require the electron-aware tuple rewrite path in that case.
    """
    edge_state = any(
        "sigma_order" in data and "pi_order" in data
        for _, _, data in rc.edges(data=True)
    )
    node_state = any(
        isinstance(data.get(key), tuple)
        and len(data[key]) == 2
        and data[key][0] != data[key][1]
        for _, data in rc.nodes(data=True)
        for key in ("radical", "lone_pairs", "valence_electrons")
    )
    return edge_state or node_state
