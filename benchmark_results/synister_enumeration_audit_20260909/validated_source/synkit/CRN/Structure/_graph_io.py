"""Conversion between a bipartite CRN digraph and the :class:`SynCRN` tables.

One direction collects species and reaction nodes out of a NetworkX digraph and
builds the canonical tables; the other rebuilds a digraph from those tables,
preserving the original node ids, node kinds and edge attributes so that a
round-trip is lossless.

Node ordering uses a natural (digit-aware) sort, so ``r_2`` precedes ``r_10``
and a graph round-trip does not permute a network's reactions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Hashable, List, Optional, Tuple
import re

import networkx as nx

from .reaction import RXNSide, Reaction
from .rule import Rule
from .species import Species
from ._parse import ID_STYLES

_DIGIT_RUN_RE = re.compile(r"(\d+)")


def _natural_chunks(text: str) -> Tuple[Tuple[int, int, str], ...]:
    """Split a string into comparable alternating text and integer chunks.

    Each chunk is normalized to a 3-tuple so that chunks of different kinds stay
    comparable: digit runs become ``(0, value, "")`` and text runs become
    ``(1, 0, text)``.

    :param text:
        String to split.
    :type text: str

    :return:
        Tuple of comparable chunks.
    :rtype: Tuple[Tuple[int, int, str], ...]
    """
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in _DIGIT_RUN_RE.split(text)
        if part != ""
    )


def _stable_sort_key(x: Any) -> Tuple[str, Tuple[Tuple[int, int, str], ...]]:
    """Return a stable, natural sort key for mixed node-id types.

    Ordering is by type name first, so heterogeneous node ids never compare
    across types, and then by a natural (digit-aware) reading of ``repr``. The
    digit awareness is what keeps ``r_2`` before ``r_10``; a plain lexicographic
    key would reorder a network's reactions on every graph round-trip.

    :param x:
        Any object that can be represented with ``repr``.
    :type x: Any

    :return:
        Tuple of type name and natural chunks of ``repr(x)``.
    :rtype: Tuple[str, Tuple[Tuple[int, int, str], ...]]

    .. rubric:: Example

    .. code-block:: python

        sorted(["r_10", "r_2"], key=_stable_sort_key)
        # ['r_2', 'r_10']
    """
    return (type(x).__name__, _natural_chunks(repr(x)))


def _edge_side(is_incoming: bool, role: Optional[str]) -> str:
    """Resolve the side of a reaction incidence edge.

    Explicit edge role takes priority. If no role is provided, graph direction
    is used:

    - ``species -> reaction`` means ``lhs``
    - ``reaction -> species`` means ``rhs``

    :param is_incoming:
        Whether the edge is incoming to the reaction node.
    :type is_incoming: bool

    :param role:
        Optional edge role, usually ``"reactant"`` or ``"product"``.
    :type role: Optional[str]

    :return:
        Either ``"lhs"`` or ``"rhs"``.
    :rtype: str

    .. rubric:: Example

    .. code-block:: python

        _edge_side(True, "reactant")
        # "lhs"
    """
    if role == "reactant":
        return "lhs"
    if role == "product":
        return "rhs"
    return "lhs" if is_incoming else "rhs"


def _collect_bipartite_nodes(
    crn: nx.DiGraph,
    *,
    species_kind: str,
    reaction_kinds: Tuple[str, ...],
) -> Tuple[List[Hashable], List[Hashable]]:
    """Collect species and reaction-like nodes from a bipartite CRN graph.

    :param crn:
        Directed bipartite CRN graph.
    :type crn: nx.DiGraph

    :param species_kind:
        Node-kind value identifying species nodes.
    :type species_kind: str

    :param reaction_kinds:
        Node-kind values identifying reaction or rule nodes.
    :type reaction_kinds: Tuple[str, ...]

    :return:
        Pair ``(species_nodes, reaction_nodes)`` in deterministic order.
    :rtype: Tuple[List[Hashable], List[Hashable]]

    .. rubric:: Example

    .. code-block:: python

        species_nodes, reaction_nodes = _collect_bipartite_nodes(
            crn,
            species_kind="species",
            reaction_kinds=("reaction", "rule"),
        )
    """
    reaction_kind_set = set(reaction_kinds)

    species_nodes: List[Hashable] = []
    reaction_nodes: List[Hashable] = []

    for node, attrs in crn.nodes(data=True):
        kind = attrs.get("kind")
        if kind == species_kind:
            species_nodes.append(node)
        elif kind in reaction_kind_set:
            reaction_nodes.append(node)

    return (
        sorted(species_nodes, key=_stable_sort_key),
        sorted(reaction_nodes, key=_stable_sort_key),
    )


def _validate_bipartite_node_sets(
    *,
    species_nodes: List[Hashable],
    reaction_nodes: List[Hashable],
    strict: bool,
) -> None:
    """Validate that a bipartite CRN graph contains required node classes.

    :param species_nodes:
        Collected species nodes.
    :type species_nodes: List[Hashable]

    :param reaction_nodes:
        Collected reaction-like nodes.
    :type reaction_nodes: List[Hashable]

    :param strict:
        Whether missing classes should raise an error.
    :type strict: bool

    :return:
        None.
    :rtype: None
    """
    if strict and not species_nodes:
        raise ValueError("No species nodes found in graph")
    if strict and not reaction_nodes:
        raise ValueError("No reaction/rule nodes found in graph")


def _make_internal_id_maps(
    *,
    species_nodes: List[Hashable],
    reaction_nodes: List[Hashable],
    species_prefix: str,
    reaction_prefix: str,
    id_style: str = "prefixed",
) -> Tuple[Dict[Hashable, str], Dict[Hashable, str]]:
    """Build internal id maps for species and reaction nodes.

    :param species_nodes:
        Ordered species nodes.
    :type species_nodes: List[Hashable]

    :param reaction_nodes:
        Ordered reaction-like nodes.
    :type reaction_nodes: List[Hashable]

    :param species_prefix:
        Prefix for generated species ids.
    :type species_prefix: str

    :param reaction_prefix:
        Prefix for generated reaction ids.
    :type reaction_prefix: str

    :param id_style:
        Either ``"prefixed"`` or the legacy shared-namespace ``"numeric"``.
    :type id_style: str

    :return:
        Pair ``(species_node_to_id, reaction_node_to_id)``.
    :rtype: Tuple[Dict[Hashable, str], Dict[Hashable, str]]

    :raises ValueError:
        If ``id_style`` is not a recognised policy.
    """
    if id_style not in ID_STYLES:
        raise ValueError(
            f"id_style must be one of {sorted(ID_STYLES)}, got {id_style!r}"
        )

    if id_style == "numeric":
        offset = len(species_nodes)
        species_node_to_id = {
            node: str(i) for i, node in enumerate(species_nodes, start=1)
        }
        reaction_node_to_id = {
            node: str(offset + i)
            for i, node in enumerate(reaction_nodes, start=1)
        }
        return species_node_to_id, reaction_node_to_id

    species_node_to_id = {
        node: f"{species_prefix}{i}" for i, node in enumerate(species_nodes, start=1)
    }
    reaction_node_to_id = {
        node: f"{reaction_prefix}{i}" for i, node in enumerate(reaction_nodes, start=1)
    }
    return species_node_to_id, reaction_node_to_id


def _build_species_table_from_graph(
    crn: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    species_node_to_id: Dict[Hashable, str],
) -> Dict[str, Species]:
    """Build the canonical species table from graph species nodes.

    :param crn:
        Source CRN graph.
    :type crn: nx.DiGraph

    :param species_nodes:
        Ordered species nodes.
    :type species_nodes: List[Hashable]

    :param species_node_to_id:
        Mapping from source node id to internal species id.
    :type species_node_to_id: Dict[Hashable, str]

    :return:
        Species table keyed by internal species id.
    :rtype: Dict[str, Species]
    """
    species: Dict[str, Species] = {}
    for node in species_nodes:
        attrs = dict(crn.nodes[node])
        sid = species_node_to_id[node]
        species[sid] = Species(
            id=sid,
            source_node_id=node,
            label=str(attrs.get("label", sid)),
            smiles=attrs.get("smiles"),
            source_attrs=dict(attrs),
            metadata={},
        )
    return species


def _collect_reaction_sides_from_graph(
    crn: nx.DiGraph,
    *,
    rnode: Hashable,
    species_node_to_id: Dict[Hashable, str],
    strict: bool,
) -> Tuple[RXNSide, RXNSide, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Collect reactant and product sides for one reaction node.

    :param crn:
        Source CRN graph.
    :type crn: nx.DiGraph

    :param rnode:
        Reaction node in the source graph.
    :type rnode: Hashable

    :param species_node_to_id:
        Mapping from source species node id to canonical species id.
    :type species_node_to_id: Dict[Hashable, str]

    :param strict:
        Whether malformed structure should raise an error.
    :type strict: bool

    :return:
        Tuple ``(lhs, rhs, reactant_edge_attrs, product_edge_attrs)``.
    :rtype: Tuple[RXNSide, RXNSide, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]
    """
    lhs_counter: Counter[str] = Counter()
    rhs_counter: Counter[str] = Counter()
    reactant_edge_attrs: Dict[str, Dict[str, Any]] = {}
    product_edge_attrs: Dict[str, Dict[str, Any]] = {}

    for u, _, eattrs in crn.in_edges(rnode, data=True):
        if u not in species_node_to_id:
            if strict:
                raise ValueError(
                    f"Incoming edge into reaction node {rnode!r} must come from "
                    f"a species node, got {u!r}"
                )
            continue

        sid = species_node_to_id[u]
        attrs = dict(eattrs)
        stoich = int(attrs.get("stoich", 1))
        if stoich <= 0:
            raise ValueError(f"Invalid stoich={stoich} on edge ({u!r}, {rnode!r})")

        side = _edge_side(True, attrs.get("role"))
        if side == "lhs":
            lhs_counter[sid] += stoich
            reactant_edge_attrs[sid] = attrs
        else:
            rhs_counter[sid] += stoich
            product_edge_attrs[sid] = attrs

    for _, v, eattrs in crn.out_edges(rnode, data=True):
        if v not in species_node_to_id:
            if strict:
                raise ValueError(
                    f"Outgoing edge from reaction node {rnode!r} must go to "
                    f"a species node, got {v!r}"
                )
            continue

        sid = species_node_to_id[v]
        attrs = dict(eattrs)
        stoich = int(attrs.get("stoich", 1))
        if stoich <= 0:
            raise ValueError(f"Invalid stoich={stoich} on edge ({rnode!r}, {v!r})")

        side = _edge_side(False, attrs.get("role"))
        if side == "lhs":
            lhs_counter[sid] += stoich
            reactant_edge_attrs[sid] = attrs
        else:
            rhs_counter[sid] += stoich
            product_edge_attrs[sid] = attrs

    lhs = RXNSide(dict(lhs_counter))
    rhs = RXNSide(dict(rhs_counter))
    return lhs, rhs, reactant_edge_attrs, product_edge_attrs


def _validate_nonempty_reaction_sides(
    *,
    rnode: Hashable,
    lhs: RXNSide,
    rhs: RXNSide,
    strict: bool,
) -> None:
    """Validate that a reaction has non-empty lhs and rhs sides when strict mode is enabled.

    :param rnode:
        Source reaction node id.
    :type rnode: Hashable

    :param lhs:
        Reactant side.
    :type lhs: RXNSide

    :param rhs:
        Product side.
    :type rhs: RXNSide

    :param strict:
        Whether empty sides should raise an error.
    :type strict: bool

    :return:
        None.
    :rtype: None
    """
    if strict and not lhs:
        raise ValueError(f"Reaction node {rnode!r} has empty reactant side")
    if strict and not rhs:
        raise ValueError(f"Reaction node {rnode!r} has empty product side")


def _get_or_create_rule_from_attrs(
    *,
    rattrs: Dict[str, Any],
    rules: Dict[str, Rule],
    rule_key_to_id: Dict[Tuple[Optional[int], Optional[str]], str],
    rule_prefix: str,
) -> Optional[str]:
    """Get or create a canonical rule entry from reaction-node attributes.

    :param rattrs:
        Reaction-node attribute dictionary.
    :type rattrs: Dict[str, Any]

    :param rules:
        Mutable rule table.
    :type rules: Dict[str, Rule]

    :param rule_key_to_id:
        Mapping from abstract rule key to canonical rule id.
    :type rule_key_to_id: Dict[Tuple[Optional[int], Optional[str]], str]

    :param rule_prefix:
        Prefix used when creating new rule ids.
    :type rule_prefix: str

    :return:
        Canonical rule id if a rule is defined, else ``None``.
    :rtype: Optional[str]
    """
    rule_key = (rattrs.get("rule_index"), rattrs.get("rule_repr"))
    if rule_key == (None, None):
        return None

    if rule_key not in rule_key_to_id:
        rule_id = f"{rule_prefix}{len(rule_key_to_id) + 1}"
        rule_key_to_id[rule_key] = rule_id
        rules[rule_id] = Rule(
            id=rule_id,
            rule_index=rattrs.get("rule_index"),
            rule_repr=rattrs.get("rule_repr"),
            label=(
                f"rule[{rattrs.get('rule_index')}]"
                if rattrs.get("rule_index") is not None
                else None
            ),
        )

    return rule_key_to_id[rule_key]


def _build_reaction_from_graph_node(
    crn: nx.DiGraph,
    *,
    rnode: Hashable,
    rid: str,
    species_node_to_id: Dict[Hashable, str],
    rules: Dict[str, Rule],
    rule_key_to_id: Dict[Tuple[Optional[int], Optional[str]], str],
    rule_prefix: str,
    strict: bool,
) -> Reaction:
    """Build one canonical Reaction object from a graph reaction node.

    :param crn:
        Source CRN graph.
    :type crn: nx.DiGraph

    :param rnode:
        Reaction node id in the source graph.
    :type rnode: Hashable

    :param rid:
        Canonical reaction id.
    :type rid: str

    :param species_node_to_id:
        Mapping from source species node ids to canonical species ids.
    :type species_node_to_id: Dict[Hashable, str]

    :param rules:
        Mutable rule table.
    :type rules: Dict[str, Rule]

    :param rule_key_to_id:
        Mapping from abstract rule key to canonical rule id.
    :type rule_key_to_id: Dict[Tuple[Optional[int], Optional[str]], str]

    :param rule_prefix:
        Prefix for rule ids.
    :type rule_prefix: str

    :param strict:
        Whether malformed structure should raise an error.
    :type strict: bool

    :return:
        Canonical reaction object.
    :rtype: Reaction
    """
    rattrs = dict(crn.nodes[rnode])

    lhs, rhs, reactant_edge_attrs, product_edge_attrs = (
        _collect_reaction_sides_from_graph(
            crn,
            rnode=rnode,
            species_node_to_id=species_node_to_id,
            strict=strict,
        )
    )
    _validate_nonempty_reaction_sides(rnode=rnode, lhs=lhs, rhs=rhs, strict=strict)

    rule_id = _get_or_create_rule_from_attrs(
        rattrs=rattrs,
        rules=rules,
        rule_key_to_id=rule_key_to_id,
        rule_prefix=rule_prefix,
    )

    return Reaction(
        id=rid,
        source_node_id=rnode,
        source_kind=str(rattrs.get("kind", "reaction")),
        lhs=lhs,
        rhs=rhs,
        label=rattrs.get("label", rid),
        step=rattrs.get("step"),
        rule_index=rattrs.get("rule_index"),
        app_index=rattrs.get("app_index"),
        rule_repr=rattrs.get("rule_repr"),
        rule_id=rule_id,
        source_attrs=dict(rattrs),
        metadata={},
        reactant_edge_attrs=reactant_edge_attrs,
        product_edge_attrs=product_edge_attrs,
    )


def _resolve_species_node_id(sp: Species, *, node_ids: str) -> Hashable:
    """Resolve the node id for one species during graph reconstruction.

    :param sp:
        Species record.
    :type sp: Species

    :param node_ids:
        Either ``"source"`` or ``"internal"``.
    :type node_ids: str

    :return:
        Reconstructed graph node id.
    :rtype: Hashable
    """
    return sp.source_node_id if node_ids == "source" else sp.id


def _resolve_reaction_node_id(rxn: Reaction, *, node_ids: str) -> Hashable:
    """Resolve the node id for one reaction during graph reconstruction.

    :param rxn:
        Reaction record.
    :type rxn: Reaction

    :param node_ids:
        Either ``"source"`` or ``"internal"``.
    :type node_ids: str

    :return:
        Reconstructed graph node id.
    :rtype: Hashable
    """
    return rxn.source_node_id if node_ids == "source" else rxn.id


def _species_node_attrs(
    sp: Species,
    *,
    include_internal_ids: bool,
) -> Dict[str, Any]:
    """Build species-node attributes for graph reconstruction.

    :param sp:
        Species record.
    :type sp: Species

    :param include_internal_ids:
        Whether canonical ids should be attached as attributes.
    :type include_internal_ids: bool

    :return:
        Node-attribute dictionary.
    :rtype: Dict[str, Any]
    """
    attrs = dict(sp.source_attrs)
    attrs["kind"] = "species"
    attrs["label"] = sp.label
    attrs["smiles"] = sp.smiles
    if include_internal_ids:
        attrs["syncrn_id"] = sp.id
        attrs["source_node_id"] = sp.source_node_id
    return attrs


def _reaction_node_attrs(
    rxn: Reaction,
    *,
    reaction_kind: Optional[str],
    include_internal_ids: bool,
) -> Dict[str, Any]:
    """Build reaction-node attributes for graph reconstruction.

    :param rxn:
        Reaction record.
    :type rxn: Reaction

    :param reaction_kind:
        Optional override for reconstructed reaction kind.
    :type reaction_kind: Optional[str]

    :param include_internal_ids:
        Whether canonical ids should be attached as attributes.
    :type include_internal_ids: bool

    :return:
        Node-attribute dictionary.
    :rtype: Dict[str, Any]
    """
    attrs = dict(rxn.source_attrs)
    attrs["kind"] = reaction_kind or rxn.source_kind
    attrs["label"] = rxn.label
    attrs["step"] = rxn.step
    attrs["rule_index"] = rxn.rule_index
    attrs["app_index"] = rxn.app_index
    attrs["rule_repr"] = rxn.rule_repr
    if rxn.rule_id is not None:
        attrs.setdefault("rule_id", rxn.rule_id)
    if include_internal_ids:
        attrs["syncrn_id"] = rxn.id
        attrs["source_node_id"] = rxn.source_node_id
    return attrs


def _add_reaction_edges_to_graph(
    g: nx.DiGraph,
    *,
    rxn: Reaction,
    reaction_node: Hashable,
    species_node_map: Dict[str, Hashable],
) -> None:
    """Add reactant and product incidence edges for one reaction.

    :param g:
        Graph under construction.
    :type g: nx.DiGraph

    :param rxn:
        Reaction record.
    :type rxn: Reaction

    :param reaction_node:
        Graph node id of the reconstructed reaction node.
    :type reaction_node: Hashable

    :param species_node_map:
        Mapping from canonical species ids to graph node ids.
    :type species_node_map: Dict[str, Hashable]

    :return:
        None.
    :rtype: None
    """
    for sid, coeff in rxn.lhs.items():
        snode = species_node_map[sid]
        attrs = dict(rxn.reactant_edge_attrs.get(sid, {}))
        attrs["role"] = "reactant"
        attrs["stoich"] = coeff
        if rxn.step is not None:
            attrs.setdefault("step", rxn.step)
        if rxn.rule_index is not None:
            attrs.setdefault("rule_index", rxn.rule_index)
        g.add_edge(snode, reaction_node, **attrs)

    for sid, coeff in rxn.rhs.items():
        snode = species_node_map[sid]
        attrs = dict(rxn.product_edge_attrs.get(sid, {}))
        attrs["role"] = "product"
        attrs["stoich"] = coeff
        if rxn.step is not None:
            attrs.setdefault("step", rxn.step)
        if rxn.rule_index is not None:
            attrs.setdefault("rule_index", rxn.rule_index)
        g.add_edge(reaction_node, snode, **attrs)
