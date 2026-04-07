from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple
import re

import networkx as nx

from .reaction import RXNSide, Reaction
from .rule import Rule
from .species import Species

_REACTION_SPLIT_RE = re.compile(r"\s*>>\s*")


def _default_parse_side_text(side_text: str) -> Dict[str, int]:
    """
    Parse one reaction side into a ``species -> coefficient`` mapping.

    Supported examples include plain species lists, integer stoichiometric
    prefixes, and dot-separated species tokens.

    Supported examples
    ------------------
    - ``A + B``
    - ``2A + B``
    - ``A.A``
    - ``B.3C``
    - ``∅``

    :param side_text:
        One side of a reaction string.
    :type side_text: str

    :returns:
        Mapping from species label to stoichiometric coefficient.
    :rtype: Dict[str, int]

    Example
    -------
    .. code-block:: python

        _default_parse_side_text("2A + B")
        # {"A": 2, "B": 1}
    """
    text = str(side_text).strip()
    if text in {"", "∅", "0", "None", "null"}:
        return {}

    parts = re.split(r"\s*(?:\+|\.)\s*", text)
    out: Dict[str, int] = {}

    for part in parts:
        token = part.strip()
        if not token:
            continue

        m = re.match(r"^\s*(\d+)\s*\*?\s*(.+?)\s*$", token)
        if m:
            coeff = int(m.group(1))
            species = m.group(2).strip()
        else:
            m2 = re.match(r"^\s*(\d+)([A-Za-z].*?)\s*$", token)
            if m2:
                coeff = int(m2.group(1))
                species = m2.group(2).strip()
            else:
                coeff = 1
                species = token

        if species and coeff > 0:
            out[species] = out.get(species, 0) + coeff

    return out


def _coerce_side_counts(obj: Any) -> Dict[str, int]:
    """
    Coerce a parsed side object into a plain dictionary.

    This helper accepts plain mappings, objects exposing ``to_dict()``, and
    RXNSide-like objects exposing ``items()``.

    :param obj:
        Parsed side object.
    :type obj: Any

    :returns:
        Mapping from species label to stoichiometric coefficient.
    :rtype: Dict[str, int]

    :raises TypeError:
        If the object cannot be interpreted as a side-count mapping.

    Example
    -------
    .. code-block:: python

        _coerce_side_counts({"A": 2, "B": 1})
        # {"A": 2, "B": 1}
    """
    if obj is None:
        return {}

    if isinstance(obj, Mapping):
        return {str(k): int(v) for k, v in obj.items()}

    if hasattr(obj, "to_dict"):
        d = obj.to_dict()
        return {str(k): int(v) for k, v in d.items()}

    if hasattr(obj, "items"):
        return {str(k): int(v) for k, v in obj.items()}

    raise TypeError(
        "Parsed side must be a mapping, or expose to_dict(), or expose items()."
    )


def _stable_sort_key(x: Any) -> Tuple[str, str]:
    """
    Return a stable sort key for mixed node-id types.

    :param x:
        Any object that can be represented with ``repr``.
    :type x: Any

    :returns:
        Tuple based on type name and repr.
    :rtype: Tuple[str, str]

    Example
    -------
    .. code-block:: python

        sorted(nodes, key=_stable_sort_key)
    """
    return (type(x).__name__, repr(x))


def _edge_side(is_incoming: bool, role: Optional[str]) -> str:
    """
    Resolve the side of a reaction incidence edge.

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

    :returns:
        Either ``"lhs"`` or ``"rhs"``.
    :rtype: str

    Example
    -------
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
    """
    Collect species and reaction-like nodes from a bipartite CRN graph.

    :param crn:
        Directed bipartite CRN graph.
    :type crn: nx.DiGraph

    :param species_kind:
        Node-kind value identifying species nodes.
    :type species_kind: str

    :param reaction_kinds:
        Node-kind values identifying reaction or rule nodes.
    :type reaction_kinds: Tuple[str, ...]

    :returns:
        Pair ``(species_nodes, reaction_nodes)`` in deterministic order.
    :rtype: Tuple[List[Hashable], List[Hashable]]

    Example
    -------
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
    """
    Validate that a bipartite CRN graph contains required node classes.

    :param species_nodes:
        Collected species nodes.
    :type species_nodes: List[Hashable]

    :param reaction_nodes:
        Collected reaction-like nodes.
    :type reaction_nodes: List[Hashable]

    :param strict:
        Whether missing classes should raise an error.
    :type strict: bool

    :returns:
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
) -> Tuple[Dict[Hashable, str], Dict[Hashable, str]]:
    """
    Build internal id maps for species and reaction nodes.

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

    :returns:
        Pair ``(species_node_to_id, reaction_node_to_id)``.
    :rtype: Tuple[Dict[Hashable, str], Dict[Hashable, str]]
    """
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
    """
    Build the canonical species table from graph species nodes.

    :param crn:
        Source CRN graph.
    :type crn: nx.DiGraph

    :param species_nodes:
        Ordered species nodes.
    :type species_nodes: List[Hashable]

    :param species_node_to_id:
        Mapping from source node id to internal species id.
    :type species_node_to_id: Dict[Hashable, str]

    :returns:
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
    """
    Collect reactant and product sides for one reaction node.

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

    :returns:
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
    """
    Validate that a reaction has non-empty lhs and rhs sides when strict mode is enabled.

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

    :returns:
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
    """
    Get or create a canonical rule entry from reaction-node attributes.

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

    :returns:
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
    """
    Build one canonical Reaction object from a graph reaction node.

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

    :returns:
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


def _normalize_side_counts(
    counts: Any,
    *,
    rxn_index: int,
    side_name: str,
    strict: bool,
) -> Dict[str, int]:
    """
    Normalize a parsed reaction side into a validated label-to-coefficient mapping.

    :param counts:
        Parsed side object.
    :type counts: Any

    :param rxn_index:
        Reaction index in the input list.
    :type rxn_index: int

    :param side_name:
        Name of the side, typically ``"lhs"`` or ``"rhs"``.
    :type side_name: str

    :param strict:
        Whether invalid labels or coefficients should raise an error.
    :type strict: bool

    :returns:
        Cleaned mapping from species label to coefficient.
    :rtype: Dict[str, int]
    """
    raw = _coerce_side_counts(counts)
    out: Dict[str, int] = {}

    for sp, coeff in raw.items():
        label = str(sp).strip()
        try:
            n = int(coeff)
        except Exception as exc:
            raise TypeError(
                f"Reaction at index {rxn_index} has non-integer coefficient on "
                f"{side_name}: {sp!r} -> {coeff!r}"
            ) from exc

        if not label:
            if strict:
                raise ValueError(
                    f"Reaction at index {rxn_index} has blank species label on {side_name}"
                )
            continue

        if n <= 0:
            if strict:
                raise ValueError(
                    f"Reaction at index {rxn_index} has non-positive coefficient on "
                    f"{side_name}: {sp!r} -> {n}"
                )
            continue

        out[label] = out.get(label, 0) + n

    return out


def _parse_reaction_string_entry(
    rxn_text: str,
    *,
    rxn_index: int,
    rule_repr: Optional[str],
    has_rules: bool,
    parse_side: Callable[[str], Any],
    strict: bool,
) -> Dict[str, Any]:
    """
    Parse one reaction string into normalized lhs/rhs count dictionaries.

    :param rxn_text:
        Raw reaction string such as ``"2A>>B+3C"``.
    :type rxn_text: str

    :param rxn_index:
        Index of the reaction in the input list.
    :type rxn_index: int

    :param rule_repr:
        Optional rule string paired with the reaction.
    :type rule_repr: Optional[str]

    :param has_rules:
        Whether pairwise rules are being used.
    :type has_rules: bool

    :param parse_side:
        Side parser function.
    :type parse_side: Callable[[str], Any]

    :param strict:
        Whether malformed input should raise an error.
    :type strict: bool

    :returns:
        Parsed reaction record.
    :rtype: Dict[str, Any]
    """
    text = str(rxn_text).strip()
    pieces = _REACTION_SPLIT_RE.split(text, maxsplit=1)

    if len(pieces) != 2:
        raise ValueError(
            f"Reaction string at index {rxn_index} must contain exactly one '>>': {rxn_text!r}"
        )

    lhs_text, rhs_text = pieces[0].strip(), pieces[1].strip()

    lhs_counts = _normalize_side_counts(
        parse_side(lhs_text),
        rxn_index=rxn_index,
        side_name="lhs",
        strict=strict,
    )
    rhs_counts = _normalize_side_counts(
        parse_side(rhs_text),
        rxn_index=rxn_index,
        side_name="rhs",
        strict=strict,
    )

    if strict and not lhs_counts:
        raise ValueError(f"Reaction at index {rxn_index} has empty lhs: {rxn_text!r}")
    if strict and not rhs_counts:
        raise ValueError(f"Reaction at index {rxn_index} has empty rhs: {rxn_text!r}")

    return {
        "rxn_text": text,
        "lhs_counts": lhs_counts,
        "rhs_counts": rhs_counts,
        "rule_repr": rule_repr,
        "rule_index": rxn_index if has_rules else None,
    }


def _species_order_from_parsed_reactions(
    parsed_rxns: List[Dict[str, Any]],
) -> List[str]:
    """
    Derive species order by first appearance in parsed reactions.

    :param parsed_rxns:
        Parsed reaction entries.
    :type parsed_rxns: List[Dict[str, Any]]

    :returns:
        Ordered species labels.
    :rtype: List[str]
    """
    order: List[str] = []
    seen = set()

    for item in parsed_rxns:
        for sp in list(item["lhs_counts"].keys()) + list(item["rhs_counts"].keys()):
            if sp not in seen:
                seen.add(sp)
                order.append(sp)

    return order


def _build_species_table_from_labels(species_order: List[str]) -> Dict[str, Species]:
    """
    Build the canonical species table from an ordered list of species labels.

    :param species_order:
        Species labels in canonical order.
    :type species_order: List[str]

    :returns:
        Species table keyed by canonical string ids.
    :rtype: Dict[str, Species]
    """
    species: Dict[str, Species] = {}
    for idx, label in enumerate(species_order, start=1):
        sid = str(idx)
        species[sid] = Species(
            id=sid,
            source_node_id=idx,
            label=label,
            smiles=None,
            source_attrs={
                "kind": "species",
                "label": label,
            },
            metadata={},
        )
    return species


def _build_rules_table_from_reaction_strings(
    parsed_rxns: List[Dict[str, Any]],
    *,
    has_rules: bool,
) -> Dict[str, Rule]:
    """
    Build the abstract rules table for reaction-string input.

    :param parsed_rxns:
        Parsed reaction entries.
    :type parsed_rxns: List[Dict[str, Any]]

    :param has_rules:
        Whether rules were supplied pairwise.
    :type has_rules: bool

    :returns:
        Rule table keyed by canonical rule ids.
    :rtype: Dict[str, Rule]
    """
    if not has_rules:
        return {}

    rules_table: Dict[str, Rule] = {}
    for i, item in enumerate(parsed_rxns):
        rule_id = str(i + 1)
        rule_index = item["rule_index"]
        rule_repr = item["rule_repr"]
        rules_table[rule_id] = Rule(
            id=rule_id,
            rule_index=rule_index,
            rule_repr=rule_repr,
            label=f"r{rule_index}",
            metadata={},
        )
    return rules_table


def _build_reactions_from_parsed_strings(
    parsed_rxns: List[Dict[str, Any]],
    *,
    label_to_sid: Dict[str, str],
    reaction_start_index: int,
    has_rules: bool,
) -> Dict[str, Reaction]:
    """
    Build canonical Reaction objects from parsed reaction-string entries.

    :param parsed_rxns:
        Parsed reaction entries.
    :type parsed_rxns: List[Dict[str, Any]]

    :param label_to_sid:
        Mapping from species label to canonical species id.
    :type label_to_sid: Dict[str, str]

    :param reaction_start_index:
        First numeric reaction id.
    :type reaction_start_index: int

    :param has_rules:
        Whether rules were supplied pairwise.
    :type has_rules: bool

    :returns:
        Reaction table keyed by canonical reaction ids.
    :rtype: Dict[str, Reaction]
    """
    reactions: Dict[str, Reaction] = {}

    for i, item in enumerate(parsed_rxns):
        rid_int = reaction_start_index + i
        rid = str(rid_int)

        rule_id: Optional[str] = None
        rule_index = item["rule_index"]
        rule_repr = item["rule_repr"]

        if has_rules:
            rule_id = str(i + 1)

        lhs = RXNSide(
            {label_to_sid[sp]: coeff for sp, coeff in item["lhs_counts"].items()}
        )
        rhs = RXNSide(
            {label_to_sid[sp]: coeff for sp, coeff in item["rhs_counts"].items()}
        )

        reactions[rid] = Reaction(
            id=rid,
            source_node_id=rid_int,
            source_kind="rule",
            lhs=lhs,
            rhs=rhs,
            label=rule_repr if has_rules and rule_repr is not None else rid,
            step=None,
            rule_index=rule_index,
            app_index=None,
            rule_repr=rule_repr,
            rule_id=rule_id,
            source_attrs={
                "kind": "rule",
                "label": rid,
                "rxn_repr": item["rxn_text"],
                "rule_index": rule_index,
                "rule_repr": rule_repr,
            },
            metadata={},
            reactant_edge_attrs={
                label_to_sid[sp]: {"role": "reactant", "stoich": coeff}
                for sp, coeff in item["lhs_counts"].items()
            },
            product_edge_attrs={
                label_to_sid[sp]: {"role": "product", "stoich": coeff}
                for sp, coeff in item["rhs_counts"].items()
            },
        )

    return reactions


def _resolve_species_node_id(sp: Species, *, node_ids: str) -> Hashable:
    """
    Resolve the node id for one species during graph reconstruction.

    :param sp:
        Species record.
    :type sp: Species

    :param node_ids:
        Either ``"source"`` or ``"internal"``.
    :type node_ids: str

    :returns:
        Reconstructed graph node id.
    :rtype: Hashable
    """
    return sp.source_node_id if node_ids == "source" else sp.id


def _resolve_reaction_node_id(rxn: Reaction, *, node_ids: str) -> Hashable:
    """
    Resolve the node id for one reaction during graph reconstruction.

    :param rxn:
        Reaction record.
    :type rxn: Reaction

    :param node_ids:
        Either ``"source"`` or ``"internal"``.
    :type node_ids: str

    :returns:
        Reconstructed graph node id.
    :rtype: Hashable
    """
    return rxn.source_node_id if node_ids == "source" else rxn.id


def _species_node_attrs(
    sp: Species,
    *,
    include_internal_ids: bool,
) -> Dict[str, Any]:
    """
    Build species-node attributes for graph reconstruction.

    :param sp:
        Species record.
    :type sp: Species

    :param include_internal_ids:
        Whether canonical ids should be attached as attributes.
    :type include_internal_ids: bool

    :returns:
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
    """
    Build reaction-node attributes for graph reconstruction.

    :param rxn:
        Reaction record.
    :type rxn: Reaction

    :param reaction_kind:
        Optional override for reconstructed reaction kind.
    :type reaction_kind: Optional[str]

    :param include_internal_ids:
        Whether canonical ids should be attached as attributes.
    :type include_internal_ids: bool

    :returns:
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
    """
    Add reactant and product incidence edges for one reaction.

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

    :returns:
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


@dataclass
class SynCRN:
    """
    Canonical reaction-system object for SynKit-CRN.

    Official representations exposed by this object
    ------------------------------------------------
    - ``SynCRN``: master reaction-system object
    - ``SynCRN.to_digraph()``: species--reaction bipartite graph view
    - ``SynCRN.to_stoichiometric_matrices()``: matrix view for stoichiometric analysis
    - ``SynCRN.to_petrinet()``: pre/post incidence view for pathways
    - ``SynCRN.to_equations()``: human-readable reaction list

    Design notes
    ------------
    Input digraph nodes with ``kind="rule"`` are preserved exactly on round-trip
    via ``Reaction.source_kind`` and ``Reaction.source_attrs``. They are also
    normalized as concrete reaction instances for downstream computation.

    :param species:
        Mapping from internal species ids to Species records.
    :type species: Dict[str, Species]

    :param reactions:
        Mapping from internal reaction ids to Reaction records.
    :type reactions: Dict[str, Reaction]

    :param rules:
        Mapping from internal rule ids to Rule records.
    :type rules: Dict[str, Rule]

    :param graph_attrs:
        Original graph-level attributes from the input digraph.
    :type graph_attrs: Dict[str, Any]

    :param metadata:
        Additional canonical metadata for the SynCRN object.
    :type metadata: Dict[str, Any]

    Example
    -------
    .. code-block:: python

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        print(syn.n_species)
        print(syn.to_equations())
    """

    species: Dict[str, Species] = field(default_factory=dict)
    reactions: Dict[str, Reaction] = field(default_factory=dict)
    rules: Dict[str, Rule] = field(default_factory=dict)
    graph_attrs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_digraph(
        cls,
        crn: nx.DiGraph,
        *,
        species_kind: str = "species",
        reaction_kinds: Tuple[str, ...] = ("reaction", "rule"),
        species_prefix: str = "s_",
        reaction_prefix: str = "r_",
        rule_prefix: str = "rule_",
        strict: bool = True,
    ) -> "SynCRN":
        """
        Build a canonical SynCRN object from a species--reaction bipartite digraph.

        The current SynKit-CRN graph frequently stores concrete reaction-instance
        nodes with ``kind="rule"``. This constructor preserves that original kind
        in ``Reaction.source_kind`` while also converting such nodes into concrete
        reaction instances for downstream computation.

        :param crn:
            Directed bipartite graph with species and reaction-like nodes.
        :type crn: nx.DiGraph

        :param species_kind:
            Node-kind value used for species nodes.
        :type species_kind: str

        :param reaction_kinds:
            Node-kind values used for reaction-instance nodes.
        :type reaction_kinds: Tuple[str, ...]

        :param species_prefix:
            Prefix for generated internal species ids.
        :type species_prefix: str

        :param reaction_prefix:
            Prefix for generated internal reaction ids.
        :type reaction_prefix: str

        :param rule_prefix:
            Prefix for generated internal rule ids.
        :type rule_prefix: str

        :param strict:
            Whether malformed graph structure should raise an error.
        :type strict: bool

        :returns:
            Canonical SynCRN object.
        :rtype: SynCRN

        :raises TypeError:
            If ``crn`` is not an ``nx.DiGraph``.

        Example
        -------
        .. code-block:: python

            syn = SynCRN.from_digraph(crn)
            print(syn.to_equations())
        """
        if not isinstance(crn, nx.DiGraph):
            raise TypeError(f"crn must be nx.DiGraph, got {type(crn).__name__}")

        species_nodes, reaction_nodes = _collect_bipartite_nodes(
            crn,
            species_kind=species_kind,
            reaction_kinds=reaction_kinds,
        )
        _validate_bipartite_node_sets(
            species_nodes=species_nodes,
            reaction_nodes=reaction_nodes,
            strict=strict,
        )

        species_node_to_id, reaction_node_to_id = _make_internal_id_maps(
            species_nodes=species_nodes,
            reaction_nodes=reaction_nodes,
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
        )

        species = _build_species_table_from_graph(
            crn,
            species_nodes=species_nodes,
            species_node_to_id=species_node_to_id,
        )

        rules: Dict[str, Rule] = {}
        rule_key_to_id: Dict[Tuple[Optional[int], Optional[str]], str] = {}
        reactions: Dict[str, Reaction] = {}

        for rnode in reaction_nodes:
            rid = reaction_node_to_id[rnode]
            reactions[rid] = _build_reaction_from_graph_node(
                crn,
                rnode=rnode,
                rid=rid,
                species_node_to_id=species_node_to_id,
                rules=rules,
                rule_key_to_id=rule_key_to_id,
                rule_prefix=rule_prefix,
                strict=strict,
            )

        return cls(
            species=species,
            reactions=reactions,
            rules=rules,
            graph_attrs=dict(crn.graph),
            metadata={"source_graph_type": type(crn).__name__},
        )

    @property
    def species_ids(self) -> List[str]:
        """
        Return the internal species order.

        :returns:
            Ordered list of species ids.
        :rtype: List[str]

        Example
        -------
        .. code-block:: python

            print(syn.species_ids)
        """
        return list(self.species.keys())

    @property
    def reaction_ids(self) -> List[str]:
        """
        Return the internal reaction order.

        :returns:
            Ordered list of reaction ids.
        :rtype: List[str]

        Example
        -------
        .. code-block:: python

            print(syn.reaction_ids)
        """
        return list(self.reactions.keys())

    @property
    def rule_ids(self) -> List[str]:
        """
        Return the internal rule order.

        :returns:
            Ordered list of rule ids.
        :rtype: List[str]

        Example
        -------
        .. code-block:: python

            print(syn.rule_ids)
        """
        return list(self.rules.keys())

    @property
    def n_species(self) -> int:
        """
        Return the number of species.

        :returns:
            Number of species.
        :rtype: int
        """
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        """
        Return the number of reactions.

        :returns:
            Number of reactions.
        :rtype: int
        """
        return len(self.reactions)

    @property
    def n_rules(self) -> int:
        """
        Return the number of unique abstract rules.

        :returns:
            Number of rules.
        :rtype: int
        """
        return len(self.rules)

    def __repr__(self) -> str:
        """
        Return a compact developer-facing representation.

        :returns:
            Summary representation string.
        :rtype: str
        """
        return (
            f"SynCRN(n_species={self.n_species}, "
            f"n_reactions={self.n_reactions}, n_rules={self.n_rules})"
        )

    def __str__(self) -> str:
        """
        Return a human-readable text summary.

        :returns:
            Multiline description string.
        :rtype: str
        """
        return self.describe(include_species=True, species="label")

    def _species_token(self, species_id: str, mode: str = "label") -> str:
        """
        Resolve how a species should be displayed.

        Supported modes are ``"id"``, ``"label"``, ``"smiles"``, and ``"source"``.

        :param species_id:
            Internal species id.
        :type species_id: str

        :param mode:
            Species display mode.
        :type mode: str

        :returns:
            Display token for the species.
        :rtype: str

        :raises ValueError:
            If the display mode is unsupported.

        Example
        -------
        .. code-block:: python

            syn._species_token("1", mode="label")
        """
        sp = self.species[species_id]
        if mode == "id":
            return sp.id
        if mode == "label":
            return sp.label
        if mode == "smiles":
            return sp.smiles or sp.label
        if mode == "source":
            return str(sp.source_node_id)
        raise ValueError("species mode must be one of: id, label, smiles, source")

    def format_reaction(
        self,
        reaction_id: str,
        *,
        species: str = "label",
        include_id: bool = True,
        include_rule: bool = False,
        include_step: bool = False,
        arrow: str = ">>",
    ) -> str:
        """
        Format one reaction as text.

        :param reaction_id:
            Internal reaction id.
        :type reaction_id: str

        :param species:
            Species display mode.
        :type species: str

        :param include_id:
            Whether to include the internal reaction id.
        :type include_id: bool

        :param include_rule:
            Whether to include rule provenance.
        :type include_rule: bool

        :param include_step:
            Whether to include step provenance.
        :type include_step: bool

        :param arrow:
            Arrow string between lhs and rhs.
        :type arrow: str

        :returns:
            Human-readable reaction string.
        :rtype: str

        Example
        -------
        .. code-block:: python

            syn.format_reaction("r_1", species="label", include_rule=True)
        """
        rxn = self.reactions[reaction_id]
        return rxn.format(
            lambda sid: self._species_token(sid, species),
            include_id=include_id,
            include_rule=include_rule,
            include_step=include_step,
            arrow=arrow,
        )

    def to_equations(
        self,
        *,
        species: str = "label",
        include_id: bool = True,
        include_rule: bool = False,
        include_step: bool = False,
        arrow: str = ">>",
    ) -> List[str]:
        """
        Return the network as a list of formatted reaction equations.

        :param species:
            Species display mode.
        :type species: str

        :param include_id:
            Whether to include internal reaction ids.
        :type include_id: bool

        :param include_rule:
            Whether to include rule provenance.
        :type include_rule: bool

        :param include_step:
            Whether to include step provenance.
        :type include_step: bool

        :param arrow:
            Arrow string between lhs and rhs.
        :type arrow: str

        :returns:
            List of formatted reaction equations.
        :rtype: List[str]

        Example
        -------
        .. code-block:: python

            eqs = syn.to_equations(species="smiles", include_rule=True)
            print("\\n".join(eqs))
        """
        return [
            self.format_reaction(
                rid,
                species=species,
                include_id=include_id,
                include_rule=include_rule,
                include_step=include_step,
                arrow=arrow,
            )
            for rid in self.reaction_ids
        ]

    def describe(
        self,
        *,
        include_species: bool = False,
        species: str = "label",
    ) -> str:
        """
        Return a human-readable multiline description of the network.

        :param include_species:
            Whether to append a final line listing species names.
        :type include_species: bool

        :param species:
            Species display mode used in the text summary.
        :type species: str

        :returns:
            Multiline text description.
        :rtype: str

        Example
        -------
        .. code-block:: python

            print(syn.describe(include_species=True, species="label"))
        """
        lines = [f"SynCRN: {self.n_species} species, {self.n_reactions} reactions"]

        for rid in self.reaction_ids:
            rxn = self.reactions[rid]
            lhs = rxn.format_side(
                rxn.lhs, lambda sid: self._species_token(sid, species)
            )
            rhs = rxn.format_side(
                rxn.rhs, lambda sid: self._species_token(sid, species)
            )

            line = f"  {rxn.id}: {lhs} >> {rhs}"
            if rxn.rule_index is not None:
                line += f" rule {rxn.rule_index}"

            lines.append(line)

        if include_species:
            names = [self._species_token(sid, species) for sid in self.species_ids]
            lines.append("Species: " + ", ".join(names))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a nested JSON-like dictionary representation.

        :returns:
            Full SynCRN object as a dictionary.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            data = syn.to_dict()
            print(data["species"].keys())
        """
        return {
            "graph_attrs": dict(self.graph_attrs),
            "metadata": dict(self.metadata),
            "species": {sid: sp.to_dict() for sid, sp in self.species.items()},
            "rules": {rid: rule.to_dict() for rid, rule in self.rules.items()},
            "reactions": {rid: rxn.to_dict() for rid, rxn in self.reactions.items()},
        }

    def to_stoichiometric_matrices(self) -> Dict[str, Any]:
        """
        Construct stoichiometric matrices in canonical species and reaction order.

        The returned dictionary contains:

        - ``species_order``
        - ``reaction_order``
        - ``S_minus``: reactant-incidence matrix
        - ``S_plus``: product-incidence matrix
        - ``S``: net stoichiometric matrix

        :returns:
            Stoichiometric matrix view of the network.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            mats = syn.to_stoichiometric_matrices()
            print(mats["species_order"])
            print(mats["reaction_order"])
            print(mats["S"])
        """
        species_order = self.species_ids
        reaction_order = self.reaction_ids

        n = len(species_order)
        m = len(reaction_order)

        sidx = {sid: i for i, sid in enumerate(species_order)}
        ridx = {rid: j for j, rid in enumerate(reaction_order)}

        s_minus = [[0 for _ in range(m)] for _ in range(n)]
        s_plus = [[0 for _ in range(m)] for _ in range(n)]

        for rid, rxn in self.reactions.items():
            j = ridx[rid]
            for sid, coeff in rxn.lhs.items():
                s_minus[sidx[sid]][j] = coeff
            for sid, coeff in rxn.rhs.items():
                s_plus[sidx[sid]][j] = coeff

        s_net = [[s_plus[i][j] - s_minus[i][j] for j in range(m)] for i in range(n)]

        return {
            "species_order": species_order,
            "reaction_order": reaction_order,
            "S_minus": s_minus,
            "S_plus": s_plus,
            "S": s_net,
        }

    def to_petrinet(self) -> Dict[str, Any]:
        """
        Return a Petri-net style pre/post incidence view.

        The returned dictionary contains:

        - ``places``: species ids
        - ``transitions``: reaction ids
        - ``pre``: input incidence map
        - ``post``: output incidence map

        :returns:
            Petri-net incidence representation.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            pn = syn.to_petrinet()
            print(pn["pre"])
            print(pn["post"])
        """
        pre: Dict[str, Dict[str, int]] = {sid: {} for sid in self.species_ids}
        post: Dict[str, Dict[str, int]] = {sid: {} for sid in self.species_ids}

        for rid, rxn in self.reactions.items():
            for sid, coeff in rxn.lhs.items():
                pre[sid][rid] = coeff
            for sid, coeff in rxn.rhs.items():
                post[sid][rid] = coeff

        return {
            "places": list(self.species_ids),
            "transitions": list(self.reaction_ids),
            "pre": pre,
            "post": post,
        }

    def to_digraph(
        self,
        *,
        node_ids: str = "source",
        reaction_kind: Optional[str] = None,
        include_internal_ids: bool = True,
    ) -> nx.DiGraph:
        """
        Reconstruct a species--reaction bipartite digraph.

        By default, this method preserves original node ids and original
        reaction-node kinds. This means that input reaction nodes with
        ``kind="rule"`` will still appear as ``kind="rule"`` after round-trip.

        :param node_ids:
            ``"source"`` preserves original node ids, while ``"internal"``
            uses canonical ids such as ``s_1`` and ``r_1``.
        :type node_ids: str

        :param reaction_kind:
            Optional override for reconstructed reaction-node kind.
        :type reaction_kind: Optional[str]

        :param include_internal_ids:
            Whether to attach ``syncrn_id`` and ``source_node_id`` as node attributes.
        :type include_internal_ids: bool

        :returns:
            Reconstructed bipartite digraph.
        :rtype: nx.DiGraph

        :raises ValueError:
            If ``node_ids`` is not ``"source"`` or ``"internal"``.

        Example
        -------
        .. code-block:: python

            g2 = syn.to_digraph()
            g3 = syn.to_digraph(node_ids="internal", reaction_kind="reaction")
        """
        if node_ids not in {"source", "internal"}:
            raise ValueError("node_ids must be 'source' or 'internal'")

        g = nx.DiGraph()
        g.graph.update(self.graph_attrs)
        g.graph.update(self.metadata)

        species_node_map: Dict[str, Hashable] = {}
        reaction_node_map: Dict[str, Hashable] = {}

        for sid, sp in self.species.items():
            nid = _resolve_species_node_id(sp, node_ids=node_ids)
            g.add_node(
                nid,
                **_species_node_attrs(sp, include_internal_ids=include_internal_ids),
            )
            species_node_map[sid] = nid

        for rid, rxn in self.reactions.items():
            nid = _resolve_reaction_node_id(rxn, node_ids=node_ids)
            g.add_node(
                nid,
                **_reaction_node_attrs(
                    rxn,
                    reaction_kind=reaction_kind,
                    include_internal_ids=include_internal_ids,
                ),
            )
            reaction_node_map[rid] = nid

        for rid, rxn in self.reactions.items():
            _add_reaction_edges_to_graph(
                g,
                rxn=rxn,
                reaction_node=reaction_node_map[rid],
                species_node_map=species_node_map,
            )

        return g

    @classmethod
    def from_reaction_strings(
        cls,
        rxns: List[str],
        rules: Optional[List[Optional[str]]] = None,
        *,
        parser: Optional[Callable[[str], Any]] = None,
        strict: bool = True,
    ) -> "SynCRN":
        """
        Build a SynCRN object directly from reaction strings.

        Reactions and rules are interpreted pairwise, so ``rxns[i]`` corresponds
        to ``rules[i]``.

        ID policy
        ---------
        This constructor uses plain numeric string ids instead of prefixed ids.

        - Species ids: ``"1"``, ``"2"``, ...
        - Reaction ids: continue after species ids
        - Rule ids: ``"1"``, ``"2"``, ... within the separate rule table

        Species are indexed by first appearance in the input reactions.

        :param rxns:
            List of reaction strings such as ``"2A>>B+3C"``.
        :type rxns: List[str]

        :param rules:
            Optional list of rule strings pairwise aligned with ``rxns``.
        :type rules: Optional[List[Optional[str]]]

        :param parser:
            Optional side parser. It should accept one side string such as
            ``"2A+B"`` and return either a mapping, an object with
            ``to_dict()``, or an object with ``items()``.
        :type parser: Optional[Callable[[str], Any]]

        :param strict:
            Whether malformed reaction strings or empty sides should raise an error.
        :type strict: bool

        :returns:
            Canonical SynCRN object.
        :rtype: SynCRN

        :raises TypeError:
            If ``rxns`` or ``rules`` have invalid types.

        :raises ValueError:
            If reaction strings are malformed or rules are not pairwise aligned.

        Example
        -------
        .. code-block:: python

            syn = SynCRN.from_reaction_strings(["2A>>B+3C"])

            syn = SynCRN.from_reaction_strings(
                ["A+B>>C", "C>>D"],
                rules=["rule_1", "rule_2"],
                parser=RXNSide.from_str,
            )
        """
        if not isinstance(rxns, (list, tuple)) or not all(
            isinstance(x, str) for x in rxns
        ):
            raise TypeError("rxns must be a list or tuple of reaction strings")

        if len(rxns) == 0:
            return cls(
                species={},
                reactions={},
                rules={},
                graph_attrs={},
                metadata={
                    "source": "reaction_strings",
                    "n_input_reactions": 0,
                    "has_pairwise_rules": False,
                },
            )

        has_rules = rules is not None and len(rules) > 0
        if has_rules:
            if not isinstance(rules, (list, tuple)):
                raise TypeError("rules must be a list or tuple when provided")
            if len(rules) != len(rxns):
                raise ValueError(
                    f"rules must be pairwise with rxns: got {len(rxns)} reactions and "
                    f"{len(rules)} rules"
                )
            if not all(r is None or isinstance(r, str) for r in rules):
                raise TypeError("rules entries must be strings or None")

        parse_side = parser if parser is not None else _default_parse_side_text

        parsed_rxns = [
            _parse_reaction_string_entry(
                rxn_text=rxn_text,
                rxn_index=i,
                rule_repr=rules[i] if has_rules else None,
                has_rules=has_rules,
                parse_side=parse_side,
                strict=strict,
            )
            for i, rxn_text in enumerate(rxns)
        ]

        species_order = _species_order_from_parsed_reactions(parsed_rxns)
        label_to_sid = {
            label: str(idx) for idx, label in enumerate(species_order, start=1)
        }

        species = _build_species_table_from_labels(species_order)
        rules_table = _build_rules_table_from_reaction_strings(
            parsed_rxns,
            has_rules=has_rules,
        )
        reactions = _build_reactions_from_parsed_strings(
            parsed_rxns,
            label_to_sid=label_to_sid,
            reaction_start_index=len(species_order) + 1,
            has_rules=has_rules,
        )

        return cls(
            species=species,
            reactions=reactions,
            rules=rules_table,
            graph_attrs={},
            metadata={
                "source": "reaction_strings",
                "n_input_reactions": len(rxns),
                "has_pairwise_rules": has_rules,
            },
        )
