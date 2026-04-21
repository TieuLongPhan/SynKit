from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import hashlib

import networkx as nx

__all__ = [
    "SymmetryConfig",
    "WLResult",
    "AutomorphismResult",
    "IsomorphismResult",
    "CanonicalResult",
    "freeze",
    "hash_text",
    "prepare_graph",
    "node_token",
    "edge_token",
    "node_matcher",
    "edge_matcher",
    "build_fast_signature",
    "invert_mapping",
    "orbits_from_mappings",
    "should_stop",
    "graph_key_from_order",
    "approx_automorphism_count_from_cells",
]


@dataclass(frozen=True)
class SymmetryConfig:
    """
    Configuration controlling semantic versus topological symmetry matching.

    The configuration determines which node and edge attributes participate in
    token construction, canonicalization, isomorphism tests, and automorphism
    analysis.

    :param kind_attr_key:
        Node attribute name used to identify node kind.
    :type kind_attr_key: str

    :param species_kind_value:
        Attribute value indicating a species node.
    :type species_kind_value: str

    :param rule_kind_value:
        Attribute value indicating a rule/reaction node.
    :type rule_kind_value: str

    :param require_kind:
        Whether node kind must always be included in node tokens.
    :type require_kind: bool

    :param species_identity_keys:
        Preferred attribute keys used to identify species nodes.
    :type species_identity_keys: Tuple[str, ...]

    :param rule_identity_keys:
        Preferred attribute keys used to identify rule nodes.
    :type rule_identity_keys: Tuple[str, ...]

    :param species_extra_attr_keys:
        Additional species-node attributes to include in node tokens.
    :type species_extra_attr_keys: Tuple[str, ...]

    :param rule_extra_attr_keys:
        Additional rule-node attributes to include in node tokens.
    :type rule_extra_attr_keys: Tuple[str, ...]

    :param other_node_attr_keys:
        Attributes to include for nodes that are neither species nor rule nodes.
    :type other_node_attr_keys: Tuple[str, ...]

    :param generic_node_attr_keys:
        Generic node attributes appended for all node kinds.
    :type generic_node_attr_keys: Tuple[str, ...]

    :param ignored_node_attr_keys:
        Node attributes ignored during token construction.
    :type ignored_node_attr_keys: Tuple[str, ...]

    :param edge_attr_keys:
        Edge attributes included in edge tokens.
    :type edge_attr_keys: Tuple[str, ...]

    :param ignored_edge_attr_keys:
        Edge attributes ignored during edge token construction.
    :type ignored_edge_attr_keys: Tuple[str, ...]

    :param use_edge_direction:
        Whether edge direction is respected in symmetry computations.
    :type use_edge_direction: bool

    :param include_missing_keys:
        Whether missing identity keys should be represented explicitly as
        ``None`` in node tokens.
    :type include_missing_keys: bool

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Sym._common import SymmetryConfig

        semantic_cfg = SymmetryConfig.semantic()
        topo_cfg = SymmetryConfig.topological()
    """

    kind_attr_key: str = "kind"
    species_kind_value: str = "species"
    rule_kind_value: str = "rule"
    require_kind: bool = True

    species_identity_keys: Tuple[str, ...] = ("smiles", "label")
    rule_identity_keys: Tuple[str, ...] = ("rule_repr", "rxn_repr", "label")

    species_extra_attr_keys: Tuple[str, ...] = ()
    rule_extra_attr_keys: Tuple[str, ...] = ()
    other_node_attr_keys: Tuple[str, ...] = ()
    generic_node_attr_keys: Tuple[str, ...] = ()

    ignored_node_attr_keys: Tuple[str, ...] = (
        "syncrn_id",
        "source_node_id",
        "app_index",
        "rule_index",
        "step",
        "rxn_id",
    )

    edge_attr_keys: Tuple[str, ...] = ("role", "stoich")
    ignored_edge_attr_keys: Tuple[str, ...] = ("rxn_id",)
    use_edge_direction: bool = True
    include_missing_keys: bool = False

    @classmethod
    def semantic(cls) -> "SymmetryConfig":
        """
        Build a semantic symmetry configuration.

        :returns:
            Configuration that preserves node identities and stoichiometric edge
            semantics.
        :rtype: SymmetryConfig
        """
        return cls(
            species_identity_keys=("smiles", "label"),
            rule_identity_keys=("rule_repr", "rxn_repr", "label"),
            edge_attr_keys=("role", "stoich"),
            require_kind=True,
            use_edge_direction=True,
        )

    @classmethod
    def topological(cls) -> "SymmetryConfig":
        """
        Build a topology-focused symmetry configuration.

        This mode ignores species and rule identities while still respecting
        node kinds and selected edge annotations.

        :returns:
            Configuration emphasizing graph topology over node identity.
        :rtype: SymmetryConfig
        """
        return cls(
            species_identity_keys=(),
            rule_identity_keys=(),
            species_extra_attr_keys=(),
            rule_extra_attr_keys=(),
            other_node_attr_keys=(),
            generic_node_attr_keys=(),
            edge_attr_keys=("role", "stoich"),
            require_kind=True,
            use_edge_direction=True,
        )


@dataclass(frozen=True)
class WLResult:
    """
    Result of Weisfeiler-Lehman refinement.

    :param colors:
        Final color assigned to each node.
    :type colors: Dict[Any, str]

    :param orbits:
        Approximate orbit partition induced by final color classes.
    :type orbits: List[Set[Any]]

    :param color_hist:
        Histogram of final colors.
    :type color_hist: Dict[str, int]

    :param iters_run:
        Number of WL iterations actually executed.
    :type iters_run: int

    :param stabilized:
        Whether refinement stabilized before the iteration limit.
    :type stabilized: bool

    :param approx_automorphism_count:
        Approximate automorphism count inferred from unresolved cells, if
        available.
    :type approx_automorphism_count: Optional[int]

    :param canonical_order:
        Canonical node ordering induced by WL colors and tie-breaking rules.
    :type canonical_order: List[Any]
    """

    colors: Dict[Any, str]
    orbits: List[Set[Any]]
    color_hist: Dict[str, int]
    iters_run: int
    stabilized: bool
    approx_automorphism_count: Optional[int]
    canonical_order: List[Any]


@dataclass(frozen=True)
class AutomorphismResult:
    """
    Result of an automorphism computation.

    :param graph_type:
        Graph representation type used in the computation.
    :type graph_type: str

    :param automorphism_count:
        Number of discovered automorphisms.
    :type automorphism_count: int

    :param sample_mappings:
        Sample node mappings realizing automorphisms.
    :type sample_mappings: List[Dict[Any, Any]]

    :param orbits:
        Orbit partition induced by the sampled mappings.
    :type orbits: List[Set[Any]]

    :param elapsed_seconds:
        Runtime in seconds.
    :type elapsed_seconds: float

    :param stopped_early:
        Whether the computation stopped early due to timeout or count limit.
    :type stopped_early: bool
    """

    graph_type: str
    automorphism_count: int
    sample_mappings: List[Dict[Any, Any]]
    orbits: List[Set[Any]]
    elapsed_seconds: float
    stopped_early: bool


@dataclass(frozen=True)
class IsomorphismResult:
    """
    Result of an isomorphism test.

    :param isomorphic:
        Whether the compared graphs are isomorphic.
    :type isomorphic: bool

    :param mapping:
        Witness mapping from the first graph to the second graph, if found.
    :type mapping: Optional[Dict[Any, Any]]

    :param elapsed_seconds:
        Runtime in seconds.
    :type elapsed_seconds: float

    :param rejected_by_invariants:
        Whether the comparison was rejected using fast invariants before an exact
        search.
    :type rejected_by_invariants: bool

    :param mode:
        Matching mode identifier, for example semantic or topological.
    :type mode: str
    """

    isomorphic: bool
    mapping: Optional[Dict[Any, Any]]
    elapsed_seconds: float
    rejected_by_invariants: bool
    mode: str


@dataclass(frozen=True)
class CanonicalResult:
    """
    Result of graph canonicalization.

    :param canonical_order:
        Canonical node ordering.
    :type canonical_order: List[Any]

    :param canonical_key:
        Canonical key derived from the canonical order.
    :type canonical_key: Tuple[Any, ...]

    :param exact:
        Whether the result is exact rather than approximate.
    :type exact: bool

    :param elapsed_seconds:
        Runtime in seconds.
    :type elapsed_seconds: float
    """

    canonical_order: List[Any]
    canonical_key: Tuple[Any, ...]
    exact: bool
    elapsed_seconds: float


def freeze(x: Any) -> Hashable:
    """
    Convert common container types into a stable hashable representation.

    :param x:
        Object to freeze.
    :type x: Any

    :returns:
        Hashable recursively frozen representation.
    :rtype: Hashable

    Examples
    --------
    .. code-block:: python

        freeze({"b": [2, 1], "a": {"x", "y"}})
    """
    if isinstance(x, dict):
        return tuple(
            (k, freeze(v)) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted((freeze(v) for v in x), key=str))
    return x


def hash_text(text: str, *, digest_size: int = 16) -> str:
    """
    Hash text with BLAKE2b.

    :param text:
        Input text.
    :type text: str

    :param digest_size:
        Output digest size in bytes.
    :type digest_size: int

    :returns:
        Hex digest string.
    :rtype: str
    """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()


def should_stop(
    start: float,
    timeout_sec: Optional[float],
    *,
    count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> bool:
    """
    Decide whether a search should stop.

    :param start:
        Start time from :func:`perf_counter`.
    :type start: float

    :param timeout_sec:
        Optional timeout in seconds.
    :type timeout_sec: Optional[float]

    :param count:
        Current number of discovered solutions.
    :type count: Optional[int]

    :param max_count:
        Optional maximum number of solutions to collect.
    :type max_count: Optional[int]

    :returns:
        ``True`` if the search should stop.
    :rtype: bool
    """
    if timeout_sec is not None and perf_counter() - start > timeout_sec:
        return True
    if max_count is not None and count is not None and count >= max_count:
        return True
    return False


def _factorial_capped(n: int, cap: int) -> int:
    """
    Compute ``n!`` with an upper cap.

    :param n:
        Nonnegative integer.
    :type n: int

    :param cap:
        Upper cap for early stopping.
    :type cap: int

    :returns:
        ``min(n!, cap)``.
    :rtype: int
    """
    out = 1
    for i in range(2, n + 1):
        out *= i
        if out >= cap:
            return cap
    return out


def _product_factorials_capped(values: Iterable[int], cap: int) -> int:
    """
    Compute a capped product of factorials.

    :param values:
        Iterable of cell sizes.
    :type values: Iterable[int]

    :param cap:
        Upper cap for early stopping.
    :type cap: int

    :returns:
        Product of factorials, capped at ``cap``.
    :rtype: int
    """
    out = 1
    for v in values:
        out *= _factorial_capped(v, cap)
        if out >= cap:
            return cap
    return out


def approx_automorphism_count_from_cells(
    cells: Iterable[Sequence[Any]], *, cap: int = 10**18
) -> int:
    """
    Approximate automorphism count from unresolved cell sizes.

    This estimate assumes each cell can be permuted independently.

    :param cells:
        Iterable of partition cells.
    :type cells: Iterable[Sequence[Any]]

    :param cap:
        Upper cap on the returned count.
    :type cap: int

    :returns:
        Approximate capped automorphism count.
    :rtype: int
    """
    return _product_factorials_capped((len(c) for c in cells), cap)


def _copy_as_digraph(G: nx.Graph) -> nx.DiGraph:
    """
    Copy a NetworkX graph into a simple directed graph.

    Undirected edges are duplicated in both directions. Multi-edges are merged
    into a simple :class:`nx.DiGraph`.

    :param G:
        Input graph.
    :type G: nx.Graph

    :returns:
        Directed graph copy.
    :rtype: nx.DiGraph
    """
    if isinstance(G, nx.DiGraph):
        return G.copy()
    if isinstance(G, nx.MultiDiGraph):
        H = nx.DiGraph()
        H.add_nodes_from((n, dict(d)) for n, d in G.nodes(data=True))
        for u, v, _, d in G.edges(keys=True, data=True):
            if H.has_edge(u, v):
                H[u][v].update(d)
            else:
                H.add_edge(u, v, **dict(d))
        return H
    if isinstance(G, nx.MultiGraph):
        H = nx.DiGraph()
        H.add_nodes_from((n, dict(d)) for n, d in G.nodes(data=True))
        for u, v, _, d in G.edges(keys=True, data=True):
            if not H.has_edge(u, v):
                H.add_edge(u, v, **dict(d))
            if not H.has_edge(v, u):
                H.add_edge(v, u, **dict(d))
        return H
    H = nx.DiGraph()
    H.add_nodes_from((n, dict(d)) for n, d in G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **dict(d))
        if not G.is_directed() and not H.has_edge(v, u):
            H.add_edge(v, u, **dict(d))
    return H



def _remove_stoich_attrs(G: nx.DiGraph) -> nx.DiGraph:
    """
    Return a copy of G with stoichiometric edge attributes removed.
    """
    H = G.__class__()
    H.graph.update(dict(G.graph))

    for n, attrs in G.nodes(data=True):
        H.add_node(n, **dict(attrs))

    if G.is_multigraph():
        for u, v, k, attrs in G.edges(keys=True, data=True):
            new_attrs = dict(attrs)
            new_attrs.pop("stoich", None)
            H.add_edge(u, v, key=k, **new_attrs)
    else:
        for u, v, attrs in G.edges(data=True):
            new_attrs = dict(attrs)
            new_attrs.pop("stoich", None)
            H.add_edge(u, v, **new_attrs)

    return H


def _integerize_graph_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """
    Return a copy of G with node ids relabeled to consecutive integers starting at 1.

    Original labels are stored in node attribute ``source_node_id``.
    """
    order = sorted(G.nodes(), key=str)
    mapping = {old: i + 1 for i, old in enumerate(order)}
    H = nx.relabel_nodes(G, mapping, copy=True)

    for old, new in mapping.items():
        H.nodes[new]["source_node_id"] = old

    return H


def prepare_graph(
    source: Any,
    *,
    include_rule: bool = True,
    integer_ids: bool = False,
    include_stoich: bool = True,
) -> Tuple[nx.DiGraph, str]:
    """
    Convert a supported source object into a directed graph.

    Supported inputs are:
    - a NetworkX graph
    - an object exposing ``to_digraph()``

    :param source:
        Input graph-like object.
    :type source: Any

    :param include_rule:
        Whether the resulting graph should be treated as bipartite
        species-rule graph rather than a species-only graph.
    :type include_rule: bool

    :param integer_ids:
        Reserved compatibility flag.
    :type integer_ids: bool

    :param include_stoich:
        Reserved compatibility flag.
    :type include_stoich: bool

    :returns:
        Pair ``(graph, graph_type)``.
    :rtype: Tuple[nx.DiGraph, str]

    :raises TypeError:
        If the input is unsupported.
    """
    _ = integer_ids
    _ = include_stoich

    if isinstance(source, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        G = _copy_as_digraph(source)
    elif hasattr(source, "to_digraph") and callable(source.to_digraph):
        G = _copy_as_digraph(source.to_digraph())
    else:
        raise TypeError("Expected a NetworkX graph or an object with to_digraph().")

    if not include_stoich:
        G = _remove_stoich_attrs(G)

    if integer_ids:
        G = _integerize_graph_nodes(G)

    return G, ("bipartite" if include_rule else "species")


def _first_present(attrs: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """
    Return the first non-``None`` attribute value among preferred keys.

    :param attrs:
        Attribute mapping.
    :type attrs: Mapping[str, Any]

    :param keys:
        Candidate attribute keys in priority order.
    :type keys: Sequence[str]

    :returns:
        First present non-``None`` value, or ``None``.
    :rtype: Any
    """
    for k in keys:
        v = attrs.get(k, None)
        if v is not None:
            return v
    return None


def node_kind(attrs: Mapping[str, Any], config: SymmetryConfig) -> Any:
    """
    Extract node kind from an attribute mapping.

    :param attrs:
        Node attribute mapping.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :returns:
        Node kind value or ``None``.
    :rtype: Any
    """
    return attrs.get(config.kind_attr_key, None)


def _append_identity_token(
    out: List[Any],
    attrs: Mapping[str, Any],
    identity_keys: Sequence[str],
    config: SymmetryConfig,
) -> None:
    """
    Append an identity token to a node token list when available.

    :param out:
        Output token list to mutate.
    :type out: List[Any]

    :param attrs:
        Node attributes.
    :type attrs: Mapping[str, Any]

    :param identity_keys:
        Preferred identity keys.
    :type identity_keys: Sequence[str]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :returns:
        ``None``.
    :rtype: None
    """
    ident = _first_present(attrs, identity_keys)
    if ident is not None:
        out.append(("identity", freeze(ident)))
    elif config.include_missing_keys and identity_keys:
        out.append(("identity", None))


def _append_selected_attrs(
    out: List[Any],
    attrs: Mapping[str, Any],
    keys: Sequence[str],
    ignored_keys: Sequence[str],
) -> None:
    """
    Append selected attributes to a token list.

    :param out:
        Output token list to mutate.
    :type out: List[Any]

    :param attrs:
        Attribute mapping.
    :type attrs: Mapping[str, Any]

    :param keys:
        Attribute keys to include.
    :type keys: Sequence[str]

    :param ignored_keys:
        Attribute keys to skip.
    :type ignored_keys: Sequence[str]

    :returns:
        ``None``.
    :rtype: None
    """
    for k in keys:
        if k not in ignored_keys:
            out.append((k, freeze(attrs.get(k, None))))


def _species_node_token(
    attrs: Mapping[str, Any], config: SymmetryConfig, out: List[Any]
) -> None:
    """
    Append species-specific token components.

    :param attrs:
        Node attributes.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :param out:
        Output token list to mutate.
    :type out: List[Any]

    :returns:
        ``None``.
    :rtype: None
    """
    _append_identity_token(out, attrs, config.species_identity_keys, config)
    _append_selected_attrs(
        out,
        attrs,
        config.species_extra_attr_keys,
        config.ignored_node_attr_keys,
    )


def _rule_node_token(
    attrs: Mapping[str, Any], config: SymmetryConfig, out: List[Any]
) -> None:
    """
    Append rule-specific token components.

    :param attrs:
        Node attributes.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :param out:
        Output token list to mutate.
    :type out: List[Any]

    :returns:
        ``None``.
    :rtype: None
    """
    _append_identity_token(out, attrs, config.rule_identity_keys, config)
    _append_selected_attrs(
        out,
        attrs,
        config.rule_extra_attr_keys,
        config.ignored_node_attr_keys,
    )


def _other_node_token(
    attrs: Mapping[str, Any], config: SymmetryConfig, out: List[Any]
) -> None:
    """
    Append generic token components for non-species, non-rule nodes.

    :param attrs:
        Node attributes.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :param out:
        Output token list to mutate.
    :type out: List[Any]

    :returns:
        ``None``.
    :rtype: None
    """
    _append_selected_attrs(
        out,
        attrs,
        config.other_node_attr_keys,
        config.ignored_node_attr_keys,
    )


def _append_generic_node_attrs(
    attrs: Mapping[str, Any], config: SymmetryConfig, out: List[Any]
) -> None:
    """
    Append generic node attributes shared across all node kinds.

    :param attrs:
        Node attributes.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :param out:
        Output token list to mutate.
    :type out: List[Any]

    :returns:
        ``None``.
    :rtype: None
    """
    for k in config.generic_node_attr_keys:
        if k == config.kind_attr_key or k in config.ignored_node_attr_keys:
            continue
        out.append((k, freeze(attrs.get(k, None))))


def node_token(attrs: Mapping[str, Any], config: SymmetryConfig) -> Tuple[Any, ...]:
    """
    Build a canonical token for a node.

    The token depends on node kind, configured identity keys, optional extra
    attributes, and generic attributes shared by all node kinds.

    :param attrs:
        Node attribute mapping.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration controlling token content.
    :type config: SymmetryConfig

    :returns:
        Immutable node token.
    :rtype: Tuple[Any, ...]

    Examples
    --------
    .. code-block:: python

        cfg = SymmetryConfig.semantic()
        tok = node_token({"kind": "species", "smiles": "CCO"}, cfg)
        print(tok)
    """
    kind = node_kind(attrs, config)
    out: List[Any] = []

    if config.require_kind:
        out.append((config.kind_attr_key, freeze(kind)))

    if kind == config.species_kind_value:
        _species_node_token(attrs, config, out)
    elif kind == config.rule_kind_value:
        _rule_node_token(attrs, config, out)
    else:
        _other_node_token(attrs, config, out)

    _append_generic_node_attrs(attrs, config, out)
    return tuple(out)


def edge_token(attrs: Mapping[str, Any], config: SymmetryConfig) -> Tuple[Any, ...]:
    """
    Build a canonical token for an edge.

    :param attrs:
        Edge attribute mapping.
    :type attrs: Mapping[str, Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :returns:
        Immutable edge token.
    :rtype: Tuple[Any, ...]

    Examples
    --------
    .. code-block:: python

        cfg = SymmetryConfig.semantic()
        tok = edge_token({"role": "reactant", "stoich": 2}, cfg)
    """
    out: List[Any] = []
    for k in config.edge_attr_keys:
        if k in config.ignored_edge_attr_keys:
            continue
        out.append((k, freeze(attrs.get(k, None))))
    return tuple(out)


def node_matcher(
    config: SymmetryConfig,
) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """
    Build a node matcher function based on :func:`node_token`.

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :returns:
        Predicate comparing two node attribute dictionaries.
    :rtype: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    """

    def match(a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
        return node_token(a1, config) == node_token(a2, config)

    return match


def edge_matcher(
    config: SymmetryConfig,
) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """
    Build an edge matcher function based on :func:`edge_token`.

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :returns:
        Predicate comparing two edge attribute dictionaries.
    :rtype: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    """

    def match(a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
        return edge_token(a1, config) == edge_token(a2, config)

    return match


def build_fast_signature(
    G: nx.DiGraph,
    graph_type: str,
    config: SymmetryConfig,
    *,
    wl_color_hist: Optional[Mapping[str, int]] = None,
) -> Tuple[Any, ...]:
    """
    Build a fast graph signature for cheap rejection tests.

    The signature includes graph size, node-token histogram, edge-token
    histogram, in/out-degree histogram, and optional WL color histogram.

    :param G:
        Directed graph.
    :type G: nx.DiGraph

    :param graph_type:
        Graph type label.
    :type graph_type: str

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :param wl_color_hist:
        Optional WL color histogram.
    :type wl_color_hist: Optional[Mapping[str, int]]

    :returns:
        Fast signature tuple.
    :rtype: Tuple[Any, ...]
    """
    node_hist = Counter(node_token(G.nodes[v], config) for v in G.nodes())
    edge_hist = Counter(edge_token(attrs, config) for _, _, attrs in G.edges(data=True))
    deg_hist = Counter((G.in_degree(v), G.out_degree(v)) for v in G.nodes())
    return (
        graph_type,
        G.number_of_nodes(),
        G.number_of_edges(),
        tuple(sorted(node_hist.items(), key=str)),
        tuple(sorted(edge_hist.items(), key=str)),
        tuple(sorted(deg_hist.items(), key=str)),
        tuple(sorted((wl_color_hist or {}).items(), key=str)),
    )


def graph_key_from_order(
    G: nx.DiGraph, order: Sequence[Any], config: SymmetryConfig
) -> Tuple[Any, ...]:
    """
    Build a canonical graph key from a node order.

    :param G:
        Directed graph.
    :type G: nx.DiGraph

    :param order:
        Node order.
    :type order: Sequence[Any]

    :param config:
        Symmetry configuration.
    :type config: SymmetryConfig

    :returns:
        Canonical graph key.
    :rtype: Tuple[Any, ...]
    """
    pos = {v: i for i, v in enumerate(order)}
    node_part = tuple((i, node_token(G.nodes[v], config)) for i, v in enumerate(order))
    edge_rows: List[Tuple[int, int, Tuple[Any, ...]]] = []
    for u, v, attrs in G.edges(data=True):
        edge_rows.append((pos[u], pos[v], edge_token(attrs, config)))
    edge_rows.sort(key=lambda x: (x[0], x[1], str(x[2])))
    return (node_part, tuple(edge_rows))


def invert_mapping(mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Invert a one-to-one mapping.

    :param mapping:
        Mapping to invert.
    :type mapping: Dict[Any, Any]

    :returns:
        Inverted mapping.
    :rtype: Dict[Any, Any]
    """
    return {v: k for k, v in mapping.items()}


def orbits_from_mappings(
    nodes: Sequence[Any], mappings: Iterable[Dict[Any, Any]]
) -> List[Set[Any]]:
    """
    Build orbit classes from a collection of automorphism mappings.

    :param nodes:
        Nodes whose orbit partition should be computed.
    :type nodes: Sequence[Any]

    :param mappings:
        Automorphism mappings.
    :type mappings: Iterable[Dict[Any, Any]]

    :returns:
        List of orbit sets.
    :rtype: List[Set[Any]]

    Examples
    --------
    .. code-block:: python

        nodes = ["A", "B", "C"]
        maps = [{"A": "B", "B": "A", "C": "C"}]
        print(orbits_from_mappings(nodes, maps))
    """
    parent: Dict[Any, Any] = {n: n for n in nodes}

    def find(x: Any) -> Any:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for m in mappings:
        for a, b in m.items():
            if a in parent and b in parent:
                union(a, b)

    buckets: Dict[Any, Set[Any]] = defaultdict(set)
    for n in nodes:
        buckets[find(n)].add(n)
    return list(buckets.values())


def canonical_graph_from_order(
    G: nx.DiGraph,
    order: Sequence[Any],
    *,
    integer_ids: bool = False,
) -> nx.DiGraph:
    """
    Build a canonical graph from a canonical node order.

    If ``integer_ids`` is True, nodes are relabeled to 1..n.
    Otherwise, original node labels are preserved, but nodes are inserted in
    canonical order.
    """
    if integer_ids:
        mapping = {v: i + 1 for i, v in enumerate(order)}
        return nx.relabel_nodes(G, mapping, copy=True)

    H = G.__class__()
    H.graph.update(dict(G.graph))

    for v in order:
        H.add_node(v, **dict(G.nodes[v]))

    pos = {v: i for i, v in enumerate(order)}

    if G.is_multigraph():
        edges = sorted(
            G.edges(keys=True, data=True),
            key=lambda x: (pos[x[0]], pos[x[1]], str(x[2]), str(x[3])),
        )
        for u, v, k, attrs in edges:
            H.add_edge(u, v, key=k, **dict(attrs))
    else:
        edges = sorted(
            G.edges(data=True),
            key=lambda x: (pos[x[0]], pos[x[1]], str(x[2])),
        )
        for u, v, attrs in edges:
            H.add_edge(u, v, **dict(attrs))

    return H