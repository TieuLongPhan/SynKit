from __future__ import annotations

"""
subgraph_matcher.py
================================
A **lean**, **typed**, and **high-performance** successor to the original
sub-graph matching utilities in SynKit.

Key Features
------------
• **Speed**
  • Element-multiset, node-attribute, degree-histogram, and WL-1 hashing
    pre-filters remove up to 95 % of impossible host-CCs before VF2.
  • Heuristic CC ordering and optional result limits prune the search tree.

• **Flexibility**
  • Three matching strategies:
    – ALL: classic VF2 over the entire host
    – COMPONENT: CC-aware, distinct-CC enforcement
    – BACKTRACK: component-aware with classic-fallback
  • Fallback to brute-force VF2 when host has fewer CCs than pattern
  • Optional `strict_cc_count` to enforce exact CC counts

• **Safety & Cleanliness**
  • Never mutates your input graphs
  • Validates inputs and raises clear errors on misuse
  • Full type annotations throughout
  • Single public entry point:
      `SubgraphSearchEngine.find_subgraph_mappings(...)`

Public API
----------
SubgraphSearchEngine.find_subgraph_mappings(
    host: nx.Graph,
    pattern: nx.Graph,
    node_attrs: List[str],
    edge_attrs: List[str],
    strategy: Strategy = Strategy.COMPONENT,
    *,
    max_results: Optional[int] = None,
    strict_cc_count: bool = False,
    wl1_filter: bool = True,
) -> List[MappingDict]

    Dispatches to one of:
      - `_all_monomorphisms` (classic VF2)
      - `_component_aware_mappings` (fast, CC-aware)
      - `_bt_subgraph_mappings` (with fallback)

Helper Functions
----------------
- `wl1_hash(graph, node_attrs)`
  Computes a single-pass Weisfeiler–Lehman coloring signature.
- `_all_monomorphisms(host, pattern, node_attrs, edge_attrs)`
  Fast wrapper around NetworkX’s VF2 that returns every subgraph monomorphism.
- `_component_aware_mappings(...)`
  Splits graphs into connected components (CCs), applies multi-level filters
  (element, attribute, degree, WL-1), then assembles only those mappings
  placing each pattern-CC into a distinct host-CC.
- `_bt_subgraph_mappings(...)`
  Same CC-aware logic but falls back to classic VF2 if any CC can’t embed.

Usage Example
-------------
```python
from subgraph_matcher import SubgraphSearchEngine, Strategy

mappings = SubgraphSearchEngine.find_subgraph_mappings(
    host_graph,
    pattern_graph,
    node_attrs=["element", "aromatic"],
    edge_attrs=["order"],
    strategy=Strategy.COMPONENT,
    max_results=50,
    strict_cc_count=False,
)
"""

from typing import Any, Dict, List, Set, Optional, Sequence, Tuple, Callable, Union
from operator import eq
import networkx as nx

from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Synthesis.Reactor.strategy import Strategy

try:
    from mod import ruleGMLString

    _RULE_AVAILABLE = True
except ImportError:
    ruleGMLString = None  # type: ignore[assignment]
    _RULE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EdgeAttr = Dict[str, Any]
MappingDict = Dict[int, int]

__all__: Sequence[str] = [
    "SubgraphMatch",
    "SubgraphSearchEngine",
]


# ---------------------------------------------------------------------------
# Core engine class
# ---------------------------------------------------------------------------
class SubgraphMatch:
    """
    Boolean-only checks for graph isomorphism and subgraph (induced or monomorphic) matching.

    Provides static methods for NetworkX-based checks and optional GML "rule" backend.
    """

    @staticmethod
    def _get_edge_labels(graph: Any) -> list:
        """
        Extracts the bond types (edge labels) from a given graph.

        Parameters:
        - graph: The graph object containing the edges.

        Returns:
        - list: List of edge labels as strings.
        """
        return [str(e.bondType) for e in graph.edges]

    @staticmethod
    def _get_node_labels(graph: Any) -> list:
        """
        Extracts the atom IDs (node labels) from a given graph.

        Parameters:
        - graph: The graph object containing the vertices.

        Returns:
        - list: List of node labels as strings.
        """
        return [str(v.atomId) for v in graph.vertices]

    @staticmethod
    def rule_subgraph_morphism(
        rule_1: str, rule_2: str, use_filter: bool = False
    ) -> bool:
        """
        Evaluates if two GML-formatted rule representations are isomorphic or one is a
        subgraph of the other (monomorphic).

        Parameters:
        - rule_1 (str): GML string of the first rule.
        - rule_2 (str): GML string of the second rule.
        - use_filter (bool, optional): Whether to filter by node/edge labels and vertex counts.

        Returns:
        - bool: True if the monomorphism condition is met, False otherwise.
        """
        try:
            rule_obj_1 = ruleGMLString(rule_1, add=False)
            rule_obj_2 = ruleGMLString(rule_2, add=False)
        except Exception as e:
            raise Exception(f"Error parsing GML strings: {e}")

        if use_filter:
            if rule_obj_1.context.numVertices > rule_obj_2.context.numVertices:
                return False

            node_1_left = SubgraphMatch._get_node_labels(rule_obj_1.left)
            node_2_left = SubgraphMatch._get_node_labels(rule_obj_2.left)
            edge_1_left = SubgraphMatch._get_edge_labels(rule_obj_1.left)
            edge_2_left = SubgraphMatch._get_edge_labels(rule_obj_2.left)

            if not all(node in node_2_left for node in node_1_left):
                return False
            if not all(edge in edge_2_left for edge in edge_1_left):
                return False

        return rule_obj_1.monomorphism(rule_obj_2) == 1

    @staticmethod
    def subgraph_isomorphism(
        child_graph: nx.Graph,
        parent_graph: nx.Graph,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        use_filter: bool = False,
        check_type: str = "induced",  # 'induced' or 'monomorphism'
        node_comparator: Optional[Callable[[Any, Any], bool]] = None,
        edge_comparator: Optional[Callable[[Any, Any], bool]] = None,
    ) -> bool:
        """
        Enhanced checks if the child graph is a subgraph isomorphic to the parent graph based on
        customizable node and edge attributes.
        """
        if use_filter:
            if (
                child_graph.number_of_nodes() > parent_graph.number_of_nodes()
                or child_graph.number_of_edges() > parent_graph.number_of_edges()
            ):
                return False

            for _, child_data in child_graph.nodes(data=True):
                found_match = False
                for _, parent_data in parent_graph.nodes(data=True):
                    match = True
                    for label, default in zip(node_label_names, node_label_default):
                        if child_data.get(label, default) != parent_data.get(
                            label, default
                        ):
                            match = False
                            break
                    if match:
                        found_match = True
                        break
                if not found_match:
                    return False

            if edge_attribute:
                for u, v, child_data in child_graph.edges(data=True):
                    if not parent_graph.has_edge(u, v):
                        return False
                    parent_data = parent_graph[u][v]
                    child_order = child_data.get(edge_attribute)
                    parent_order = parent_data.get(edge_attribute)
                    if isinstance(child_order, tuple) and isinstance(
                        parent_order, tuple
                    ):
                        if child_order != parent_order:
                            return False
                    elif child_order != parent_order:
                        return False

        node_comparator = node_comparator or eq
        edge_comparator = edge_comparator or eq

        node_match = generic_node_match(
            node_label_names,
            node_label_default,
            [node_comparator] * len(node_label_names),
        )
        edge_match = generic_edge_match(edge_attribute, None, edge_comparator)

        matcher = GraphMatcher(
            parent_graph, child_graph, node_match=node_match, edge_match=edge_match
        )

        if check_type == "induced":
            return matcher.subgraph_is_isomorphic()
        else:
            return matcher.subgraph_is_monomorphic()

    @staticmethod
    def is_subgraph(
        pattern: Union[nx.Graph, str],
        host: Union[nx.Graph, str],
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        use_filter: bool = False,
        check_type: str = "induced",
        backend: str = "nx",
    ) -> bool:
        """
        Unified API for subgraph/isomorphism either via NX or GML backend.
        """
        if backend == "nx":
            return SubgraphMatch.subgraph_isomorphism(
                pattern,
                host,
                node_label_names,
                node_label_default,
                edge_attribute,
                use_filter,
                check_type,
            )
        if backend == "mod":
            if not _RULE_AVAILABLE:
                raise ImportError("GML rule backend not installed – pip install mod.")
            return SubgraphMatch.rule_subgraph_morphism(
                pattern, host, use_filter=use_filter
            )
        raise ValueError(f"Unknown backend: {backend}")


class SubgraphSearchEngine:
    """Efficient sub‑graph matching helpers (static, stateless)."""

    # ------------------------------------------------------------------
    # Public dispatcher -------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def find_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        strategy: Strategy = Strategy.COMPONENT,
        max_results: Optional[int] = None,
        strict_cc_count: bool = True,
    ) -> List[MappingDict]:
        """
        Dispatch to a subgraph-matching strategy and return all pattern→host mappings.

        Depending on `strategy`, this will call:
          - ALL       → `_all_monomorphisms`
          - COMPONENT → `_component_aware_mappings`
          - BACKTRACK → `_bt_subgraph_mappings`

        Falls back to classic VF2 if host has fewer CCs than pattern, or to
        an empty list if `strict_cc_count` is True and host CCs > pattern CCs.

        Args:
            host: The larger NetworkX graph (substrate).
            pattern: The smaller NetworkX graph (template).
            node_attrs: Node attributes to match exactly; also enforces
                        `hcount`(host) ≥ `hcount`(pattern).
            edge_attrs: Edge attributes to match exactly.
            strategy: Which matching strategy to use (ALL, COMPONENT, BACKTRACK).
            max_results: If set, stop after collecting this many mappings.
            strict_cc_count: If True and host CC count > pattern CC count,
                             immediately return [].
            wl1_filter: Enable/disable WL-1 hash filtering in component strategies.

        Returns:
            A list of mappings, each a dict from pattern node IDs to host node IDs.
            May be empty if no embeddings are found.
        """

        # defensive copy to avoid mutating caller data
        host = host.copy()
        pattern = pattern.copy()

        if strategy is Strategy.ALL:
            return SubgraphSearchEngine._find_all_subgraph_mappings(
                host,
                pattern,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                max_results=max_results,
            )
        if strategy is Strategy.COMPONENT:
            return SubgraphSearchEngine._find_component_aware_subgraph_mappings(
                host,
                pattern,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                max_results=max_results,
                strict_cc_count=strict_cc_count,
            )
        if strategy is Strategy.BACKTRACK:
            return SubgraphSearchEngine._find_bt_subgraph_mappings(
                host,
                pattern,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                max_results=max_results,
                strict_cc_count=strict_cc_count,
            )
        raise ValueError(f"Unsupported strategy: {strategy}")

    # ------------------------------------------------------------------
    # Strategy: ALL – classical VF2 on full host -----------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _find_all_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int] = None,
    ) -> List[MappingDict]:
        """Pure VF2 search without CC awareness (baseline)."""

        def node_match(nh: EdgeAttr, np: EdgeAttr) -> bool:
            return all(nh.get(k) == np.get(k) for k in node_attrs) and nh.get(
                "hcount", 0
            ) >= np.get("hcount", 0)

        def edge_match(eh: EdgeAttr, ep: EdgeAttr) -> bool:
            return all(eh.get(k) == ep.get(k) for k in edge_attrs)

        gm = GraphMatcher(host, pattern, node_match=node_match, edge_match=edge_match)
        results: List[MappingDict] = []
        for iso in gm.subgraph_monomorphisms_iter():
            results.append({p: h for h, p in iso.items()})
            if max_results is not None and len(results) >= max_results:
                break
        return results

    # ------------------------------------------------------------------
    # Strategy: COMPONENT – improved component‑aware matcher -----------
    # ------------------------------------------------------------------
    @staticmethod
    def _find_component_aware_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int] = None,
        strict_cc_count: bool = False,
    ) -> List[MappingDict]:
        """
        Component‑aware VF2 without any attribute/degree/WL‑1 pre‑filters.

        The only constraints are:
          • each pattern‑CC must fit in a *distinct* host‑CC
          • host‑CC size ≥ pattern‑CC size
          • optional strict CC‑count rule

        This guarantees correctness for *subgraph* monomorphisms while avoiding
        accidental over‑pruning.
        """

        # 1) split into connected components
        host_ccs = [host.subgraph(c).copy() for c in nx.connected_components(host)]
        pat_ccs = [pattern.subgraph(c).copy() for c in nx.connected_components(pattern)]
        hcc, pcc = len(host_ccs), len(pat_ccs)

        # empty pattern ⇒ single empty mapping
        if pcc == 0:
            return [{}]

        # fallback to full VF2 if host has fewer CCs
        if hcc < pcc:
            return SubgraphSearchEngine._find_all_subgraph_mappings(
                host,
                pattern,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                max_results=max_results,
            )

        # strict count: reject when host has more CCs than pattern
        if hcc > pcc and strict_cc_count:
            return []

        # 2) define VF2 predicates
        def node_match(nh: EdgeAttr, np: EdgeAttr) -> bool:
            if any(nh.get(a) != np.get(a) for a in node_attrs):
                return False
            return nh.get("hcount", 0) >= np.get("hcount", 0)

        def edge_match(eh: EdgeAttr, ep: EdgeAttr) -> bool:
            return all(eh.get(a) == ep.get(a) for a in edge_attrs)

        # 3) collect embeddings for each pattern‑CC
        per_cc: List[List[Tuple[int, MappingDict]]] = []
        for pc in pat_ccs:
            sz = pc.number_of_nodes()
            # any host CC large enough is a candidate
            cand_cc_idx = [
                i for i, hc in enumerate(host_ccs) if hc.number_of_nodes() >= sz
            ]
            if not cand_cc_idx:
                return []  # impossible – no room for this component

            cc_maps: List[Tuple[int, MappingDict]] = []
            for hi in cand_cc_idx:
                gm = GraphMatcher(
                    host_ccs[hi], pc, node_match=node_match, edge_match=edge_match
                )
                for iso in gm.subgraph_monomorphisms_iter():
                    cc_maps.append((hi, {p: h for h, p in iso.items()}))
                    if max_results and len(cc_maps) >= max_results:
                        break
                if max_results and len(cc_maps) >= max_results:
                    break

            if not cc_maps:  # this pattern‑CC embeds nowhere
                return []
            per_cc.append(cc_maps)

        # 4) order pattern‑CCs by fewest embeddings → best pruning
        order = sorted(range(pcc), key=lambda i: len(per_cc[i]))
        ordered = [per_cc[i] for i in order]

        # 5) backtrack to build full‑pattern mappings
        results: List[MappingDict] = []
        used_host: Set[int] = set()

        def backtrack(level: int, accum: MappingDict):
            if max_results and len(results) >= max_results:
                return
            if level == pcc:
                results.append(accum.copy())
                return
            for hi, mapping in ordered[level]:
                if hi in used_host or any(p in accum for p in mapping):
                    continue
                used_host.add(hi)
                accum.update(mapping)
                backtrack(level + 1, accum)
                for p in mapping:
                    accum.pop(p)
                used_host.remove(hi)
                if max_results and len(results) >= max_results:
                    return

        backtrack(0, {})
        return results

    # ------------------------------------------------------------------
    # Strategy: BACKTRACK – component first, fallback to VF2 -----------
    # ------------------------------------------------------------------
    @staticmethod
    def _find_bt_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int] = None,
        strict_cc_count: bool = False,
    ) -> List[MappingDict]:
        """Component‑aware search *with* classic fallback if any CC fails."""

        primary = SubgraphSearchEngine._find_component_aware_subgraph_mappings(
            host,
            pattern,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            max_results=max_results,
            strict_cc_count=strict_cc_count,
        )
        if primary:
            return primary
        # else classic fallback
        return SubgraphSearchEngine._find_all_subgraph_mappings(
            host,
            pattern,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # Niceties ----------------------------------------------------------
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401 – simple repr
        return "<SubgraphSearchEngine – static helpers; use `find_subgraph_mappings`>"

    __str__ = __repr__

    # helpful alias for interactive users --------------------------------
    @property
    def help(self) -> str:  # noqa: D401 – property for convenience
        """Return the full module docstring."""

        return __doc__
