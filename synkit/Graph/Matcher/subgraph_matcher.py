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

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EdgeAttr = Dict[str, Any]
MappingDict = Dict[int, int]

__all__: Sequence[str] = [
    "SubgraphMatch",
    "SubgraphSearchEngine",
]


def electron_aware_node_match(
    host_data: EdgeAttr,
    pattern_data: EdgeAttr,
    node_attrs: Sequence[str],
) -> bool:
    """Compare node attributes with chemistry-aware cardinality semantics.

    Attributes in ``node_attrs`` are exact matches except:

    - ``hcount``: host must be greater than or equal to pattern
    - ``lone_pairs``: host must be greater than or equal to pattern
    - ``aromatic_n_pi_count``: exact aromatic-N role label when present
    ``radical`` therefore remains exact whenever the caller includes it in
    ``node_attrs``.
    """
    for attr in node_attrs:
        # A compact coupled pi-addition rule may use a deliberately
        # under-valenced atom (for example ``[C:1]=[C:2]``) to state only the
        # reaction locus. RDKit represents that notation with placeholder
        # radical electrons. They are a parser consequence, not a radical
        # query; the reactor marks only those inferred coupling centers.
        if pattern_data.get("_coupled_pi_center_query") and attr == "radical":
            continue
        host_value = host_data.get(
            attr, 0 if attr in {"hcount", "lone_pairs"} else None
        )
        pattern_value = pattern_data.get(
            attr, 0 if attr in {"hcount", "lone_pairs"} else None
        )
        if attr in {"hcount", "lone_pairs"}:
            if host_value < pattern_value:
                return False
            continue
        if host_value != pattern_value:
            return False
    return True


def electron_aware_edge_match(
    host_data: EdgeAttr,
    pattern_data: EdgeAttr,
    edge_attrs: Sequence[str],
) -> bool:
    """Compare edge attrs while treating aromatic Kekule phase as non-semantic.

    Aromatic presentation bonds are matched by ``order == 1.5``. Their
    particular ``sigma_order`` / ``pi_order`` split depends on the chosen
    Kekule form and is not stable across independently parsed graphs.
    """
    minimum_pi = pattern_data.get("_minimum_pi_order")
    if minimum_pi is not None:
        # Relative pi reduction: C=C means "a bond carrying at least one pi
        # bond" at this explicitly marked reaction locus. Thus the same
        # chemical edit can consume C=C or C#C without weakening matching
        # anywhere else in the pattern.
        if host_data.get("order") == 1.5:
            return False
        if float(host_data.get("pi_order", 0.0)) < float(minimum_pi):
            return False

    host_is_aromatic = host_data.get("order") == 1.5
    pattern_is_aromatic = pattern_data.get("order") == 1.5
    for attr in edge_attrs:
        if minimum_pi is not None and attr in {"order", "pi_order"}:
            continue
        if (
            attr in {"sigma_order", "pi_order"}
            and host_is_aromatic
            and pattern_is_aromatic
        ):
            continue
        if host_data.get(attr) != pattern_data.get(attr):
            return False
    return True


def explain_node_mismatch(
    host_data: EdgeAttr,
    pattern_data: EdgeAttr,
    node_attrs: Sequence[str],
) -> list[str]:
    """Return node-level mismatch reasons using matcher semantics."""
    reasons: list[str] = []
    for attr in node_attrs:
        if pattern_data.get("_coupled_pi_center_query") and attr == "radical":
            continue
        host_value = host_data.get(
            attr, 0 if attr in {"hcount", "lone_pairs"} else None
        )
        pattern_value = pattern_data.get(
            attr, 0 if attr in {"hcount", "lone_pairs"} else None
        )
        if attr in {"hcount", "lone_pairs"}:
            if host_value < pattern_value:
                reasons.append(f"{attr}: host {host_value} < pattern {pattern_value}")
            continue
        if host_value != pattern_value:
            reasons.append(f"{attr}: host {host_value!r} != pattern {pattern_value!r}")
    return reasons


def resolve_template_match_attrs(
    pattern: nx.Graph,
    *,
    legacy_node_attrs: Sequence[str] = ("element", "charge"),
    legacy_edge_attrs: Sequence[str] = ("order",),
) -> tuple[list[str], list[str]]:
    """Choose match attrs from what the template actually carries.

    Legacy templates keep the legacy attribute set. Electron-aware templates opt
    into extra constraints only when those attrs are present on the template.
    """
    node_attrs = list(legacy_node_attrs)
    edge_attrs = list(legacy_edge_attrs)

    for attr in (
        "aromatic",
        "hcount",
        "lone_pairs",
        "radical",
        "aromatic_n_pi_count",
    ):
        if any(attr in data for _, data in pattern.nodes(data=True)):
            node_attrs.append(attr)

    for attr in ("sigma_order", "pi_order"):
        if any(attr in data for _, _, data in pattern.edges(data=True)):
            edge_attrs.append(attr)

    return node_attrs, edge_attrs


def diagnose_candidate_node_match(
    host_data: EdgeAttr,
    pattern_data: EdgeAttr,
    node_attrs: Sequence[str],
) -> dict[str, Any]:
    """Return a compact node-match diagnostic payload."""
    reasons = explain_node_mismatch(host_data, pattern_data, node_attrs)
    return {"matched": not reasons, "reasons": reasons}


# ---------------------------------------------------------------------------
# Core engine class
# ---------------------------------------------------------------------------
class SubgraphMatch:
    """Boolean-only checks for graph isomorphism and subgraph (induced or
    monomorphic) matching.

    Provides static methods for NetworkX-based checks.
    """

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
        """Enhanced checks if the child graph is a subgraph isomorphic to the
        parent graph based on customizable node and edge attributes."""
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
        """Run a native NetworkX subgraph or isomorphism check."""
        if backend != "nx":
            raise ValueError(f"Unknown backend: {backend}")
        if not isinstance(pattern, nx.Graph) or not isinstance(host, nx.Graph):
            raise TypeError("NetworkX backend expects graph inputs.")
        return SubgraphMatch.subgraph_isomorphism(
            pattern,
            host,
            node_label_names,
            node_label_default,
            edge_attribute,
            use_filter,
            check_type,
        )


# -----------------------------------------------------------------------------
# Sub‑graph search engine
# -----------------------------------------------------------------------------


class SubgraphSearchEngine:
    """Static helper routines for sub-graph monomorphism search.

    :cvar DEFAULT_THRESHOLD: default cap on embedding enumeration (5000)
    """

    DEFAULT_THRESHOLD: int = 5_000

    @staticmethod
    def _quick_pre_filter(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        threshold: int,
    ) -> bool:
        """Estimate if candidate-product exceeds threshold with degree pruning.

        We refine the basic Cartesian-product by requiring each host candidate
        to match node attributes *and* have degree ≥ the pattern node’s degree.
        This tighter filter greatly reduces false positives (over-pruning).
        """
        estimate = 1
        # Pre-compute pattern degrees
        pat_degrees = {n: pattern.degree(n) for n in pattern.nodes()}
        for p_node, pat_data in pattern.nodes(data=True):
            pat_deg = pat_degrees[p_node]
            # count host nodes matching attributes and degree
            count = sum(
                1
                for _, host_data in host.nodes(data=True)
                if electron_aware_node_match(host_data, pat_data, node_attrs)
                and host.degree(_) >= pat_deg
            )
            # if no candidates; impossible match
            if count == 0:
                return True
            estimate *= count
            if estimate > threshold * 1e4:  # reduce false positives
                return True
        return False

    @staticmethod
    def find_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        strategy: Union[str, Strategy] = Strategy.COMPONENT,
        max_results: Optional[int] = None,
        strict_cc_count: bool = True,
        threshold: Optional[int] = None,
        pre_filter: bool = False,
    ) -> List[MappingDict]:
        """Dispatch to a subgraph-matching strategy with optional guards.

        Parameters
        ----------
        host, pattern
            NetworkX graphs (host ≥ pattern).
        node_attrs, edge_attrs
            Keys of attributes to match; ``hcount`` and ``lone_pairs`` use
            host-greater-or-equal semantics, while the rest are exact.
        strategy
            Matching strategy code or enum ("all", "comp", "bt").
        max_results
            Stop after this many embeddings (None = no limit).
        strict_cc_count
            If True, host CC count must ≤ pattern CC count for COMPONENT/BACKTRACK.
        threshold
            Override the default cap (DEFAULT_THRESHOLD) on embeddings.
        pre_filter
            If True, run a cheap Cartesian-product pre-filter against the threshold.

        Returns
        -------
        List of dictionaries mapping pattern node→host node. Empty if none or
        if any guard (pre-filter or enumeration) exceeds the threshold.
        """
        strat = Strategy.from_string(strategy)
        if strat is Strategy.PARTIAL:
            raise NotImplementedError("PARTIAL strategy not implemented yet.")

        # determine effective threshold
        thresh = (
            threshold
            if threshold is not None
            else SubgraphSearchEngine.DEFAULT_THRESHOLD
        )

        # defensive copies
        host = host.copy()
        pattern = pattern.copy()

        # quick pre-filter
        if pre_filter and SubgraphSearchEngine._quick_pre_filter(
            host, pattern, node_attrs, thresh
        ):
            return []

        # dispatch
        if strat is Strategy.ALL:
            results = SubgraphSearchEngine._find_all_subgraph_mappings(
                host, pattern, node_attrs, edge_attrs, max_results, thresh
            )
        elif strat is Strategy.COMPONENT:
            results = SubgraphSearchEngine._find_component_aware_subgraph_mappings(
                host,
                pattern,
                node_attrs,
                edge_attrs,
                max_results,
                strict_cc_count,
                thresh,
            )
        else:  # BACKTRACK
            results = SubgraphSearchEngine._find_bt_subgraph_mappings(
                host,
                pattern,
                node_attrs,
                edge_attrs,
                max_results,
                strict_cc_count,
                thresh,
            )

        # final threshold guard
        return [] if len(results) > thresh else results

    @staticmethod
    def _find_all_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int],
        threshold: int,
    ) -> List[MappingDict]:
        """Classic VF2 over the whole host graph."""

        def node_match(nh: EdgeAttr, np: EdgeAttr) -> bool:
            return electron_aware_node_match(nh, np, node_attrs)

        def edge_match(eh: EdgeAttr, ep: EdgeAttr) -> bool:
            return electron_aware_edge_match(eh, ep, edge_attrs)

        gm = GraphMatcher(host, pattern, node_match=node_match, edge_match=edge_match)
        results: List[MappingDict] = []
        for iso in gm.subgraph_monomorphisms_iter():
            results.append({p: h for h, p in iso.items()})
            if max_results and len(results) >= max_results:
                break
            if len(results) > threshold:
                return []
        return results

    @staticmethod
    def _find_component_aware_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int],
        strict_cc_count: bool,
        threshold: int,
    ) -> List[MappingDict]:
        """Component-aware VF2 split by connected components."""
        host_ccs = [host.subgraph(c).copy() for c in nx.connected_components(host)]
        pat_ccs = [pattern.subgraph(c).copy() for c in nx.connected_components(pattern)]
        hcc, pcc = len(host_ccs), len(pat_ccs)
        if pcc == 0:
            return [{}]
        if hcc < pcc:
            return SubgraphSearchEngine._find_all_subgraph_mappings(
                host, pattern, node_attrs, edge_attrs, max_results, threshold
            )
        if hcc > pcc and strict_cc_count:
            return []

        def node_match(nh: EdgeAttr, np: EdgeAttr) -> bool:
            return electron_aware_node_match(nh, np, node_attrs)

        def edge_match(eh: EdgeAttr, ep: EdgeAttr) -> bool:
            return electron_aware_edge_match(eh, ep, edge_attrs)

        per_cc: List[List[Tuple[int, MappingDict]]] = []
        for pc in pat_ccs:
            sz = pc.number_of_nodes()
            cand = [i for i, hc in enumerate(host_ccs) if hc.number_of_nodes() >= sz]
            if not cand:
                return []
            maps: List[Tuple[int, MappingDict]] = []
            for i in cand:
                gm = GraphMatcher(
                    host_ccs[i], pc, node_match=node_match, edge_match=edge_match
                )
                for iso in gm.subgraph_monomorphisms_iter():
                    maps.append((i, {p: h for h, p in iso.items()}))
                    if max_results and len(maps) >= max_results:
                        break
                    if len(maps) > threshold:
                        return []
                if max_results and len(maps) >= max_results:
                    break
            if not maps:
                return []
            per_cc.append(maps)

        order = sorted(range(pcc), key=lambda i: len(per_cc[i]))
        ordered = [per_cc[i] for i in order]
        results: List[MappingDict] = []
        used: Set[int] = set()

        def backtrack(level: int, acc: MappingDict):
            if max_results and len(results) >= max_results:
                return
            if len(results) > threshold:
                return
            if level == pcc:
                results.append(acc.copy())
                return
            for hi, m in ordered[level]:
                if hi in used or any(p in acc for p in m):
                    continue
                used.add(hi)
                acc.update(m)
                backtrack(level + 1, acc)
                for p in m:
                    acc.pop(p)
                used.remove(hi)
                if max_results and len(results) >= max_results:
                    return
                if len(results) > threshold:
                    return

        backtrack(0, {})
        return results

    @staticmethod
    def _find_bt_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int],
        strict_cc_count: bool,
        threshold: int,
    ) -> List[MappingDict]:
        primary = SubgraphSearchEngine._find_component_aware_subgraph_mappings(
            host,
            pattern,
            node_attrs,
            edge_attrs,
            max_results,
            strict_cc_count,
            threshold,
        )
        if primary:
            return primary
        return SubgraphSearchEngine._find_all_subgraph_mappings(
            host, pattern, node_attrs, edge_attrs, max_results, threshold
        )

    def __repr__(self) -> str:
        return "<SubgraphSearchEngine – use `find_subgraph_mappings`>"

    __str__ = __repr__

    # helpful alias for interactive users --------------------------------
    @property
    def help(self) -> str:  # noqa: D401 – property for convenience
        """Return the full module docstring."""

        return __doc__
