# from __future__ import annotations

# """
# approx_mcs_matcher.py — Heuristic / Approximate MCS Matcher
# ===========================================================

# A compact, OOP-style wrapper that mimics the public behaviour of
# :class:`MCSMatcher`, but uses a **greedy, heuristic** search instead of
# exhaustive subgraph enumeration.

# Key ideas
# ---------
# * Use :func:`generic_node_match` for flexible node-attribute comparison.
# * Score node pairs based on attribute compatibility and local degrees.
# * Select a small set of high-scoring **seed pairs**.
# * For each seed, **greedily grow** a mapping along neighbouring nodes,
#   enforcing local edge compatibility along the way.
# * Cache the resulting (approximate) mappings in the same orientation
#   scheme as :class:`MCSMatcher`:

#   - Internal cache: **pattern → host** (pattern = smaller graph after
#     optional wildcard pruning).
#   - :py:meth:`get_mappings` converts to ``G1→G2`` or ``G2→G1``.

# Public API
# ~~~~~~~~~~
# ``ApproxMCSMatcher(node_attrs=None,
#                   node_defaults=None,
#                   allow_shift=True,
#                   edge_attrs=None,
#                   prune_wc=False,
#                   prune_automorphisms=False,
#                   wildcard_element='*',
#                   element_key='element')``
#     Construct a matcher instance.

# ``matcher.find_common_subgraph(G1, G2,
#                                mcs=False,
#                                mcs_mol=False,
#                                max_seeds=16,
#                                max_steps=256)``
#     Run an **approximate** search (stores and returns ``self``). The
#     signature mirrors :class:`MCSMatcher`, but the implementation is
#     greedy/heuristic rather than exhaustive.

# ``matcher.find_common_subgraph_approx(G1, G2,
#                                       max_seeds=16,
#                                       max_steps=256)``
#     Lightweight alias that exposes only the approximation parameters.

# ``matcher.find_rc_mapping(rc1, rc2,
#                           side='op',
#                           mcs=True,
#                           mcs_mol=False,
#                           component=True,
#                           max_seeds=16,
#                           max_steps=256)``
#     Convenience wrapper for ITS reaction-centre or ITS-like graph
#     objects (via :func:`synkit.Graph.ITS.its_decompose` when available),
#     analogous to :py:meth:`MCSMatcher.find_rc_mapping` but using the
#     heuristic search internally.

# ``matcher.get_mappings(direction='pattern_to_host')``
#     Retrieve the stored mapping list. ``direction`` can be one of
#     ``"pattern_to_host"``, ``"G1_to_G2"``, ``"G2_to_G1"``.

# ``matcher.mappings``
#     Shorthand for ``get_mappings(direction='pattern_to_host')``.

# ``matcher.mapping_direction``
#     String describing internal orientation: ``"G1_to_G2"``, ``"G2_to_G1"``,
#     or ``"unknown"`` if no search has been run yet.

# As with :class:`MCSMatcher`, all public methods return ``self`` to
# support a fluent, chainable style, and simple niceties such as
# :py:meth:`__repr__` and :pyattr:`help` are provided.
# """

# from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# import networkx as nx
# from networkx.algorithms.isomorphism import generic_node_match

# try:  # pragma: no cover - optional dependency
#     from synkit.Graph.ITS import its_decompose
# except ImportError:  # pragma: no cover
#     its_decompose = None  # type: ignore[assignment]

# __all__ = ["ApproxMCSMatcher"]

# MappingDict = Dict[int, int]


# class ApproxMCSMatcher:
#     """
#     Heuristic / approximate common-subgraph matcher.

#     This class provides a **fast, approximate** alternative to
#     :class:`MCSMatcher`. It does *not* enumerate all subgraph
#     isomorphisms. Instead, it:

#     1. Picks a set of high-scoring node pairs as **seeds** based on
#        attribute and local-structure similarity.
#     2. For each seed, **greedily grows** a subgraph isomorphism by
#        extending along neighbouring nodes while maintaining local
#        adjacency consistency.
#     3. Collects the resulting partial mappings and reports them in the
#        same orientation style as :class:`MCSMatcher`.

#     The result is an approximate maximum-common-subgraph (MCS) mapping
#     that is usually close to optimal for molecular graphs, but can be
#     computed much faster and can be bounded by simple iteration
#     parameters.

#     Orientation and storage
#     -----------------------
#     Internally, mappings are stored in **pattern → host** orientation,
#     where the pattern is the smaller of the two graphs after optional
#     wildcard pruning. The :py:meth:`get_mappings` helper converts these
#     into ``G1→G2`` or ``G2→G1`` orientation based on which original
#     graph served as the pattern.

#     API compatibility
#     -----------------
#     The constructor and public methods mirror :class:`MCSMatcher` as
#     closely as possible so that :class:`ApproxMCSMatcher` can drop in as
#     a faster, heuristic substitute in many workflows.

#     :param node_attrs: Node attribute keys to compare. If ``None``,
#         defaults to ``["element"]``.
#     :type node_attrs: list[str] | None
#     :param node_defaults: Fallback values for each node attribute when
#         missing. If ``None``, defaults to a list of ``"*"``
#         with the same length as :paramref:`node_attrs`.
#     :type node_defaults: list[Any] | None
#     :param allow_shift: Placeholder for future asymmetric rules. Kept
#         for API compatibility with :class:`MCSMatcher`.
#     :type allow_shift: bool
#     :param edge_attrs: Edge attribute keys to use for scalar comparison
#         (e.g. ``["order"]``). If ``None``, defaults to ``["order"]``.
#     :type edge_attrs: list[str] | None
#     :param prune_wc: If ``True``, strip wildcard nodes (see
#         :paramref:`wildcard_element`, :paramref:`element_key`) from both
#         graphs before searching.
#     :type prune_wc: bool
#     :param prune_automorphisms: If ``True``, collapse mappings that
#         have the same host node set (automorphism pruning).
#     :type prune_automorphisms: bool
#     :param wildcard_element: Attribute value denoting wildcard nodes
#         (typically ``"*"``, used together with :paramref:`element_key`).
#     :type wildcard_element: Any
#     :param element_key: Node attribute key used to detect wildcard nodes
#         when :paramref:`prune_wc` is ``True``.
#     :type element_key: str
#     """

#     # ------------------------------------------------------------------
#     # Construction
#     # ------------------------------------------------------------------
#     def __init__(
#         self,
#         node_attrs: Optional[List[str]] = None,
#         node_defaults: Optional[List[Any]] = None,
#         allow_shift: bool = True,
#         *,
#         edge_attrs: Optional[List[str]] = None,
#         prune_wc: bool = False,
#         prune_automorphisms: bool = False,
#         wildcard_element: Any = "*",
#         element_key: str = "element",
#     ) -> None:
#         if node_attrs is None:
#             node_attrs = ["element"]
#         if node_defaults is None:
#             node_defaults = ["*"] * len(node_attrs)
#         if len(node_defaults) != len(node_attrs):
#             raise ValueError(
#                 "ApproxMCSMatcher: node_defaults must have the same length "
#                 "as node_attrs."
#             )

#         self._node_attrs: List[str] = node_attrs
#         self._node_defaults: List[Any] = node_defaults
#         self._edge_attrs: List[str] = edge_attrs or ["order"]
#         self.allow_shift: bool = allow_shift

#         self.prune_wc: bool = prune_wc
#         self.prune_automorphisms: bool = prune_automorphisms
#         self.wildcard_element: Any = wildcard_element
#         self.element_key: str = element_key

#         comparators: List[Callable[[Any, Any], bool]] = [
#             (lambda x, y: x == y) for _ in node_attrs
#         ]
#         self.node_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] = (
#             generic_node_match(
#                 node_attrs,
#                 node_defaults,
#                 comparators,
#             )
#         )

#         # Internal cache (pattern → host)
#         self._mappings: List[MappingDict] = []
#         self._last_size: int = 0
#         self._last_pattern_is_G1: Optional[bool] = None

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------
#     def _prune_graph(self, G: nx.Graph) -> nx.Graph:
#         """
#         Remove wildcard nodes from ``G`` (non-inplace) if applicable.

#         When :pyattr:`prune_wc` is ``False``, the input graph is returned
#         as-is.

#         :param G: Input graph.
#         :type G: nx.Graph
#         :returns: Possibly pruned graph.
#         :rtype: nx.Graph
#         """
#         if not self.prune_wc:
#             return G

#         key = self.element_key
#         wc = self.wildcard_element
#         keep_nodes = [n for n, d in G.nodes(data=True) if d.get(key) != wc]
#         return G.subgraph(keep_nodes).copy()

#     @staticmethod
#     def _prepare_orientation(
#         G1: nx.Graph,
#         G2: nx.Graph,
#     ) -> Tuple[nx.Graph, nx.Graph, bool]:
#         """
#         Ensure the smaller graph is used as pattern.

#         :param G1: First input graph.
#         :type G1: nx.Graph
#         :param G2: Second input graph.
#         :type G2: nx.Graph
#         :returns: Tuple ``(pattern, host, pattern_is_G1)``.
#         :rtype: tuple[nx.Graph, nx.Graph, bool]
#         """
#         if G1.number_of_nodes() <= G2.number_of_nodes():
#             return G1, G2, True
#         return G2, G1, False

#     @staticmethod
#     def _invert_mapping(mapping: MappingDict) -> MappingDict:
#         """
#         Invert a mapping from host→pattern to pattern→host or vice versa.

#         :param mapping: Mapping to invert.
#         :type mapping: dict[int, int]
#         :returns: Inverted mapping.
#         :rtype: dict[int, int]
#         """
#         return {v: k for k, v in mapping.items()}

#     def _edge_match(
#         self,
#         attrs1: Dict[str, Any],
#         attrs2: Dict[str, Any],
#     ) -> bool:
#         """
#         Compare edge attributes listed in :pyattr:`_edge_attrs`.

#         For each attribute name:

#         * If both values can be cast to ``float``, use numeric equality.
#         * Otherwise, fall back to direct ``==`` comparison.
#         * Missing values on both sides are ignored.

#         :param attrs1: Edge attributes of the first edge.
#         :type attrs1: dict[str, Any]
#         :param attrs2: Edge attributes of the second edge.
#         :type attrs2: dict[str, Any]
#         :returns: ``True`` if the attributes are compatible.
#         :rtype: bool
#         """
#         for name in self._edge_attrs:
#             v1 = attrs1.get(name)
#             v2 = attrs2.get(name)
#             if v1 is None and v2 is None:
#                 continue
#             try:
#                 if float(v1) != float(v2):
#                     return False
#             except (TypeError, ValueError):
#                 if v1 != v2:
#                     return False
#         return True

#     def _node_similarity(
#         self,
#         p: int,
#         h: int,
#         pattern: nx.Graph,
#         host: nx.Graph,
#     ) -> float:
#         """
#         Compute a simple structural similarity score for a node pair.

#         The score is based on:

#         * Node attribute compatibility via :pyattr:`node_match`.
#         * Degree difference between ``p`` and ``h``.
#         * Overlap in neighbour degrees.

#         :param p: Node id in the pattern graph.
#         :type p: int
#         :param h: Node id in the host graph.
#         :type h: int
#         :param pattern: Pattern graph.
#         :type pattern: nx.Graph
#         :param host: Host graph.
#         :type host: nx.Graph
#         :returns: Similarity score (larger is better; negative for
#             incompatible pairs).
#         :rtype: float
#         """
#         pdata = pattern.nodes[p]
#         hdata = host.nodes[h]
#         if not self.node_match(pdata, hdata):
#             return -1.0

#         deg_p = pattern.degree[p]
#         deg_h = host.degree[h]
#         base = 2.0
#         penalty = float(abs(deg_p - deg_h))

#         neigh_deg_p = {pattern.degree[n] for n in pattern.neighbors(p)}
#         neigh_deg_h = {host.degree[n] for n in host.neighbors(h)}
#         overlap = float(len(neigh_deg_p & neigh_deg_h))

#         return base - penalty + 0.5 * overlap

#     def _generate_seeds(
#         self,
#         pattern: nx.Graph,
#         host: nx.Graph,
#         max_seeds: int,
#     ) -> List[Tuple[int, int]]:
#         """
#         Generate a list of high-scoring seed node pairs.

#         :param pattern: Pattern graph.
#         :type pattern: nx.Graph
#         :param host: Host graph.
#         :type host: nx.Graph
#         :param max_seeds: Maximum number of seed pairs to keep.
#         :type max_seeds: int
#         :returns: List of ``(pattern_node, host_node)`` pairs.
#         :rtype: list[tuple[int, int]]
#         """
#         candidates: List[Tuple[float, int, int]] = []
#         for p in pattern.nodes():
#             for h in host.nodes():
#                 score = self._node_similarity(p, h, pattern, host)
#                 if score <= 0.0:
#                     continue
#                 candidates.append((score, p, h))

#         candidates.sort(reverse=True, key=lambda t: t[0])
#         top = candidates[:max_seeds]
#         return [(p, h) for _, p, h in top]

#     def _respects_edges(
#         self,
#         p: int,
#         h: int,
#         pattern: nx.Graph,
#         host: nx.Graph,
#         mapping: MappingDict,
#     ) -> bool:
#         """
#         Check whether extending mapping with ``p → h`` is locally valid.

#         This ensures that for any already-mapped neighbour ``p_n`` of
#         ``p``, the candidate host node ``h`` is adjacent to the mapped
#         host node and that edge attributes are compatible.

#         :param p: Pattern node to extend with.
#         :type p: int
#         :param h: Candidate host node.
#         :type h: int
#         :param pattern: Pattern graph.
#         :type pattern: nx.Graph
#         :param host: Host graph.
#         :type host: nx.Graph
#         :param mapping: Current partial mapping (pattern→host).
#         :type mapping: dict[int, int]
#         :returns: ``True`` if the extension is feasible.
#         :rtype: bool
#         """
#         for p_n, h_n in mapping.items():
#             if not pattern.has_edge(p, p_n):
#                 continue
#             if not host.has_edge(h, h_n):
#                 return False
#             attrs_p = pattern[p][p_n]
#             attrs_h = host[h][h_n]
#             if not self._edge_match(attrs_p, attrs_h):
#                 return False
#         return True

#     def _candidate_hosts(
#         self,
#         p: int,
#         pattern: nx.Graph,
#         host: nx.Graph,
#         mapping: MappingDict,
#     ) -> List[int]:
#         """
#         Enumerate host nodes that can be matched to pattern node ``p``.

#         :param p: Pattern node.
#         :type p: int
#         :param pattern: Pattern graph.
#         :type pattern: nx.Graph
#         :param host: Host graph.
#         :type host: nx.Graph
#         :param mapping: Current partial mapping (pattern→host).
#         :type mapping: dict[int, int]
#         :returns: List of feasible host node ids.
#         :rtype: list[int]
#         """
#         mapped_hosts = set(mapping.values())
#         pdata = pattern.nodes[p]
#         candidates: List[int] = []

#         for h in host.nodes():
#             if h in mapped_hosts:
#                 continue
#             hdata = host.nodes[h]
#             if not self.node_match(pdata, hdata):
#                 continue
#             if not self._respects_edges(p, h, pattern, host, mapping):
#                 continue
#             candidates.append(h)

#         return candidates

#     def _grow_from_seed(
#         self,
#         pattern: nx.Graph,
#         host: nx.Graph,
#         seed_p: int,
#         seed_h: int,
#         max_steps: int,
#     ) -> MappingDict:
#         """
#         Greedily grow a subgraph mapping starting from a single seed.

#         :param pattern: Pattern graph.
#         :type pattern: nx.Graph
#         :param host: Host graph.
#         :type host: nx.Graph
#         :param seed_p: Seed node in pattern graph.
#         :type seed_p: int
#         :param seed_h: Seed node in host graph.
#         :type seed_h: int
#         :param max_steps: Maximum number of growth steps.
#         :type max_steps: int
#         :returns: Completed partial mapping (pattern→host).
#         :rtype: dict[int, int]
#         """
#         mapping: MappingDict = {seed_p: seed_h}
#         frontier: Set[int] = {n for n in pattern.neighbors(seed_p)}
#         steps = 0

#         while frontier and steps < max_steps:
#             steps += 1
#             # pick a node with largest degree in pattern
#             p = max(frontier, key=pattern.degree.__getitem__)
#             frontier.remove(p)

#             candidates = self._candidate_hosts(p, pattern, host, mapping)
#             if not candidates:
#                 continue

#             best_h = max(
#                 candidates,
#                 key=lambda h: self._node_similarity(p, h, pattern, host),
#             )
#             mapping[p] = best_h

#             for q in pattern.neighbors(p):
#                 if q in mapping or q in frontier:
#                     continue
#                 frontier.add(q)

#         return mapping

#     def _approximate_mappings(
#         self,
#         pattern: nx.Graph,
#         host: nx.Graph,
#         max_seeds: int,
#         max_steps: int,
#     ) -> Tuple[List[MappingDict], int]:
#         """
#         Core heuristic search between ``pattern`` and ``host``.

#         This is factored out so that it can be reused for whole-graph,
#         component-wise, and reaction-centre searches.

#         :param pattern: Pattern graph (smaller or equal).
#         :type pattern: nx.Graph
#         :param host: Host graph (larger or equal).
#         :type host: nx.Graph
#         :param max_seeds: Maximum number of seed node pairs to explore.
#         :type max_seeds: int
#         :param max_steps: Maximum number of growth steps per seed.
#         :type max_steps: int
#         :returns: Tuple ``(mappings, best_size)`` where ``mappings`` are
#             pattern→host and ``best_size`` is the size of the largest
#             mapping found.
#         :rtype: tuple[list[dict[int, int]], int]
#         """
#         seeds = self._generate_seeds(pattern, host, max_seeds=max_seeds)
#         if not seeds:
#             return [], 0

#         best_size = 0
#         all_maps: List[MappingDict] = []

#         for seed_p, seed_h in seeds:
#             mapping = self._grow_from_seed(
#                 pattern,
#                 host,
#                 seed_p,
#                 seed_h,
#                 max_steps=max_steps,
#             )
#             all_maps.append(mapping)
#             size = len(mapping)
#             if size > best_size:
#                 best_size = size

#         if self.prune_automorphisms:
#             filtered: List[MappingDict] = []
#             seen_host_sets: Set[frozenset[int]] = set()
#             # Sort by descending mapping size to keep best representatives
#             all_maps.sort(key=lambda d: (-len(d), tuple(sorted(d.items()))))
#             for mp in all_maps:
#                 hset = frozenset(mp.values())
#                 if hset in seen_host_sets:
#                     continue
#                 seen_host_sets.add(hset)
#                 filtered.append(mp)
#             all_maps = filtered

#         return all_maps, best_size

#     def _componentwise_approx(
#         self,
#         G1: nx.Graph,
#         G2: nx.Graph,
#         *,
#         max_seeds: int,
#         max_steps: int,
#     ) -> MappingDict:
#         """
#         Component-wise approximate matching between ``G1`` and ``G2``.

#         Connected components of each graph are sorted by size
#         (descending) and matched pairwise (largest with largest, etc.).
#         For each pair, an approximate mapping is computed and the best
#         mapping is merged into a combined **G1 → G2** mapping.

#         :param G1: First input graph.
#         :type G1: nx.Graph
#         :param G2: Second input graph.
#         :type G2: nx.Graph
#         :param max_seeds: Maximum number of seed pairs per component pair.
#         :type max_seeds: int
#         :param max_steps: Maximum growth steps per seed.
#         :type max_steps: int
#         :returns: Combined mapping from nodes of ``G1`` to nodes of ``G2``.
#         :rtype: dict[int, int]
#         """
#         comps1 = [G1.subgraph(c).copy() for c in nx.connected_components(G1)]
#         comps2 = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]

#         comps1.sort(key=lambda g: g.number_of_nodes(), reverse=True)
#         comps2.sort(key=lambda g: g.number_of_nodes(), reverse=True)

#         limit = min(len(comps1), len(comps2))
#         combined: MappingDict = {}

#         for idx in range(limit):
#             sub1 = comps1[idx]
#             sub2 = comps2[idx]

#             pattern, host, pattern_is_G1 = self._prepare_orientation(sub1, sub2)
#             maps, _ = self._approximate_mappings(
#                 pattern,
#                 host,
#                 max_seeds=max_seeds,
#                 max_steps=max_steps,
#             )
#             if not maps:
#                 continue

#             best = maps[0]
#             if pattern_is_G1:
#                 combined.update(best)
#             else:
#                 combined.update(self._invert_mapping(best))

#         return combined

#     def _find_mcs_mol_approx(
#         self,
#         G1: nx.Graph,
#         G2: nx.Graph,
#         *,
#         max_seeds: int,
#         max_steps: int,
#     ) -> MappingDict:
#         """
#         Approximate molecule-level (component) matching in G1→G2.

#         This mirrors :py:meth:`MCSMatcher._find_mcs_mol` but uses the
#         heuristic search rather than exact isomorphism checks.

#         :param G1: First graph (treated as source of components).
#         :type G1: nx.Graph
#         :param G2: Second graph (target for component mapping).
#         :type G2: nx.Graph
#         :param max_seeds: Maximum number of seeds per component pair.
#         :type max_seeds: int
#         :param max_steps: Maximum growth steps per seed.
#         :type max_steps: int
#         :returns: Combined mapping from nodes of ``G1`` to nodes of ``G2``.
#         :rtype: dict[int, int]
#         """
#         comps1 = sorted(nx.connected_components(G1), key=len, reverse=True)
#         comps2 = sorted(nx.connected_components(G2), key=len, reverse=True)

#         used2: Set[frozenset[int]] = set()
#         combined: MappingDict = {}

#         for comp1 in comps1:
#             size1 = len(comp1)
#             sub1 = G1.subgraph(comp1)

#             best_map: MappingDict = {}
#             best_size = 0
#             best_key2: Optional[frozenset[int]] = None

#             for comp2 in comps2:
#                 if len(comp2) != size1:
#                     continue
#                 key2 = frozenset(comp2)
#                 if key2 in used2:
#                     continue

#                 sub2 = G2.subgraph(comp2)
#                 pattern, host, pattern_is_G1 = self._prepare_orientation(
#                     sub1,
#                     sub2,
#                 )
#                 maps, local_best = self._approximate_mappings(
#                     pattern,
#                     host,
#                     max_seeds=max_seeds,
#                     max_steps=max_steps,
#                 )
#                 if not maps or local_best == 0:
#                     continue

#                 if local_best > best_size:
#                     best_size = local_best
#                     best = maps[0]
#                     if pattern_is_G1:
#                         best_map = best
#                     else:
#                         best_map = self._invert_mapping(best)
#                     best_key2 = key2

#             if best_map and best_key2 is not None:
#                 combined.update(best_map)
#                 used2.add(best_key2)

#         return combined

#     # ------------------------------------------------------------------
#     # Public approximate search – graph level
#     # ------------------------------------------------------------------
#     def find_common_subgraph_approx(
#         self,
#         G1: nx.Graph,
#         G2: nx.Graph,
#         *,
#         max_seeds: int = 16,
#         max_steps: int = 256,
#     ) -> "ApproxMCSMatcher":
#         """
#         Heuristically search for approximate common subgraphs.

#         This is a lightweight wrapper that ignores molecule-level
#         options and simply runs the greedy approximate search on the
#         whole (possibly wildcard-pruned) graphs.

#         :param G1: First input graph.
#         :type G1: nx.Graph
#         :param G2: Second input graph.
#         :type G2: nx.Graph
#         :param max_seeds: Maximum number of seed node pairs to explore.
#         :type max_seeds: int
#         :param max_steps: Maximum number of growth steps per seed.
#         :type max_steps: int
#         :returns: The matcher instance (with cache updated).
#         :rtype: ApproxMCSMatcher
#         """
#         return self.find_common_subgraph(
#             G1,
#             G2,
#             mcs=False,
#             mcs_mol=False,
#             max_seeds=max_seeds,
#             max_steps=max_steps,
#         )

#     def find_common_subgraph(
#         self,
#         G1: nx.Graph,
#         G2: nx.Graph,
#         *,
#         mcs: bool = False,
#         mcs_mol: bool = False,
#         max_seeds: int = 16,
#         max_steps: int = 256,
#     ) -> "ApproxMCSMatcher":
#         """
#         Approximate analogue of :py:meth:`MCSMatcher.find_common_subgraph`.

#         The signature mirrors the exact matcher, but the implementation
#         is greedy/heuristic:

#         1. Optionally prunes wildcard nodes from both graphs.
#         2. If :paramref:`mcs_mol` is ``True``, performs component-level
#            (molecule-level) approximate matching with
#            :py:meth:`_find_mcs_mol_approx`.
#         3. Otherwise, orients the pair so that the smaller graph is the
#            pattern and runs the heuristic search.

#         The :paramref:`mcs` flag is accepted for API compatibility but
#         has no distinct effect here; the heuristic always aims for large
#         mappings.

#         :param G1: First input graph.
#         :type G1: nx.Graph
#         :param G2: Second input graph.
#         :type G2: nx.Graph
#         :param mcs: Ignored (kept for API compatibility).
#         :type mcs: bool
#         :param mcs_mol: If ``True``, perform approximate
#             connected-component (molecule-level) matching.
#         :type mcs_mol: bool
#         :param max_seeds: Maximum number of seed pairs.
#         :type max_seeds: int
#         :param max_steps: Maximum growth steps per seed.
#         :type max_steps: int
#         :returns: The matcher instance (with internal cache updated).
#         :rtype: ApproxMCSMatcher
#         """
#         del mcs  # unused, kept for signature compatibility

#         self._mappings = []
#         self._last_size = 0
#         self._last_pattern_is_G1 = None

#         G1_use = self._prune_graph(G1)
#         G2_use = self._prune_graph(G2)

#         if mcs_mol:
#             combined = self._find_mcs_mol_approx(
#                 G1_use,
#                 G2_use,
#                 max_seeds=max_seeds,
#                 max_steps=max_steps,
#             )
#             self._mappings = [combined]  # G1 -> G2
#             self._last_size = len(combined)
#             self._last_pattern_is_G1 = True
#             return self

#         pattern, host, pattern_is_G1 = self._prepare_orientation(G1_use, G2_use)
#         self._last_pattern_is_G1 = pattern_is_G1

#         maps, best_size = self._approximate_mappings(
#             pattern,
#             host,
#             max_seeds=max_seeds,
#             max_steps=max_steps,
#         )
#         self._mappings = maps
#         self._last_size = best_size

#         return self

#     # ------------------------------------------------------------------
#     # Public approximate search – ITS / reaction-centre level
#     # ------------------------------------------------------------------
#     def find_rc_mapping(
#         self,
#         rc1: Any,
#         rc2: Any,
#         *,
#         side: str = "op",
#         mcs: bool = True,
#         mcs_mol: bool = False,
#         component: bool = True,
#         max_seeds: int = 16,
#         max_steps: int = 256,
#     ) -> "ApproxMCSMatcher":
#         """
#         Convenience wrapper for ITS reaction-centre or ITS-like graph
#         objects, analogous to :py:meth:`MCSMatcher.find_rc_mapping` but
#         using the heuristic search internally.

#         Depending on :paramref:`side`, this either uses
#         :func:`synkit.Graph.ITS.its_decompose` to obtain left/right
#         graphs or treats the inputs directly as graphs.

#         Side selection
#         --------------
#         * ``'r'``   → compare right sides:  ``r1`` vs ``r2``.
#         * ``'l'``   → compare left sides:   ``l1`` vs ``l2``.
#         * ``'op'``  → compare opposite:     ``r1`` vs ``l2``.
#         * ``'its'`` → treat ``rc1`` and ``rc2`` directly as graphs
#           (no decomposition), useful when the inputs are already ITS
#           (or ITS-like) :class:`networkx.Graph` objects.

#         Component-wise mode
#         -------------------
#         If :paramref:`component` is ``True``, the selected graphs are
#         decomposed into connected components, sorted by size
#         (descending), and matched pairwise using
#         :py:meth:`_componentwise_approx`. The resulting mappings are
#         combined into a single **G1 → G2** mapping in terms of the
#         original node ids. In this mode, :paramref:`mcs_mol` is ignored.

#         :param rc1: First reaction-centre or ITS-like graph object.
#         :type rc1: Any
#         :param rc2: Second reaction-centre or ITS-like graph object.
#         :type rc2: Any
#         :param side: Which ITS sides to compare (``'r'``, ``'l'``,
#             ``'op'``, or ``'its'``).
#         :type side: str
#         :param mcs: Ignored (kept for compatibility with
#             :class:`MCSMatcher`).
#         :type mcs: bool
#         :param mcs_mol: If ``True`` and :paramref:`component` is
#             ``False``, use approximate molecule-level matching via
#             :py:meth:`find_common_subgraph` with
#             :paramref:`mcs_mol=True`.
#         :type mcs_mol: bool
#         :param component: If ``True``, perform size-sorted,
#             component-wise approximate matching between the selected
#             sides and combine the per-component mappings into a single
#             mapping.
#         :type component: bool
#         :param max_seeds: Maximum number of seeds per call.
#         :type max_seeds: int
#         :param max_steps: Maximum growth steps per seed.
#         :type max_steps: int
#         :returns: The matcher instance (with internal cache updated).
#         :rtype: ApproxMCSMatcher
#         :raises ImportError: If :mod:`synkit` ITS utilities are not
#             available for ``side`` in ``{'r', 'l', 'op'}``.
#         :raises ValueError: If ``side`` is not one of
#             ``'r'``, ``'l'``, ``'op'``, ``'its'``.
#         """
#         del mcs  # unused, kept for signature compatibility

#         # reset cache
#         self._mappings = []
#         self._last_size = 0
#         self._last_pattern_is_G1 = None

#         side_norm = side.lower()

#         if side_norm == "its":
#             # Treat rc1 and rc2 directly as graphs
#             G1, G2 = rc1, rc2
#         else:
#             if its_decompose is None:
#                 raise ImportError(
#                     "synkit is not available; cannot decompose reaction centres "
#                     "for side values 'r', 'l' or 'op'."
#                 )

#             l1, r1 = its_decompose(rc1)
#             l2, r2 = its_decompose(rc2)

#             if side_norm == "r":
#                 G1, G2 = r1, r2
#             elif side_norm == "l":
#                 G1, G2 = l1, l2
#             elif side_norm == "op":
#                 G1, G2 = r1, l2
#             else:
#                 raise ValueError(
#                     "ApproxMCSMatcher.find_rc_mapping: side must be one of "
#                     "'r', 'l', 'op', 'its', got "
#                     f"{side!r}."
#                 )

#         G1_use = self._prune_graph(G1)
#         G2_use = self._prune_graph(G2)

#         if component:
#             combined = self._componentwise_approx(
#                 G1_use,
#                 G2_use,
#                 max_seeds=max_seeds,
#                 max_steps=max_steps,
#             )
#             self._mappings = [combined]  # G1 -> G2
#             self._last_size = len(combined)
#             self._last_pattern_is_G1 = True
#             return self

#         return self.find_common_subgraph(
#             G1_use,
#             G2_use,
#             mcs=False,
#             mcs_mol=mcs_mol,
#             max_seeds=max_seeds,
#             max_steps=max_steps,
#         )

#     # ------------------------------------------------------------------
#     # Accessors / properties
#     # ------------------------------------------------------------------
#     def get_mappings(self, direction: str = "pattern_to_host") -> List[MappingDict]:
#         """
#         Return a copy of the cached mapping list in the requested
#         orientation.

#         Internal orientation is **pattern → host**. This method can
#         convert to ``G1→G2`` or ``G2→G1`` based on the last call to
#         :py:meth:`find_common_subgraph` or
#         :py:meth:`find_common_subgraph_approx`.

#         :param direction: Orientation of the returned mappings. One of:
#             * ``"pattern_to_host"`` (default)
#             * ``"G1_to_G2"``
#             * ``"G2_to_G1"``
#         :type direction: str
#         :returns: List of mapping dictionaries.
#         :rtype: list[dict[int, int]]
#         :raises ValueError: If the direction is not supported.
#         """
#         if direction == "pattern_to_host" or self._last_pattern_is_G1 is None:
#             return [dict(m) for m in self._mappings]

#         if direction not in {"G1_to_G2", "G2_to_G1"}:
#             raise ValueError(
#                 "ApproxMCSMatcher.get_mappings: direction must be one of "
#                 "'pattern_to_host', 'G1_to_G2', 'G2_to_G1'."
#             )

#         pattern_is_G1 = self._last_pattern_is_G1
#         result: List[MappingDict] = []

#         for m in self._mappings:
#             if direction == "G1_to_G2":
#                 if pattern_is_G1:
#                     result.append(dict(m))
#                 else:
#                     result.append(self._invert_mapping(m))
#             else:
#                 if pattern_is_G1:
#                     result.append(self._invert_mapping(m))
#                 else:
#                     result.append(dict(m))

#         return result

#     @property
#     def mappings(self) -> List[MappingDict]:
#         """
#         Cached approximate mappings from the most recent search.

#         The orientation is pattern→host. For ``G1→G2`` or ``G2→G1``,
#         use :py:meth:`get_mappings`.

#         :returns: List of cached mapping dictionaries.
#         :rtype: list[dict[int, int]]
#         """
#         return self.get_mappings(direction="pattern_to_host")

#     @property
#     def last_size(self) -> int:
#         """
#         Size of the largest approximate mapping from the last search.

#         :returns: Size of the best mapping.
#         :rtype: int
#         """
#         return self._last_size

#     @property
#     def num_mappings(self) -> int:
#         """
#         Number of approximate mappings stored from the last search.

#         :returns: Count of mappings.
#         :rtype: int
#         """
#         return len(self._mappings)

#     @property
#     def mapping_direction(self) -> str:
#         """
#         Human-readable description of internal mapping orientation.

#         :returns: ``"G1_to_G2"``, ``"G2_to_G1"``, or ``"unknown"`` if
#             no search has been run.
#         :rtype: str
#         """
#         if self._last_pattern_is_G1 is None:
#             return "unknown"
#         return "G1_to_G2" if self._last_pattern_is_G1 else "G2_to_G1"

#     # ------------------------------------------------------------------
#     # Iteration & niceties
#     # ------------------------------------------------------------------
#     def __iter__(self) -> Iterable[MappingDict]:
#         """
#         Iterate over cached mappings in pattern→host orientation.

#         :returns: Iterator over mapping dictionaries.
#         :rtype: Iterable[dict[int, int]]
#         """
#         return iter(self._mappings)

#     def __repr__(self) -> str:
#         """
#         Short textual representation for debugging.

#         :returns: Summary string with key attributes.
#         :rtype: str
#         """
#         return (
#             f"<ApproxMCSMatcher mappings={self.num_mappings} "
#             f"last_size={self.last_size} "
#             f"prune_wc={self.prune_wc} "
#             f"prune_automorphisms={self.prune_automorphisms} "
#             f"direction={self.mapping_direction}>"
#         )

#     __str__ = __repr__

#     @property
#     def help(self) -> str:
#         """
#         Return the module-level documentation string.

#         :returns: The full module docstring, if available.
#         :rtype: str
#         """
#         return __doc__ or ""

from __future__ import annotations

"""
approx_mcs_matcher.py — Heuristic / Approximate MCS Matcher
===========================================================

A compact, OOP-style wrapper that mimics the public behaviour of
:class:`MCSMatcher`, but uses a **greedy, heuristic** search instead of
exhaustive subgraph enumeration.

Key ideas
---------
* Use :func:`generic_node_match` for flexible node-attribute comparison.
* Optionally augment structural similarity with a 1-WL-style
  color-refinement term (``use_wl=True``).
* Score node pairs based on attribute compatibility, local degrees,
  neighbour-degree overlap and (optionally) WL colors.
* Select a small set of high-scoring **seed pairs**.
* For each seed, **greedily grow** a mapping along neighbouring nodes,
  enforcing local edge compatibility along the way.
* Cache the resulting (approximate) mappings in the same orientation
  scheme as :class:`MCSMatcher`:

  - Internal cache: **pattern → host** (pattern = smaller graph after
    optional wildcard pruning).
  - :py:meth:`ApproxMCSMatcher.get_mappings` converts to ``G1→G2`` or
    ``G2→G1``.

Public API
~~~~~~~~~~
.. code-block:: python

    from synkit.Graph.Matcher.approx_mcs import ApproxMCSMatcher
    import networkx as nx

    # build two toy molecular graphs
    g1 = nx.Graph()
    g1.add_node(0, element="C")
    g1.add_node(1, element="O")
    g1.add_edge(0, 1, order=1)

    g2 = nx.Graph()
    g2.add_node(10, element="C")
    g2.add_node(11, element="O")
    g2.add_edge(10, 11, order=1)

    matcher = ApproxMCSMatcher(use_wl=True).find_common_subgraph(g1, g2)
    g1_to_g2 = matcher.get_mappings(direction="G1_to_G2")[0]
    print(g1_to_g2)  # e.g. {0: 10, 1: 11}

Classes
~~~~~~~
.. autosummary::
   :toctree: generated/

   ApproxMCSMatcher
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import generic_node_match

try:  # pragma: no cover - optional dependency
    from synkit.Graph.ITS import its_decompose
except ImportError:  # pragma: no cover
    its_decompose = None  # type: ignore[assignment]

__all__ = ["ApproxMCSMatcher"]

MappingDict = Dict[int, int]


class ApproxMCSMatcher:
    """
    Heuristic / approximate common-subgraph matcher.

    This class provides a **fast, approximate** alternative to
    :class:`MCSMatcher`. It does *not* enumerate all subgraph
    isomorphisms. Instead, it:

    1. Picks a set of high-scoring node pairs as **seeds** based on
       attribute and local-structure similarity.
    2. For each seed, **greedily grows** a subgraph isomorphism by
       extending along neighbouring nodes while maintaining local
       adjacency consistency.
    3. Collects the resulting partial mappings and reports them in the
       same orientation style as :class:`MCSMatcher`.

    Optionally, a 1-WL-style color refinement (``use_wl=True``) is used
    to produce coarse structural colors; matching nodes with identical
    WL colors receive a small similarity bonus.

    The result is an approximate maximum-common-subgraph (MCS) mapping
    that is usually close to optimal for molecular graphs, but can be
    computed much faster and can be bounded by simple iteration
    parameters.

    Orientation and storage
    -----------------------
    Internally, mappings are stored in **pattern → host** orientation,
    where the pattern is the smaller of the two graphs after optional
    wildcard pruning. The :py:meth:`get_mappings` helper converts these
    into ``G1→G2`` or ``G2→G1`` orientation based on which original
    graph served as the pattern.

    API compatibility
    -----------------
    The constructor and public methods mirror :class:`MCSMatcher` as
    closely as possible so that :class:`ApproxMCSMatcher` can drop in as
    a faster, heuristic substitute in many workflows.

    Parameters
    ----------
    node_attrs : list[str] or None, optional
        Node attribute keys to compare. If ``None``, defaults to
        ``["element"]``.
    node_defaults : list[Any] or None, optional
        Fallback values for each node attribute when missing. If
        ``None``, defaults to a list of ``"*"``
        with the same length as :paramref:`node_attrs`.
    allow_shift : bool, optional
        Placeholder for future asymmetric rules. Kept for API
        compatibility with :class:`MCSMatcher`.
    edge_attrs : list[str] or None, optional
        Edge attribute keys to use for scalar comparison
        (e.g. ``["order"]``). If ``None``, defaults to ``["order"]``.
    prune_wc : bool, optional
        If ``True``, strip wildcard nodes (see
        :paramref:`wildcard_element`, :paramref:`element_key`) from both
        graphs before searching.
    prune_automorphisms : bool, optional
        If ``True``, collapse mappings that have the same host node set
        (automorphism pruning).
    wildcard_element : Any, optional
        Attribute value denoting wildcard nodes (typically ``"*"``,
        used together with :paramref:`element_key`).
    element_key : str, optional
        Node attribute key used to detect wildcard nodes when
        :paramref:`prune_wc` is ``True``.
    use_wl : bool, optional
        If ``True``, run a simple 1-WL-style color refinement on both
        graphs and include the resulting colors in the node similarity
        score.
    wl_max_iter : int, optional
        Maximum number of WL refinement iterations.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        node_attrs: Optional[List[str]] = None,
        node_defaults: Optional[List[Any]] = None,
        allow_shift: bool = True,
        *,
        edge_attrs: Optional[List[str]] = None,
        prune_wc: bool = False,
        prune_automorphisms: bool = False,
        wildcard_element: Any = "*",
        element_key: str = "element",
        use_wl: bool = False,
        wl_max_iter: int = 3,
    ) -> None:
        if node_attrs is None:
            node_attrs = ["element"]
        if node_defaults is None:
            node_defaults = ["*"] * len(node_attrs)
        if len(node_defaults) != len(node_attrs):
            raise ValueError(
                "ApproxMCSMatcher: node_defaults must have the same length "
                "as node_attrs."
            )

        self._node_attrs: List[str] = node_attrs
        self._node_defaults: List[Any] = node_defaults
        self._edge_attrs: List[str] = edge_attrs or ["order"]
        self.allow_shift: bool = allow_shift

        self.prune_wc: bool = prune_wc
        self.prune_automorphisms: bool = prune_automorphisms
        self.wildcard_element: Any = wildcard_element
        self.element_key: str = element_key

        self.use_wl: bool = use_wl
        self.wl_max_iter: int = wl_max_iter

        comparators: List[Callable[[Any, Any], bool]] = [
            (lambda x, y: x == y) for _ in node_attrs
        ]
        self.node_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] = (
            generic_node_match(
                node_attrs,
                node_defaults,
                comparators,
            )
        )

        # Internal cache (pattern → host)
        self._mappings: List[MappingDict] = []
        self._last_size: int = 0
        self._last_pattern_is_G1: Optional[bool] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prune_graph(self, G: nx.Graph) -> nx.Graph:
        """
        Remove wildcard nodes from ``G`` (non-inplace) if applicable.

        When :pyattr:`prune_wc` is ``False``, the input graph is returned
        as-is.

        :param G: Input graph.
        :type G: nx.Graph
        :returns: Possibly pruned graph.
        :rtype: nx.Graph
        """
        if not self.prune_wc:
            return G

        key = self.element_key
        wc = self.wildcard_element
        keep_nodes = [n for n, d in G.nodes(data=True) if d.get(key) != wc]
        return G.subgraph(keep_nodes).copy()

    @staticmethod
    def _prepare_orientation(
        G1: nx.Graph,
        G2: nx.Graph,
    ) -> Tuple[nx.Graph, nx.Graph, bool]:
        """
        Ensure the smaller graph is used as pattern.

        :param G1: First input graph.
        :type G1: nx.Graph
        :param G2: Second input graph.
        :type G2: nx.Graph
        :returns: Tuple ``(pattern, host, pattern_is_G1)``.
        :rtype: tuple[nx.Graph, nx.Graph, bool]
        """
        if G1.number_of_nodes() <= G2.number_of_nodes():
            return G1, G2, True
        return G2, G1, False

    @staticmethod
    def _invert_mapping(mapping: MappingDict) -> MappingDict:
        """
        Invert a mapping from host→pattern to pattern→host or vice versa.

        :param mapping: Mapping to invert.
        :type mapping: dict[int, int]
        :returns: Inverted mapping.
        :rtype: dict[int, int]
        """
        return {v: k for k, v in mapping.items()}

    def _edge_match(
        self,
        attrs1: Dict[str, Any],
        attrs2: Dict[str, Any],
    ) -> bool:
        """
        Compare edge attributes listed in :pyattr:`_edge_attrs`.

        For each attribute name:

        * If both values can be cast to ``float``, use numeric equality.
        * Otherwise, fall back to direct ``==`` comparison.
        * Missing values on both sides are ignored.

        :param attrs1: Edge attributes of the first edge.
        :type attrs1: dict[str, Any]
        :param attrs2: Edge attributes of the second edge.
        :type attrs2: dict[str, Any]
        :returns: ``True`` if the attributes are compatible.
        :rtype: bool
        """
        for name in self._edge_attrs:
            v1 = attrs1.get(name)
            v2 = attrs2.get(name)
            if v1 is None and v2 is None:
                continue
            try:
                if float(v1) != float(v2):
                    return False
            except (TypeError, ValueError):
                if v1 != v2:
                    return False
        return True

    def _compute_wl_colors(self, G: nx.Graph) -> Dict[int, int]:
        """
        Run a simple 1-WL color refinement on ``G``.

        Initial colors are based on the configured node attributes and
        the node degree; refinement iteratively refines colors using the
        multiset of neighbour colors.

        :param G: Input graph.
        :type G: nx.Graph
        :returns: Mapping from node id to WL color id.
        :rtype: dict[int, int]
        """
        colors: Dict[int, int] = {}
        for n, data in G.nodes(data=True):
            attr_vals: List[Any] = []
            for key, default in zip(self._node_attrs, self._node_defaults):
                attr_vals.append(data.get(key, default))
            colors[n] = hash((tuple(attr_vals), G.degree[n]))

        for _ in range(self.wl_max_iter):
            signatures: Dict[Tuple[int, Tuple[int, ...]], int] = {}
            new_colors: Dict[int, int] = {}
            for n in G.nodes():
                neigh = [colors[nb] for nb in G.neighbors(n)]
                neigh.sort()
                sig = (colors[n], tuple(neigh))
                if sig not in signatures:
                    signatures[sig] = len(signatures)
                new_colors[n] = signatures[sig]
            if new_colors == colors:
                break
            colors = new_colors

        return colors

    def _node_similarity(
        self,
        p: int,
        h: int,
        pattern: nx.Graph,
        host: nx.Graph,
        wl_pattern: Dict[int, int],
        wl_host: Dict[int, int],
    ) -> float:
        """
        Compute a simple structural similarity score for a node pair.

        The score is based on:

        * Node attribute compatibility via :pyattr:`node_match`.
        * Degree difference between ``p`` and ``h``.
        * Overlap in neighbour degrees.
        * Optional 1-WL color agreement (if :pyattr:`use_wl` is ``True``).

        :param p: Node id in the pattern graph.
        :type p: int
        :param h: Node id in the host graph.
        :type h: int
        :param pattern: Pattern graph.
        :type pattern: nx.Graph
        :param host: Host graph.
        :type host: nx.Graph
        :param wl_pattern: WL colors for the pattern graph.
        :type wl_pattern: dict[int, int]
        :param wl_host: WL colors for the host graph.
        :type wl_host: dict[int, int]
        :returns: Similarity score (larger is better; negative for
            incompatible pairs).
        :rtype: float
        """
        pdata = pattern.nodes[p]
        hdata = host.nodes[h]
        if not self.node_match(pdata, hdata):
            return -1.0

        deg_p = pattern.degree[p]
        deg_h = host.degree[h]
        base = 2.0
        penalty = float(abs(deg_p - deg_h))

        neigh_deg_p = {pattern.degree[n] for n in pattern.neighbors(p)}
        neigh_deg_h = {host.degree[n] for n in host.neighbors(h)}
        overlap = float(len(neigh_deg_p & neigh_deg_h))

        color_term = 0.0
        if self.use_wl and wl_pattern and wl_host:
            if wl_pattern.get(p) == wl_host.get(h):
                color_term = 1.0
            else:
                color_term = -1.0

        return base - penalty + 0.5 * overlap + color_term

    def _generate_seeds(
        self,
        pattern: nx.Graph,
        host: nx.Graph,
        max_seeds: int,
        wl_pattern: Dict[int, int],
        wl_host: Dict[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Generate a list of high-scoring seed node pairs.

        :param pattern: Pattern graph.
        :type pattern: nx.Graph
        :param host: Host graph.
        :type host: nx.Graph
        :param max_seeds: Maximum number of seed pairs to keep.
        :type max_seeds: int
        :param wl_pattern: WL colors for ``pattern`` (may be empty).
        :type wl_pattern: dict[int, int]
        :param wl_host: WL colors for ``host`` (may be empty).
        :type wl_host: dict[int, int]
        :returns: List of ``(pattern_node, host_node)`` pairs.
        :rtype: list[tuple[int, int]]
        """
        candidates: List[Tuple[float, int, int]] = []
        for p in pattern.nodes():
            for h in host.nodes():
                score = self._node_similarity(
                    p,
                    h,
                    pattern,
                    host,
                    wl_pattern,
                    wl_host,
                )
                if score <= 0.0:
                    continue
                candidates.append((score, p, h))

        candidates.sort(reverse=True, key=lambda t: t[0])
        top = candidates[:max_seeds]
        return [(p, h) for _, p, h in top]

    def _respects_edges(
        self,
        p: int,
        h: int,
        pattern: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
    ) -> bool:
        """
        Check whether extending mapping with ``p → h`` is locally valid.

        This ensures that for any already-mapped neighbour ``p_n`` of
        ``p``, the candidate host node ``h`` is adjacent to the mapped
        host node and that edge attributes are compatible.

        :param p: Pattern node to extend with.
        :type p: int
        :param h: Candidate host node.
        :type h: int
        :param pattern: Pattern graph.
        :type pattern: nx.Graph
        :param host: Host graph.
        :type host: nx.Graph
        :param mapping: Current partial mapping (pattern→host).
        :type mapping: dict[int, int]
        :returns: ``True`` if the extension is feasible.
        :rtype: bool
        """
        for p_n, h_n in mapping.items():
            if not pattern.has_edge(p, p_n):
                continue
            if not host.has_edge(h, h_n):
                return False
            attrs_p = pattern[p][p_n]
            attrs_h = host[h][h_n]
            if not self._edge_match(attrs_p, attrs_h):
                return False
        return True

    def _candidate_hosts(
        self,
        p: int,
        pattern: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
        wl_pattern: Dict[int, int],
        wl_host: Dict[int, int],
    ) -> List[int]:
        """
        Enumerate host nodes that can be matched to pattern node ``p``.

        :param p: Pattern node.
        :type p: int
        :param pattern: Pattern graph.
        :type pattern: nx.Graph
        :param host: Host graph.
        :type host: nx.Graph
        :param mapping: Current partial mapping (pattern→host).
        :type mapping: dict[int, int]
        :param wl_pattern: WL colors for ``pattern`` (may be empty).
        :type wl_pattern: dict[int, int]
        :param wl_host: WL colors for ``host`` (may be empty).
        :type wl_host: dict[int, int]
        :returns: List of feasible host node ids.
        :rtype: list[int]
        """
        mapped_hosts = set(mapping.values())
        pdata = pattern.nodes[p]
        candidates: List[int] = []

        for h in host.nodes():
            if h in mapped_hosts:
                continue
            hdata = host.nodes[h]
            if not self.node_match(pdata, hdata):
                continue
            if not self._respects_edges(p, h, pattern, host, mapping):
                continue
            if self.use_wl and wl_pattern and wl_host:
                score = self._node_similarity(
                    p,
                    h,
                    pattern,
                    host,
                    wl_pattern,
                    wl_host,
                )
                if score <= 0.0:
                    continue
            candidates.append(h)

        return candidates

    def _grow_from_seed(
        self,
        pattern: nx.Graph,
        host: nx.Graph,
        seed_p: int,
        seed_h: int,
        max_steps: int,
        wl_pattern: Dict[int, int],
        wl_host: Dict[int, int],
    ) -> MappingDict:
        """
        Greedily grow a subgraph mapping starting from a single seed.

        :param pattern: Pattern graph.
        :type pattern: nx.Graph
        :param host: Host graph.
        :type host: nx.Graph
        :param seed_p: Seed node in pattern graph.
        :type seed_p: int
        :param seed_h: Seed node in host graph.
        :type seed_h: int
        :param max_steps: Maximum number of growth steps.
        :type max_steps: int
        :param wl_pattern: WL colors for ``pattern`` (may be empty).
        :type wl_pattern: dict[int, int]
        :param wl_host: WL colors for ``host`` (may be empty).
        :type wl_host: dict[int, int]
        :returns: Completed partial mapping (pattern→host).
        :rtype: dict[int, int]
        """
        mapping: MappingDict = {seed_p: seed_h}
        frontier: Set[int] = {n for n in pattern.neighbors(seed_p)}
        steps = 0

        while frontier and steps < max_steps:
            steps += 1
            p = max(frontier, key=pattern.degree.__getitem__)
            frontier.remove(p)

            candidates = self._candidate_hosts(
                p,
                pattern,
                host,
                mapping,
                wl_pattern,
                wl_host,
            )
            if not candidates:
                continue

            best_h = max(
                candidates,
                key=lambda h: self._node_similarity(
                    p,
                    h,
                    pattern,
                    host,
                    wl_pattern,
                    wl_host,
                ),
            )
            mapping[p] = best_h

            for q in pattern.neighbors(p):
                if q in mapping or q in frontier:
                    continue
                frontier.add(q)

        return mapping

    def _approximate_mappings(
        self,
        pattern: nx.Graph,
        host: nx.Graph,
        max_seeds: int,
        max_steps: int,
    ) -> Tuple[List[MappingDict], int]:
        """
        Core heuristic search between ``pattern`` and ``host``.

        This is factored out so that it can be reused for whole-graph,
        component-wise, and reaction-centre searches.

        :param pattern: Pattern graph (smaller or equal).
        :type pattern: nx.Graph
        :param host: Host graph (larger or equal).
        :type host: nx.Graph
        :param max_seeds: Maximum number of seed node pairs to explore.
        :type max_seeds: int
        :param max_steps: Maximum number of growth steps per seed.
        :type max_steps: int
        :returns: Tuple ``(mappings, best_size)`` where ``mappings`` are
            pattern→host and ``best_size`` is the size of the largest
            mapping found.
        :rtype: tuple[list[dict[int, int]], int]
        """
        wl_pattern: Dict[int, int] = {}
        wl_host: Dict[int, int] = {}
        if self.use_wl:
            wl_pattern = self._compute_wl_colors(pattern)
            wl_host = self._compute_wl_colors(host)

        seeds = self._generate_seeds(
            pattern,
            host,
            max_seeds=max_seeds,
            wl_pattern=wl_pattern,
            wl_host=wl_host,
        )
        if not seeds:
            return [], 0

        best_size = 0
        all_maps: List[MappingDict] = []

        for seed_p, seed_h in seeds:
            mapping = self._grow_from_seed(
                pattern,
                host,
                seed_p,
                seed_h,
                max_steps=max_steps,
                wl_pattern=wl_pattern,
                wl_host=wl_host,
            )
            all_maps.append(mapping)
            size = len(mapping)
            if size > best_size:
                best_size = size

        if self.prune_automorphisms:
            filtered: List[MappingDict] = []
            seen_host_sets: Set[frozenset[int]] = set()
            all_maps.sort(key=lambda d: (-len(d), tuple(sorted(d.items()))))
            for mp in all_maps:
                hset = frozenset(mp.values())
                if hset in seen_host_sets:
                    continue
                seen_host_sets.add(hset)
                filtered.append(mp)
            all_maps = filtered

        return all_maps, best_size

    def _componentwise_approx(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        *,
        max_seeds: int,
        max_steps: int,
    ) -> MappingDict:
        """
        Component-wise approximate matching between ``G1`` and ``G2``.

        Connected components of each graph are sorted by size
        (descending) and matched pairwise (largest with largest, etc.).
        For each pair, an approximate mapping is computed and the best
        mapping is merged into a combined **G1 → G2** mapping.

        :param G1: First input graph.
        :type G1: nx.Graph
        :param G2: Second input graph.
        :type G2: nx.Graph
        :param max_seeds: Maximum number of seed pairs per component pair.
        :type max_seeds: int
        :param max_steps: Maximum growth steps per seed.
        :type max_steps: int
        :returns: Combined mapping from nodes of ``G1`` to nodes of ``G2``.
        :rtype: dict[int, int]
        """
        comps1 = [G1.subgraph(c).copy() for c in nx.connected_components(G1)]
        comps2 = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]

        comps1.sort(key=lambda g: g.number_of_nodes(), reverse=True)
        comps2.sort(key=lambda g: g.number_of_nodes(), reverse=True)

        limit = min(len(comps1), len(comps2))
        combined: MappingDict = {}

        for idx in range(limit):
            sub1 = comps1[idx]
            sub2 = comps2[idx]

            pattern, host, pattern_is_G1 = self._prepare_orientation(sub1, sub2)
            maps, _ = self._approximate_mappings(
                pattern,
                host,
                max_seeds=max_seeds,
                max_steps=max_steps,
            )
            if not maps:
                continue

            best = maps[0]
            if pattern_is_G1:
                combined.update(best)
            else:
                combined.update(self._invert_mapping(best))

        return combined

    def _find_mcs_mol_approx(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        *,
        max_seeds: int,
        max_steps: int,
    ) -> MappingDict:
        """
        Approximate molecule-level (component) matching in G1→G2.

        This mirrors :py:meth:`MCSMatcher._find_mcs_mol` but uses the
        heuristic search rather than exact isomorphism checks.

        :param G1: First graph (treated as source of components).
        :type G1: nx.Graph
        :param G2: Second graph (target for component mapping).
        :type G2: nx.Graph
        :param max_seeds: Maximum number of seeds per component pair.
        :type max_seeds: int
        :param max_steps: Maximum growth steps per seed.
        :type max_steps: int
        :returns: Combined mapping from nodes of ``G1`` to nodes of ``G2``.
        :rtype: dict[int, int]
        """
        comps1 = sorted(nx.connected_components(G1), key=len, reverse=True)
        comps2 = sorted(nx.connected_components(G2), key=len, reverse=True)

        used2: Set[frozenset[int]] = set()
        combined: MappingDict = {}

        for comp1 in comps1:
            size1 = len(comp1)
            sub1 = G1.subgraph(comp1)

            best_map: MappingDict = {}
            best_size = 0
            best_key2: Optional[frozenset[int]] = None

            for comp2 in comps2:
                if len(comp2) != size1:
                    continue
                key2 = frozenset(comp2)
                if key2 in used2:
                    continue

                sub2 = G2.subgraph(comp2)
                pattern, host, pattern_is_G1 = self._prepare_orientation(
                    sub1,
                    sub2,
                )
                maps, local_best = self._approximate_mappings(
                    pattern,
                    host,
                    max_seeds=max_seeds,
                    max_steps=max_steps,
                )
                if not maps or local_best == 0:
                    continue

                if local_best > best_size:
                    best_size = local_best
                    best = maps[0]
                    if pattern_is_G1:
                        best_map = best
                    else:
                        best_map = self._invert_mapping(best)
                    best_key2 = key2

            if best_map and best_key2 is not None:
                combined.update(best_map)
                used2.add(best_key2)

        return combined

    def _select_graphs_from_rc(
        self,
        rc1: Any,
        rc2: Any,
        side: str,
    ) -> Tuple[Any, Any]:
        """
        Select ITS sides or treat inputs as graphs.

        :param rc1: First reaction-centre or ITS-like graph object.
        :type rc1: Any
        :param rc2: Second reaction-centre or ITS-like graph object.
        :type rc2: Any
        :param side: Which ITS sides to compare.
        :type side: str
        :returns: Tuple ``(G1, G2)`` as graphs.
        :rtype: tuple[Any, Any]
        :raises ImportError: If ITS utilities are required but missing.
        :raises ValueError: If ``side`` is invalid.
        """
        side_norm = side.lower()

        if side_norm == "its":
            return rc1, rc2

        if its_decompose is None:
            raise ImportError(
                "synkit is not available; cannot decompose reaction centres "
                "for side values 'r', 'l' or 'op'."
            )

        l1, r1 = its_decompose(rc1)
        l2, r2 = its_decompose(rc2)

        if side_norm == "r":
            return r1, r2
        if side_norm == "l":
            return l1, l2
        if side_norm == "op":
            return r1, l2

        raise ValueError(
            "ApproxMCSMatcher.find_rc_mapping: side must be one of "
            "'r', 'l', 'op', 'its', got "
            f"{side!r}."
        )

    # ------------------------------------------------------------------
    # Public approximate search – graph level
    # ------------------------------------------------------------------
    def find_common_subgraph_approx(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        *,
        max_seeds: int = 16,
        max_steps: int = 256,
    ) -> "ApproxMCSMatcher":
        """
        Heuristically search for approximate common subgraphs.

        This is a lightweight wrapper that ignores molecule-level
        options and simply runs the greedy approximate search on the
        whole (possibly wildcard-pruned) graphs.

        :param G1: First input graph.
        :type G1: nx.Graph
        :param G2: Second input graph.
        :type G2: nx.Graph
        :param max_seeds: Maximum number of seed node pairs to explore.
        :type max_seeds: int
        :param max_steps: Maximum number of growth steps per seed.
        :type max_steps: int
        :returns: The matcher instance (with cache updated).
        :rtype: ApproxMCSMatcher
        """
        return self.find_common_subgraph(
            G1,
            G2,
            mcs=False,
            mcs_mol=False,
            max_seeds=max_seeds,
            max_steps=max_steps,
        )

    def find_common_subgraph(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        *,
        mcs: bool = False,
        mcs_mol: bool = False,
        max_seeds: int = 16,
        max_steps: int = 256,
    ) -> "ApproxMCSMatcher":
        """
        Approximate analogue of :py:meth:`MCSMatcher.find_common_subgraph`.

        The signature mirrors the exact matcher, but the implementation
        is greedy/heuristic:

        1. Optionally prunes wildcard nodes from both graphs.
        2. If :paramref:`mcs_mol` is ``True``, performs component-level
           (molecule-level) approximate matching with
           :py:meth:`_find_mcs_mol_approx`.
        3. Otherwise, orients the pair so that the smaller graph is the
           pattern and runs the heuristic search.

        The :paramref:`mcs` flag is accepted for API compatibility but
        has no distinct effect here; the heuristic always aims for large
        mappings.

        :param G1: First input graph.
        :type G1: nx.Graph
        :param G2: Second input graph.
        :type G2: nx.Graph
        :param mcs: Ignored (kept for API compatibility).
        :type mcs: bool
        :param mcs_mol: If ``True``, perform approximate
            connected-component (molecule-level) matching.
        :type mcs_mol: bool
        :param max_seeds: Maximum number of seed pairs.
        :type max_seeds: int
        :param max_steps: Maximum growth steps per seed.
        :type max_steps: int
        :returns: The matcher instance (with internal cache updated).
        :rtype: ApproxMCSMatcher
        """
        del mcs  # unused, kept for signature compatibility

        self._mappings = []
        self._last_size = 0
        self._last_pattern_is_G1 = None

        G1_use = self._prune_graph(G1)
        G2_use = self._prune_graph(G2)

        if mcs_mol:
            combined = self._find_mcs_mol_approx(
                G1_use,
                G2_use,
                max_seeds=max_seeds,
                max_steps=max_steps,
            )
            self._mappings = [combined]  # G1 -> G2
            self._last_size = len(combined)
            self._last_pattern_is_G1 = True
            return self

        pattern, host, pattern_is_G1 = self._prepare_orientation(G1_use, G2_use)
        self._last_pattern_is_G1 = pattern_is_G1

        maps, best_size = self._approximate_mappings(
            pattern,
            host,
            max_seeds=max_seeds,
            max_steps=max_steps,
        )
        self._mappings = maps
        self._last_size = best_size

        return self

    # ------------------------------------------------------------------
    # Public approximate search – ITS / reaction-centre level
    # ------------------------------------------------------------------
    def find_rc_mapping(
        self,
        rc1: Any,
        rc2: Any,
        *,
        side: str = "op",
        mcs: bool = True,
        mcs_mol: bool = False,
        component: bool = True,
        max_seeds: int = 16,
        max_steps: int = 256,
    ) -> "ApproxMCSMatcher":
        """
        Convenience wrapper for ITS reaction-centre or ITS-like graph
        objects, analogous to :py:meth:`MCSMatcher.find_rc_mapping` but
        using the heuristic search internally.

        Depending on :paramref:`side`, this either uses
        :func:`synkit.Graph.ITS.its_decompose` to obtain left/right
        graphs or treats the inputs directly as graphs.

        Side selection
        --------------
        * ``'r'``   → compare right sides:  ``r1`` vs ``r2``.
        * ``'l'``   → compare left sides:   ``l1`` vs ``l2``.
        * ``'op'``  → compare opposite:     ``r1`` vs ``l2``.
        * ``'its'`` → treat ``rc1`` and ``rc2`` directly as graphs
          (no decomposition), useful when the inputs are already ITS
          (or ITS-like) :class:`networkx.Graph` objects.

        Component-wise mode
        -------------------
        If :paramref:`component` is ``True``, the selected graphs are
        decomposed into connected components, sorted by size
        (descending), and matched pairwise using
        :py:meth:`_componentwise_approx`. The resulting mappings are
        combined into a single **G1 → G2** mapping in terms of the
        original node ids. In this mode, :paramref:`mcs_mol` is ignored.

        :param rc1: First reaction-centre or ITS-like graph object.
        :type rc1: Any
        :param rc2: Second reaction-centre or ITS-like graph object.
        :type rc2: Any
        :param side: Which ITS sides to compare (``'r'``, ``'l'``,
            ``'op'``, or ``'its'``).
        :type side: str
        :param mcs: Ignored (kept for compatibility with
            :class:`MCSMatcher`).
        :type mcs: bool
        :param mcs_mol: If ``True`` and :paramref:`component` is
            ``False``, use approximate molecule-level matching via
            :py:meth:`find_common_subgraph` with
            :paramref:`mcs_mol=True`.
        :type mcs_mol: bool
        :param component: If ``True``, perform size-sorted,
            component-wise approximate matching between the selected
            sides and combine the per-component mappings into a single
            mapping.
        :type component: bool
        :param max_seeds: Maximum number of seeds per call.
        :type max_seeds: int
        :param max_steps: Maximum growth steps per seed.
        :type max_steps: int
        :returns: The matcher instance (with internal cache updated).
        :rtype: ApproxMCSMatcher
        :raises ImportError: If :mod:`synkit` ITS utilities are not
            available for ``side`` in ``{'r', 'l', 'op'}``.
        :raises ValueError: If ``side`` is not one of
            ``'r'``, ``'l'``, ``'op'``, ``'its'``.
        """
        del mcs  # unused, kept for signature compatibility

        self._mappings = []
        self._last_size = 0
        self._last_pattern_is_G1 = None

        G1, G2 = self._select_graphs_from_rc(rc1, rc2, side)

        G1_use = self._prune_graph(G1)
        G2_use = self._prune_graph(G2)

        if component:
            combined = self._componentwise_approx(
                G1_use,
                G2_use,
                max_seeds=max_seeds,
                max_steps=max_steps,
            )
            self._mappings = [combined]  # G1 -> G2
            self._last_size = len(combined)
            self._last_pattern_is_G1 = True
            return self

        return self.find_common_subgraph(
            G1_use,
            G2_use,
            mcs=False,
            mcs_mol=mcs_mol,
            max_seeds=max_seeds,
            max_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Accessors / properties
    # ------------------------------------------------------------------
    def get_mappings(self, direction: str = "pattern_to_host") -> List[MappingDict]:
        """
        Return a copy of the cached mapping list in the requested
        orientation.

        Internal orientation is **pattern → host**. This method can
        convert to ``G1→G2`` or ``G2→G1`` based on the last call to
        :py:meth:`find_common_subgraph`,
        :py:meth:`find_common_subgraph_approx` or
        :py:meth:`find_rc_mapping`.

        :param direction: Orientation of the returned mappings. One of:
            * ``"pattern_to_host"`` (default)
            * ``"G1_to_G2"``
            * ``"G2_to_G1"``
        :type direction: str
        :returns: List of mapping dictionaries.
        :rtype: list[dict[int, int]]
        :raises ValueError: If the direction is not supported.
        """
        if direction == "pattern_to_host" or self._last_pattern_is_G1 is None:
            return [dict(m) for m in self._mappings]

        if direction not in {"G1_to_G2", "G2_to_G1"}:
            raise ValueError(
                "ApproxMCSMatcher.get_mappings: direction must be one of "
                "'pattern_to_host', 'G1_to_G2', 'G2_to_G1'."
            )

        pattern_is_G1 = self._last_pattern_is_G1
        result: List[MappingDict] = []

        for m in self._mappings:
            if direction == "G1_to_G2":
                if pattern_is_G1:
                    result.append(dict(m))
                else:
                    result.append(self._invert_mapping(m))
            else:
                if pattern_is_G1:
                    result.append(self._invert_mapping(m))
                else:
                    result.append(dict(m))

        return result

    @property
    def mappings(self) -> List[MappingDict]:
        """
        Cached approximate mappings from the most recent search.

        The orientation is pattern→host. For ``G1→G2`` or ``G2→G1``,
        use :py:meth:`get_mappings`.

        :returns: List of cached mapping dictionaries.
        :rtype: list[dict[int, int]]
        """
        return self.get_mappings(direction="pattern_to_host")

    @property
    def last_size(self) -> int:
        """
        Size of the largest approximate mapping from the last search.

        :returns: Size of the best mapping.
        :rtype: int
        """
        return self._last_size

    @property
    def num_mappings(self) -> int:
        """
        Number of approximate mappings stored from the last search.

        :returns: Count of mappings.
        :rtype: int
        """
        return len(self._mappings)

    @property
    def mapping_direction(self) -> str:
        """
        Human-readable description of internal mapping orientation.

        :returns: ``"G1_to_G2"``, ``"G2_to_G1"``, or ``"unknown"`` if
            no search has been run.
        :rtype: str
        """
        if self._last_pattern_is_G1 is None:
            return "unknown"
        return "G1_to_G2" if self._last_pattern_is_G1 else "G2_to_G1"

    # ------------------------------------------------------------------
    # Iteration & niceties
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterable[MappingDict]:
        """
        Iterate over cached mappings in pattern→host orientation.

        :returns: Iterator over mapping dictionaries.
        :rtype: Iterable[dict[int, int]]
        """
        return iter(self._mappings)

    def __repr__(self) -> str:
        """
        Short textual representation for debugging.

        :returns: Summary string with key attributes.
        :rtype: str
        """
        return (
            f"<ApproxMCSMatcher mappings={self.num_mappings} "
            f"last_size={self.last_size} "
            f"prune_wc={self.prune_wc} "
            f"prune_automorphisms={self.prune_automorphisms} "
            f"use_wl={self.use_wl} "
            f"direction={self.mapping_direction}>"
        )

    __str__ = __repr__

    @property
    def help(self) -> str:
        """
        Return the module-level documentation string.

        :returns: The full module docstring, if available.
        :rtype: str
        """
        return __doc__ or ""
