from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import networkx as nx

logger = logging.getLogger(__name__)

Node = Any
NodeData = Dict[str, Any]


class GraphAnnotator:
    """
    Compute optional topology annotations for a NetworkX molecular graph.

    The annotator mutates a graph (in-place by default) or a shallow copy if
    `in_place=False`. Methods are chainable and return ``self``; use the
    ``.graph`` property to retrieve the annotated graph.

    Supported annotations:
      - node degree -> ``atom_degree``
      - neighborhood element counts -> ``nbr_elements_counts_r1``
      - shortest distances to motif sets (halogen/hetero/aromatic/carbonyl) ->
        ``dist_to_<motif>``
      - conjugated/pi connected component size -> ``conj_component_size``
      - ring sizes via cycle basis -> ``ring_sizes`` and updated ``in_ring``

    Notes
    -----
    * Graph nodes are expected to carry an ``'element'`` key (string) and may
      optionally have ``'aromatic'`` or ``'is_halogen'`` boolean flags.
    * Edge attributes used: ``'order'`` (numeric-like) and ``'conjugated'``.
    """

    DEFAULT_MAX_DISTANCE = 99

    def __init__(
        self,
        G: nx.Graph,
        in_place: bool = True,
        max_distance: int = DEFAULT_MAX_DISTANCE,
    ):
        self._original = G
        self._in_place = bool(in_place)
        self._max_distance = int(max_distance)
        self._graph: nx.Graph = G if in_place else G.copy()

    def annotate(self) -> "GraphAnnotator":
        self._degrees()
        self._neighbor_counts()
        self._rings_via_cycle_basis()
        self._distances_to_motifs()
        self._pi_components()
        return self

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def get_node(self, n: Node) -> NodeData:
        return self._graph.nodes[n]

    def get_node_attr(self, n: Node, attr: str, default: Any = None) -> Any:
        return self._graph.nodes[n].get(attr, default)

    def _degrees(self) -> "GraphAnnotator":
        try:
            for n in self._graph.nodes():
                self._graph.nodes[n]["atom_degree"] = int(self._graph.degree(n))
        except Exception as exc:
            logger.debug("Failed to annotate degrees: %s", exc, exc_info=True)
        return self

    def _neighbor_counts(self) -> "GraphAnnotator":
        try:
            for n in self._graph.nodes():
                counts: Dict[str, int] = {}
                for nb in self._graph.neighbors(n):
                    el = self._graph.nodes[nb].get("element", "C")
                    counts[el] = counts.get(el, 0) + 1
                self._graph.nodes[n]["nbr_elements_counts_r1"] = counts
        except Exception as exc:
            logger.debug("Failed to compute neighbor counts: %s", exc, exc_info=True)
        return self

    def _rings_via_cycle_basis(self) -> "GraphAnnotator":
        try:
            cycles = nx.cycle_basis(self._graph)
            per_node: Dict[Node, List[int]] = {}
            for cyc in cycles:
                size = len(cyc)
                for node in cyc:
                    per_node.setdefault(node, []).append(size)
            for n in self._graph.nodes():
                sizes = per_node.get(n, [])
                self._graph.nodes[n].setdefault("ring_sizes", sizes)
                self._graph.nodes[n].setdefault(
                    "in_ring", bool(sizes or self._graph.nodes[n].get("in_ring", False))
                )
        except Exception as exc:
            logger.debug(
                "Failed to compute ring_sizes via cycle_basis: %s", exc, exc_info=True
            )
            for n in self._graph.nodes():
                self._graph.nodes[n].setdefault(
                    "ring_sizes", self._graph.nodes[n].get("ring_sizes", [])
                )
                self._graph.nodes[n].setdefault(
                    "in_ring", bool(self._graph.nodes[n].get("in_ring", False))
                )
        return self

    def _pi_components(self) -> "GraphAnnotator":
        try:
            PG = nx.Graph()
            for u, v, ed in self._graph.edges(data=True):
                try:
                    order = float(ed.get("order", 1.0))
                except Exception:
                    order = 1.0
                conj = bool(ed.get("conjugated", False))
                arom = bool(
                    self._graph.nodes[u].get("aromatic", False)
                    and self._graph.nodes[v].get("aromatic", False)
                )
                if order >= 2.0 or conj or arom:
                    PG.add_edge(u, v)
            for n, d in self._graph.nodes(data=True):
                if d.get("aromatic", False) and n not in PG:
                    PG.add_node(n)

            comp_size: Dict[Node, int] = {}
            for comp in nx.connected_components(PG):
                s = len(comp)
                for node in comp:
                    comp_size[node] = s

            for n in self._graph.nodes():
                self._graph.nodes[n]["conj_component_size"] = int(comp_size.get(n, 0))
        except Exception as exc:
            logger.debug("Failed to compute pi components: %s", exc, exc_info=True)
            for n in self._graph.nodes():
                self._graph.nodes[n].setdefault("conj_component_size", 0)
        return self

    # --- New small motif collectors (each purposely tiny) -----------------

    def _collect_halogens(self) -> Set[Node]:
        try:
            return {
                n
                for n, d in self._graph.nodes(data=True)
                if d.get("element") in {"F", "Cl", "Br", "I"} or d.get("is_halogen")
            }
        except Exception:
            logger.debug("Failed to collect halogens", exc_info=True)
            return set()

    def _collect_hetero(self) -> Set[Node]:
        try:
            return {
                n
                for n, d in self._graph.nodes(data=True)
                if d.get("element") not in {"C", "H"}
            }
        except Exception:
            logger.debug("Failed to collect heteroatoms", exc_info=True)
            return set()

    def _collect_aromatic(self) -> Set[Node]:
        try:
            return {
                n for n, d in self._graph.nodes(data=True) if d.get("aromatic", False)
            }
        except Exception:
            logger.debug("Failed to collect aromatic nodes", exc_info=True)
            return set()

    def _collect_carbonyl(self) -> Set[Node]:
        carbonyl: Set[Node] = set()
        try:
            for u, v, ed in self._graph.edges(data=True):
                try:
                    order = float(ed.get("order", 1.0))
                except Exception:
                    order = 1.0
                elems = {
                    self._graph.nodes[u].get("element"),
                    self._graph.nodes[v].get("element"),
                }
                if order >= 2.0 and elems == {"C", "O"}:
                    if self._graph.nodes[u].get("element") == "C":
                        carbonyl.add(u)
                    if self._graph.nodes[v].get("element") == "C":
                        carbonyl.add(v)
        except Exception:
            logger.debug("Failed to collect carbonyls", exc_info=True)
            return set()
        return carbonyl

    def _collect_motif_sets(self) -> Dict[str, Set[Node]]:
        """
        Collect node sets for motifs: halogen, hetero, aromatic, carbonyl.

        Delegates to micro-helpers to keep function complexity low.
        """
        return {
            "halogen": self._collect_halogens(),
            "hetero": self._collect_hetero(),
            "aromatic": self._collect_aromatic(),
            "carbonyl": self._collect_carbonyl(),
        }

    def _compute_distances_for_motif(
        self, sources: Set[Node], cutoff: int
    ) -> Dict[Node, int]:
        if not sources:
            return {}
        try:
            return dict(
                nx.multi_source_shortest_path_length(
                    self._graph, sources, cutoff=cutoff
                )
            )
        except Exception as exc:
            logger.debug(
                "multi_source_shortest_path_length failed: %s", exc, exc_info=True
            )
            return {}

    def _fallback_distances(
        self, motifs: Dict[str, Set[Node]], cutoff: int
    ) -> Dict[str, Dict[Node, int]]:
        distances_by_motif: Dict[str, Dict[Node, int]] = {}
        try:
            all_sp = dict(nx.all_pairs_shortest_path_length(self._graph, cutoff=cutoff))
            for name, sources in motifs.items():
                db: Dict[Node, int] = {}
                if not sources:
                    distances_by_motif[name] = db
                    continue
                for node in self._graph.nodes():
                    dmap = all_sp.get(node, {})
                    found = [dmap.get(s) for s in sources if s in dmap]
                    db[node] = min(found) if found else cutoff
                distances_by_motif[name] = db
            return distances_by_motif
        except Exception as exc:
            logger.debug(
                "Fallback all-pairs shortest path failed: %s", exc, exc_info=True
            )
            return {name: {} for name in motifs.keys()}

    def _distances_to_motifs(self) -> "GraphAnnotator":
        maxd = int(self._max_distance)
        try:
            motifs = self._collect_motif_sets()
            distances_by_motif: Dict[str, Dict[Node, int]] = {}
            for name, sources in motifs.items():
                distances_by_motif[name] = self._compute_distances_for_motif(
                    sources, cutoff=maxd
                )

            need_fallback = any(
                len(motifs[name]) > 0 and not distances_by_motif.get(name)
                for name in motifs.keys()
            )
            if need_fallback:
                distances_by_motif = self._fallback_distances(motifs, cutoff=maxd)

            for n in self._graph.nodes():
                for name in motifs.keys():
                    dmap = distances_by_motif.get(name, {})
                    dist = dmap.get(n, maxd)
                    self._graph.nodes[n][f"dist_to_{name}"] = int(dist)
        except Exception as exc:
            logger.debug(
                "Failed to compute distances to motifs: %s", exc, exc_info=True
            )
            for n in self._graph.nodes():
                for name in ("halogen", "hetero", "aromatic", "carbonyl"):
                    self._graph.nodes[n].setdefault(f"dist_to_{name}", maxd)
        return self

    def __repr__(self) -> str:
        try:
            n = self._graph.number_of_nodes()
        except Exception:
            n = -1
        return f"{self.__class__.__name__}(in_place={self._in_place}, n_nodes={n})"

    @classmethod
    def help(cls) -> str:
        return (
            "GraphAnnotator.help() -> str\n\n"
            "This annotator computes: atom_degree, nbr_elements_counts_r1, dist_to_<motif>,\n"
            "conj_component_size, ring_sizes and in_ring. Call `.annotate()` to run\n"
            "the pipeline and retrieve results via `.graph`."
        )
