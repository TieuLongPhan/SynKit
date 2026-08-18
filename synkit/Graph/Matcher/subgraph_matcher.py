"""Connected-component-aware subgraph matching utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Optional, Sequence, Tuple, Callable, Union
from operator import eq
import networkx as nx

from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Synthesis.Reactor import Strategy

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
    - an attribute explicitly marked ``unknown`` by the typed query policy is
      omitted because the template's local context cannot determine it

    ``radical`` otherwise remains exact whenever the caller includes it.
    """
    for attr in node_attrs:
        policy = pattern_data.get("_query_attribute_policies", {}).get(
            attr,
            "exact",
        )
        if policy == "unknown":
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
        policy = pattern_data.get("_query_attribute_policies", {}).get(
            attr,
            "exact",
        )
        if policy == "unknown":
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

            node_comparator = node_comparator or eq
            edge_comparator = edge_comparator or eq

            for _, child_data in child_graph.nodes(data=True):
                found_match = False
                for _, parent_data in parent_graph.nodes(data=True):
                    match = True
                    for label, default in zip(node_label_names, node_label_default):
                        if not node_comparator(
                            parent_data.get(label, default),
                            child_data.get(label, default),
                        ):
                            match = False
                            break
                    if match:
                        found_match = True
                        break
                if not found_match:
                    return False

            if edge_attribute:
                parent_edge_values = [
                    data.get(edge_attribute)
                    for _, _, data in parent_graph.edges(data=True)
                ]
                for _, _, child_data in child_graph.edges(data=True):
                    child_value = child_data.get(edge_attribute)
                    if not any(
                        edge_comparator(parent_value, child_value)
                        for parent_value in parent_edge_values
                    ):
                        return False

        node_comparator = node_comparator or eq
        edge_comparator = edge_comparator or eq

        node_match = generic_node_match(
            node_label_names,
            node_label_default,
            [node_comparator] * len(node_label_names),
        )
        edge_match = generic_edge_match(edge_attribute, None, edge_comparator)

        if child_graph.is_directed() != parent_graph.is_directed():
            return False
        if child_graph.is_multigraph() or parent_graph.is_multigraph():
            raise NotImplementedError("SubgraphMatch does not support multigraphs")
        matcher_cls = (
            nx.algorithms.isomorphism.DiGraphMatcher
            if parent_graph.is_directed()
            else GraphMatcher
        )
        matcher = matcher_cls(
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
    _COMPILED_MATCH_CAP: int = 1_000_000

    @staticmethod
    def _quick_pre_filter(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
    ) -> bool:
        """Return whether a degree-aware node domain proves no embedding exists.

        A Cartesian product of candidate-domain sizes is only a loose upper
        bound on the number of embeddings.  It must not be compared with the
        result threshold: graph connectivity can reduce a very large product
        to only a handful of valid mappings.  The enumerator itself enforces
        ``threshold`` safely.
        """
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
        return False

    @staticmethod
    def _find_compiled_domain_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
    ) -> Optional[List[MappingDict]]:
        """Enumerate an exact node-domain query with RDKit's compiled matcher.

        Each host atom receives an isotope identifying the complete set of
        pattern nodes accepted by ``electron_aware_node_match``. A pattern
        query atom is the OR of precisely those isotope domains that contain
        it. Thus the compiled query preserves the authoritative node predicate
        exactly; edge attributes are checked by the established predicate on
        every returned topological embedding.

        ``None`` requests the NetworkX fallback for unsupported graph kinds or
        if the defensive raw-match cap is reached. Consequently the cap can
        never truncate a returned mapping population.
        """
        if (
            host.is_directed()
            or pattern.is_directed()
            or host.is_multigraph()
            or pattern.is_multigraph()
            or nx.number_of_selfloops(host)
            or nx.number_of_selfloops(pattern)
        ):
            return None
        if not pattern:
            return [{}]

        from rdkit import Chem
        from rdkit.Chem import rdqueries

        pattern_nodes = tuple(pattern)
        domain_ids: Dict[Tuple[Any, ...], int] = {}
        host_domains: Dict[Any, int] = {}
        for host_node, host_attrs in host.nodes(data=True):
            domain = tuple(
                pattern_node
                for pattern_node in pattern_nodes
                if electron_aware_node_match(
                    host_attrs,
                    pattern.nodes[pattern_node],
                    node_attrs,
                )
            )
            if not domain:
                host_domains[host_node] = 0
                continue
            domain_id = domain_ids.get(domain)
            if domain_id is None:
                domain_id = len(domain_ids) + 1
                if domain_id > 65535:
                    return None
                domain_ids[domain] = domain_id
            host_domains[host_node] = domain_id

        allowed_domains = {
            pattern_node: tuple(
                domain_id
                for domain, domain_id in domain_ids.items()
                if pattern_node in domain
            )
            for pattern_node in pattern_nodes
        }
        if any(not domains for domains in allowed_domains.values()):
            return []

        host_builder = Chem.RWMol()
        host_nodes = tuple(host)
        host_index = {}
        for node in host_nodes:
            atom = Chem.Atom(0)
            atom.SetIsotope(host_domains[node])
            host_index[node] = host_builder.AddAtom(atom)
        for left, right in host.edges():
            host_builder.AddBond(
                host_index[left],
                host_index[right],
                Chem.BondType.SINGLE,
            )

        query_builder = Chem.RWMol()
        query_index = {}
        for node in pattern_nodes:
            domains = allowed_domains[node]
            query = rdqueries.IsotopeEqualsQueryAtom(domains[0])
            for domain_id in domains[1:]:
                query.ExpandQuery(
                    rdqueries.IsotopeEqualsQueryAtom(domain_id),
                    Chem.CompositeQueryType.COMPOSITE_OR,
                )
            query_index[node] = query_builder.AddAtom(query)
        for left, right in pattern.edges():
            query_builder.AddBond(
                query_index[left],
                query_index[right],
                Chem.BondType.SINGLE,
            )

        matches = host_builder.GetMol().GetSubstructMatches(
            query_builder.GetMol(),
            uniquify=False,
            useChirality=False,
            maxMatches=SubgraphSearchEngine._COMPILED_MATCH_CAP,
        )
        if len(matches) == SubgraphSearchEngine._COMPILED_MATCH_CAP:
            return None

        results = []
        for match in matches:
            mapping = {
                pattern_nodes[index]: host_nodes[host_atom_index]
                for index, host_atom_index in enumerate(match)
            }
            if all(
                electron_aware_edge_match(
                    host.edges[mapping[left], mapping[right]],
                    attrs,
                    edge_attrs,
                )
                for left, right, attrs in pattern.edges(data=True)
            ):
                results.append(mapping)
        return results

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
        threshold: Optional[int] = DEFAULT_THRESHOLD,
        pre_filter: bool = False,
    ) -> List[MappingDict]:
        """Dispatch to a subgraph-matching strategy with optional guards.

        :param host: NetworkX graphs (host ≥ pattern).
        :param pattern: NetworkX graphs (host ≥ pattern).
        :param node_attrs: Keys of attributes to match; ``hcount`` and ``lone_pairs`` use
                           host-greater-or-equal semantics, while the rest are exact.
        :param edge_attrs: Keys of attributes to match; ``hcount`` and ``lone_pairs`` use
                           host-greater-or-equal semantics, while the rest are exact.
        :param strategy: Matching strategy code or enum ("all", "comp", "bt").
        :param max_results: Stop after this many embeddings (None = no limit).
        :param strict_cc_count: If True, host CC count must ≤ pattern CC count for COMPONENT/BACKTRACK.
        :param threshold: Embedding cap. Passing ``None`` disables the cap; omitting the
                          argument uses ``DEFAULT_THRESHOLD``.
        :param pre_filter: If True, reject patterns having an empty candidate-node domain.

        :return: * *List of dictionaries mapping pattern node→host node. Empty if none or*
                  * *if any guard (pre-filter or enumeration) exceeds the threshold.*
        """
        strat = Strategy.from_string(strategy)
        if strat is Strategy.PARTIAL:
            raise NotImplementedError("PARTIAL strategy not implemented yet.")
        if host.is_directed() != pattern.is_directed():
            return []
        if host.is_multigraph() or pattern.is_multigraph():
            raise NotImplementedError(
                "SubgraphSearchEngine does not support multigraphs"
            )
        if max_results is not None and max_results < 0:
            raise ValueError("max_results must be non-negative or None")
        if threshold is not None and threshold < 0:
            raise ValueError("threshold must be non-negative or None")
        if max_results == 0:
            return []

        thresh = threshold

        # All matching strategies below treat the host and pattern as
        # read-only.  Avoid copying both complete graphs for every search;
        # component-specific working graphs are still materialized where a
        # strategy needs them.

        # The compiled matcher constructs the complete candidate-node domains
        # itself. Running the Python pre-filter first would evaluate the same
        # authoritative predicate twice. If compilation is unsupported or
        # reaches its defensive cap, retain the established guarded fallback.
        compiled_results = (
            SubgraphSearchEngine._find_compiled_domain_mappings(
                host,
                pattern,
                node_attrs,
                edge_attrs,
            )
            if strat is Strategy.ALL and max_results is None and thresh is None
            else None
        )
        if compiled_results is not None:
            return compiled_results

        # quick pre-filter
        if pre_filter and SubgraphSearchEngine._quick_pre_filter(
            host, pattern, node_attrs
        ):
            return []

        # dispatch
        if strat is Strategy.ALL:
            components = list(
                nx.weakly_connected_components(pattern)
                if pattern.is_directed()
                else nx.connected_components(pattern)
            )
            if len(components) > 1:
                results = SubgraphSearchEngine._find_disconnected_all_mappings(
                    host,
                    pattern,
                    components,
                    node_attrs,
                    edge_attrs,
                    max_results,
                    thresh,
                )
            else:
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
        return [] if thresh is not None and len(results) > thresh else results

    @staticmethod
    def _find_all_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int],
        threshold: Optional[int],
    ) -> List[MappingDict]:
        """Classic VF2 over the whole host graph."""

        def node_match(nh: EdgeAttr, np: EdgeAttr) -> bool:
            return electron_aware_node_match(nh, np, node_attrs)

        def edge_match(eh: EdgeAttr, ep: EdgeAttr) -> bool:
            return electron_aware_edge_match(eh, ep, edge_attrs)

        matcher_cls = (
            nx.algorithms.isomorphism.DiGraphMatcher
            if host.is_directed()
            else GraphMatcher
        )
        gm = matcher_cls(host, pattern, node_match=node_match, edge_match=edge_match)
        results: List[MappingDict] = []
        for iso in gm.subgraph_monomorphisms_iter():
            results.append({p: h for h, p in iso.items()})
            if max_results is not None and len(results) >= max_results:
                break
            if threshold is not None and len(results) > threshold:
                return []
        return results

    @staticmethod
    def _find_disconnected_all_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        components: List[Set[Any]],
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int],
        threshold: Optional[int],
    ) -> List[MappingDict]:
        """Join exact component embeddings with global injectivity.

        VF2's whole-pattern search repeatedly traverses unrelated host context
        for disconnected queries. Component embeddings are independent except
        for injectivity, so their disjoint Cartesian join is exactly the same
        monomorphism set, including placements in one host component.
        """
        per_component = []
        for nodes in components:
            component = pattern.subgraph(nodes).copy()
            mappings = SubgraphSearchEngine._find_all_subgraph_mappings(
                host,
                component,
                node_attrs,
                edge_attrs,
                None,
                None,
            )
            if not mappings:
                return []
            per_component.append((component.number_of_nodes(), mappings))
        per_component.sort(key=lambda item: (len(item[1]), -item[0]))

        results: List[MappingDict] = []

        def join(level: int, combined: MappingDict, used: Set[Any]) -> None:
            if max_results is not None and len(results) >= max_results:
                return
            if threshold is not None and len(results) > threshold:
                return
            if level == len(per_component):
                results.append(combined.copy())
                return
            for mapping in per_component[level][1]:
                image = set(mapping.values())
                if image & used:
                    continue
                combined.update(mapping)
                join(level + 1, combined, used | image)
                for node in mapping:
                    combined.pop(node)
                if max_results is not None and len(results) >= max_results:
                    return
                if threshold is not None and len(results) > threshold:
                    return

        join(0, {}, set())
        return results

    @staticmethod
    def _find_component_aware_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        max_results: Optional[int],
        strict_cc_count: bool,
        threshold: Optional[int],
    ) -> List[MappingDict]:
        """Component-aware VF2 split by connected components."""
        host_components = (
            nx.weakly_connected_components(host)
            if host.is_directed()
            else nx.connected_components(host)
        )
        pattern_components = (
            nx.weakly_connected_components(pattern)
            if pattern.is_directed()
            else nx.connected_components(pattern)
        )
        host_ccs = [host.subgraph(component).copy() for component in host_components]
        pat_ccs = [
            pattern.subgraph(component).copy() for component in pattern_components
        ]
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
                matcher_cls = (
                    nx.algorithms.isomorphism.DiGraphMatcher
                    if host.is_directed()
                    else GraphMatcher
                )
                gm = matcher_cls(
                    host_ccs[i], pc, node_match=node_match, edge_match=edge_match
                )
                # ``max_results`` bounds complete joined embeddings, not the
                # number of candidate host components considered here.  Keep
                # up to that many local embeddings *per component pair* so a
                # later injective assignment can still choose another host CC.
                host_maps = 0
                for iso in gm.subgraph_monomorphisms_iter():
                    maps.append((i, {p: h for h, p in iso.items()}))
                    host_maps += 1
                    if max_results is not None and host_maps >= max_results:
                        break
                    if threshold is not None and len(maps) > threshold:
                        return []
            if not maps:
                return []
            per_cc.append(maps)

        order = sorted(range(pcc), key=lambda i: len(per_cc[i]))
        ordered = [per_cc[i] for i in order]
        results: List[MappingDict] = []
        used: Set[int] = set()

        def backtrack(level: int, acc: MappingDict):
            if max_results is not None and len(results) >= max_results:
                return
            if threshold is not None and len(results) > threshold:
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
                if max_results is not None and len(results) >= max_results:
                    return
                if threshold is not None and len(results) > threshold:
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
        threshold: Optional[int],
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

    @property
    def help(self) -> str:  # noqa: D401 – property for convenience
        """Return the full module docstring."""

        return __doc__
