from __future__ import annotations

import networkx as nx
from typing import Tuple, List, Dict, Any, Optional, Iterable, Set, Mapping
from collections import defaultdict
from .rxn import RXNSide
from .hypergraph import CRNHyperGraph


# ======================================================================
# Pair 1: Hypergraph  <->  Bipartite
# ======================================================================


def hypergraph_to_bipartite(
    H: CRNHyperGraph,
    *,
    species_prefix: Optional[str] = "S:",
    reaction_prefix: Optional[str] = "R:",
    bipartite_values: Tuple[int, int] = (0, 1),
    include_stoich: bool = True,
    include_role: bool = True,
    include_isolated_species: bool = True,
    integer_ids: bool = False,
    include_edge_id_attr: bool = False,
    include_mol: bool = False,
) -> nx.DiGraph:
    """
    Export a CRN hypergraph to a **bipartite** NetworkX DiGraph
    with arcs ``species → reaction → species``.

    :param H: Hypergraph to export.
    :type H: CRNHyperGraph
    :param species_prefix: Optional prefix for species node ids when ``integer_ids=False``.
                           If ``None``, the species label is used as-is.
    :type species_prefix: Optional[str]
    :param reaction_prefix: Optional prefix for reaction node ids when ``integer_ids=False``.
                            If ``None``, the edge id is used as-is.
    :type reaction_prefix: Optional[str]
    :param bipartite_values: Bipartite marker values ``(species_value, reaction_value)``.
    :type bipartite_values: Tuple[int, int]
    :param include_stoich: If ``True``, add integer edge attribute ``'stoich'``.
    :type include_stoich: bool
    :param include_role: If ``True``, add edge attribute ``'role'`` in {``reactant``, ``product``}.
    :type include_role: bool
    :param include_isolated_species: If ``True``, keep species with no incident edges.
    :type include_isolated_species: bool
    :param integer_ids: If ``True``, species ids are ``1..N`` and reactions ``N+1..N+M``.
    :type integer_ids: bool
    :param include_edge_id_attr: If ``True``, store reaction edge id in node attr ``'edge_id'``.
    :type include_edge_id_attr: bool
    :param include_mol: If ``True``, include species-to-molecule mapping from
                        ``H.species_to_mol`` as node attribute ``'mol'``.
    :type include_mol: bool
    :returns: Bipartite DiGraph.
    :rtype: nx.DiGraph

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_bipartite
    >>> H = CRNHyperGraph().parse_rxns(["A + B >> C", "C >> A"])
    >>> G = hypergraph_to_bipartite(H, integer_ids=False)
    >>> set(nx.get_node_attributes(G, "kind").values()) == {"species", "reaction"}
    True
    """
    G = nx.DiGraph()
    species_val, reaction_val = bipartite_values

    species_to_mol: Optional[Mapping[str, Any]] = None
    if include_mol and hasattr(H, "species_to_mol"):
        species_to_mol = H.species_to_mol

    if include_isolated_species:
        species_iter = sorted(H.species)
    else:
        species_iter = sorted(
            {
                s
                for s in H.species
                if (H.species_to_in_edges.get(s) or H.species_to_out_edges.get(s))
            }
        )

    def make_sp_attrs(s: str) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "bipartite": species_val,
            "label": s,
            "kind": "species",
        }
        if include_mol and species_to_mol is not None and s in species_to_mol:
            attrs["mol"] = species_to_mol[s]
        return attrs

    def make_rxn_attrs(eid: str, rule: str) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "bipartite": reaction_val,
            "label": rule,
            "kind": "reaction",
        }
        if include_edge_id_attr:
            attrs["edge_id"] = eid
        return attrs

    species_map: Dict[str, Any] = {}
    next_id = 1

    def add_sp_node(s: str) -> Any:
        nonlocal next_id
        if s in species_map:
            return species_map[s]
        if integer_ids:
            nid = next_id
            next_id += 1
        else:
            nid = f"{species_prefix}{s}" if species_prefix is not None else s
        species_map[s] = nid
        if not G.has_node(nid):
            G.add_node(nid, **make_sp_attrs(s))
        return nid

    def add_rxn_node(eid: str, rule: str) -> Any:
        nonlocal next_id
        if integer_ids:
            nid = next_id
            next_id += 1
        else:
            nid = f"{reaction_prefix}{eid}" if reaction_prefix is not None else eid
        G.add_node(nid, **make_rxn_attrs(eid, rule))
        return nid

    # add species nodes
    for s in species_iter:
        add_sp_node(s)

    # add reactions and incidence edges
    for eid, e in sorted(H.edges.items()):
        rnode = add_rxn_node(eid, e.rule)
        for s, c in e.reactants.items():
            u = add_sp_node(s)
            attrs: Dict[str, Any] = {}
            if include_stoich:
                attrs["stoich"] = int(c)
            if include_role:
                attrs["role"] = "reactant"
            G.add_edge(u, rnode, **attrs)
        for s, c in e.products.items():
            v = add_sp_node(s)
            attrs = {}
            if include_stoich:
                attrs["stoich"] = int(c)
            if include_role:
                attrs["role"] = "product"
            G.add_edge(rnode, v, **attrs)
    return G


def bipartite_to_hypergraph(
    G: nx.DiGraph,
    *,
    species_prefix: str = "S:",
    reaction_prefix: str = "R:",
    species_label_attr: str = "label",
    reaction_edge_id_attr: str = "edge_id",
    reaction_label_attr: str = "label",
    stoich_attr: str = "stoich",
    default_rule: str = "r",
    mol_attr: Optional[str] = "mol",
) -> CRNHyperGraph:
    """
    Reconstruct a **CRNHyperGraph** from a bipartite species→reaction→species graph.

    The function is the logical inverse of :func:`hypergraph_to_bipartite` and
    supports graphs produced by it, while attempting a best-effort reconstruction
    for general bipartite-like graphs.

    :param G: Bipartite graph (``species`` and ``reaction`` nodes).
    :type G: nx.DiGraph
    :param species_prefix: Prefix to detect species ids when kind/prefix absent.
    :type species_prefix: str
    :param reaction_prefix: Prefix to detect reaction ids when kind/prefix absent.
    :type reaction_prefix: str
    :param species_label_attr: Node attribute holding species label (default: ``label``).
    :type species_label_attr: str
    :param reaction_edge_id_attr: Node attribute holding original edge id (default: ``edge_id``).
    :type reaction_edge_id_attr: str
    :param reaction_label_attr: Node attribute holding reaction rule/label (default: ``label``).
    :type reaction_label_attr: str
    :param stoich_attr: Edge attribute for stoichiometry (default: ``stoich``).
    :type stoich_attr: str
    :param default_rule: Fallback rule label when none found.
    :type default_rule: str
    :param mol_attr: Node attribute name used for molecule identifiers (default: ``"mol"``).
                     If ``None``, skip reconstruction of ``species_to_mol``.
    :type mol_attr: Optional[str]
    :returns: Reconstructed hypergraph.
    :rtype: CRNHyperGraph

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_bipartite, bipartite_to_hypergraph
    >>> H0 = CRNHyperGraph().parse_rxns(["A+B>>C", "C>>A"])
    >>> B = hypergraph_to_bipartite(H0, integer_ids=False, include_edge_id_attr=True)
    >>> H1 = bipartite_to_hypergraph(B)
    >>> sorted(H0.species) == sorted(H1.species)
    True
    """
    H = CRNHyperGraph()

    # classify nodes
    species_nodes: Set[Any] = set()
    reaction_nodes: Set[Any] = set()
    for n, d in G.nodes(data=True):
        kind = d.get("kind")
        if kind == "species":
            species_nodes.add(n)
        elif kind == "reaction":
            reaction_nodes.add(n)
        else:
            # fallback to prefix heuristics
            if isinstance(n, str) and n.startswith(species_prefix):
                species_nodes.add(n)
            elif isinstance(n, str) and n.startswith(reaction_prefix):
                reaction_nodes.add(n)

    # Further fallback: very permissive guesses if tags are missing.
    if not species_nodes and not reaction_nodes:
        for n in G.nodes():
            outdeg = G.out_degree(n)
            indeg = G.in_degree(n)
            if outdeg > 0:
                species_nodes.add(n)
            if indeg > 0:
                reaction_nodes.add(n)

    # Build mapping from reaction node -> reactant/product lists
    for rnode in sorted(reaction_nodes):
        # collect reactants (incoming edges)
        reactants_map: Dict[str, int] = {}
        for u, _, ed in G.in_edges(rnode, data=True):
            if u not in species_nodes:
                continue
            sto = int(ed.get(stoich_attr, 1))
            s_label = G.nodes[u].get(species_label_attr, str(u))
            reactants_map[s_label] = reactants_map.get(s_label, 0) + sto

        # collect products (outgoing edges)
        products_map: Dict[str, int] = {}
        for _, v, ed in G.out_edges(rnode, data=True):
            if v not in species_nodes:
                continue
            sto = int(ed.get(stoich_attr, 1))
            s_label = G.nodes[v].get(species_label_attr, str(v))
            products_map[s_label] = products_map.get(s_label, 0) + sto

        # determine edge id and rule
        node_data = G.nodes[rnode]
        eid = node_data.get(reaction_edge_id_attr)
        rule = node_data.get(reaction_label_attr, default_rule)

        # synthesize readable id if no explicit one exists
        if eid is None:
            eid = f"{rule}_{abs(hash((rnode, tuple(sorted(reactants_map.items())), tuple(sorted(products_map.items()))))) % (10**8)}"

        if reactants_map or products_map:
            H.add_rxn(reactants_map, products_map, rule=rule, edge_id=str(eid))

    # Optionally reconstruct species_to_mol from node attribute mol_attr
    if mol_attr is not None:
        for s_node in species_nodes:
            ndata = G.nodes[s_node]
            if mol_attr not in ndata:
                continue
            s_label = ndata.get(species_label_attr, str(s_node))
            if s_label in H.species:
                H.species_to_mol[s_label] = ndata[mol_attr]

    return H


# ======================================================================
# Pair 2: Hypergraph  <->  Species Graph (collapsed)
# ======================================================================


def hypergraph_to_species_graph(
    H: CRNHyperGraph,
    *,
    include_mol: bool = False,
) -> nx.DiGraph:
    """
    Collapse hyperedges to a **species→species** DiGraph.

    Aggregated edge attributes:
      - ``via``: ``set`` of contributing hyperedge ids
      - ``rules``: ``set`` of contributing rule labels
      - ``stoich_r``: aggregated reactant-side stoichiometry
      - ``stoich_p``: aggregated product-side stoichiometry
      - ``stoich_r_map``: per-hyperedge reactant stoichiometry
                          (mapping ``{eid -> coeff}``)
      - ``stoich_p_map``: per-hyperedge product stoichiometry
                          (mapping ``{eid -> coeff}``)

    The per-hyperedge maps are required for exact reconstruction when the
    same species pair participates in multiple reactions. The legacy
    ``stoich_r`` / ``stoich_p`` fields are retained for backward
    compatibility and store the minimum coefficient observed for that
    direction.

    :param H: Hypergraph to collapse.
    :type H: CRNHyperGraph
    :param include_mol: If ``True``, propagate ``H.species_to_mol`` into node
                        attribute ``'mol'`` when available.
    :type include_mol: bool
    :returns: Directed species graph.
    :rtype: nx.DiGraph

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_species_graph
    >>> H = CRNHyperGraph().parse_rxns(["2A+B>>3C", "A>>D"])
    >>> S = hypergraph_to_species_graph(H)
    >>> ("A" in S) and ("C" in S) and S.has_edge("A", "C")
    True
    """
    G = nx.DiGraph()

    species_to_mol: Optional[Mapping[str, Any]] = None
    if include_mol and hasattr(H, "species_to_mol"):
        species_to_mol = H.species_to_mol

    # nodes
    for s in H.species:
        attrs: Dict[str, Any] = {"label": s, "kind": "species"}
        if include_mol and species_to_mol is not None and s in species_to_mol:
            attrs["mol"] = species_to_mol[s]
        G.add_node(s, **attrs)

    # edges
    for eid, e in H.edges.items():
        for r, rc in e.reactants.items():
            for p, pc in e.products.items():
                if G.has_edge(r, p):
                    data = G[r][p]

                    # via / rules sets
                    via = data.get("via")
                    if via is None:
                        via = set()
                        data["via"] = via
                    via.add(eid)

                    rules = data.get("rules")
                    if rules is None:
                        rules = set()
                        data["rules"] = rules
                    rules.add(e.rule)

                    # per-eid stoichiometry maps
                    sr_map = data.get("stoich_r_map")
                    if sr_map is None:
                        sr_map = {}
                        data["stoich_r_map"] = sr_map
                    sr_map[eid] = rc

                    sp_map = data.get("stoich_p_map")
                    if sp_map is None:
                        sp_map = {}
                        data["stoich_p_map"] = sp_map
                    sp_map[eid] = pc

                    # legacy aggregate values (keep min to stay conservative)
                    prev_sr = data.get("stoich_r")
                    prev_sp = data.get("stoich_p")
                    data["stoich_r"] = rc if prev_sr is None else min(prev_sr, rc)
                    data["stoich_p"] = pc if prev_sp is None else min(prev_sp, pc)

                else:
                    # first time we see this (r, p) pair
                    G.add_edge(
                        r,
                        p,
                        via={eid},
                        rules={e.rule},
                        stoich_r=rc,
                        stoich_p=pc,
                        stoich_r_map={eid: rc},
                        stoich_p_map={eid: pc},
                    )
    return G


def species_graph_to_hypergraph(
    G: nx.DiGraph,
    *,
    default_rule: str = "r",
    mol_attr: Optional[str] = "mol",
    species_label_attr: str = "label",
) -> CRNHyperGraph:
    """
    Reconstruct a :class:`CRNHyperGraph` from a collapsed species→species graph.

    If edges expose ``'via'`` sets (carrying original hyperedge ids), arcs that
    share the same ``via`` id are grouped back into one hyperedge. If no ``via``
    is present, each species arc becomes its own hyperedge.

    Stoichiometry can be provided in two forms:

      * **Per-eid maps** (recommended):

        - ``stoich_r_map``: mapping ``{eid -> coeff}`` for the reactant side.
        - ``stoich_p_map``: mapping ``{eid -> coeff}`` for the product side.

        These are used if present and allow exact round-trips even when the
        same species pair participates in multiple reactions.

      * **Legacy single values**:

        - ``stoich_r``: single reactant-side coefficient for the arc.
        - ``stoich_p``: single product-side coefficient for the arc.

        These are used as a fallback when per-eid maps are absent. In this
        case, all eids in ``via`` for that arc share the same stoichiometry,
        which may not be sufficient to fully reconstruct the original
        hypergraph.

    When the species graph is produced by :func:`hypergraph_to_species_graph`
    with per-eid stoichiometry (``stoich_r_map`` / ``stoich_p_map``), this
    function reconstructs the original :class:`CRNHyperGraph` (up to generated
    edge ids if ``via`` is absent).

    If ``mol_attr`` is not ``None`` and nodes carry that attribute, the values
    are copied into ``H.species_to_mol`` keyed by species label.

    Species labels are taken from the node attribute ``species_label_attr``
    (default: ``"label"``). This allows you to canonicalize the species graph
    (integer node ids 1..N) and still reconstruct a hypergraph with species
    named ``"A", "B", ...`` instead of ``1, 2, ...``.

    :param G: Species→species DiGraph (typically from :func:`hypergraph_to_species_graph`).
    :type G: nx.DiGraph
    :param default_rule: Fallback rule for reconstructed edges.
    :type default_rule: str
    :param mol_attr: Node attribute name used for molecule identifiers (default: ``"mol"``).
                     If ``None``, skip reconstruction of ``species_to_mol``.
    :type mol_attr: Optional[str]
    :param species_label_attr: Node attribute holding species labels (default: ``"label"``).
    :type species_label_attr: str
    :returns: Best-effort reconstructed hypergraph.
    :rtype: CRNHyperGraph
    """
    H = CRNHyperGraph()

    # First pass: group edges by hyperedge id (via)
    eid_map: Dict[str, Dict[str, Any]] = {}

    for u, v, attrs in G.edges(data=True):
        # Map node ids to species *labels* (important after canonical relabeling)
        s_r = G.nodes[u].get(species_label_attr, str(u))
        s_p = G.nodes[v].get(species_label_attr, str(v))

        via = attrs.get("via")
        rules = attrs.get("rules", None)

        # New: per-eid stoichiometry maps (preferred if present)
        sr_map = attrs.get("stoich_r_map")
        sp_map = attrs.get("stoich_p_map")

        # Legacy: single stoichiometry values per species arc
        sr_legacy = attrs.get("stoich_r")
        sp_legacy = attrs.get("stoich_p")

        if via:
            if isinstance(via, (set, list, tuple)):
                eids = list(via)
            else:
                eids = [via]
        else:
            # No 'via': each arc gets its own synthetic eid
            eid = f"edge_{abs(hash((u, v))) % (10**8)}"
            eids = [eid]

        for eid in eids:
            # Determine stoichiometry for *this* eid, preferring per-eid maps
            sr = None
            sp = None

            if isinstance(sr_map, dict):
                sr = sr_map.get(eid)
            if isinstance(sp_map, dict):
                sp = sp_map.get(eid)

            # Fall back to legacy per-arc values if per-eid are missing
            if sr is None:
                sr = sr_legacy
            if sp is None:
                sp = sp_legacy

            sr = int(sr) if sr is not None else 1
            sp = int(sp) if sp is not None else 1

            entry = eid_map.setdefault(
                str(eid),
                {
                    "reactants": defaultdict(list),
                    "products": defaultdict(list),
                    "rules": set(),
                },
            )

            entry["reactants"][s_r].append(sr)
            entry["products"][s_p].append(sp)

            if rules is not None:
                if isinstance(rules, set):
                    entry["rules"].update(rules)
                else:
                    entry["rules"].add(rules)

    # Materialize hyperedges
    for eid, data in eid_map.items():
        reactants_lists: Dict[str, List[int]] = data["reactants"]
        products_lists: Dict[str, List[int]] = data["products"]

        # In practice each list should have length 1; we keep the first value.
        reactants: Dict[str, int] = {s: vals[0] for s, vals in reactants_lists.items()}
        products: Dict[str, int] = {s: vals[0] for s, vals in products_lists.items()}

        rules: Set[Any] = data["rules"]
        rule = next(iter(rules)) if rules else default_rule

        H.add_rxn(reactants, products, rule=rule, edge_id=str(eid))

    # Optional: reconstruct species_to_mol from node attribute mol_attr
    if mol_attr is not None:
        for n, d in G.nodes(data=True):
            if mol_attr not in d:
                continue
            s_label = d.get(species_label_attr, str(n))
            if s_label in H.species:
                H.species_to_mol[s_label] = d[mol_attr]

    return H


# ======================================================================
# Pair 3: Reaction strings  <->  Hypergraph
# ======================================================================


def rxns_to_hypergraph(
    rxns: Iterable[str],
    *,
    default_rule: str = "r",
    parse_rule_from_suffix: bool = True,
    prefer_suffix: bool = False,
) -> CRNHyperGraph:
    """
    Convenience constructor: parse reaction strings into a hypergraph.

    :param rxns: Iterable of reaction strings (e.g., ``"A + B >> C"``).
                 Supports suffix ``"| rule=Rk"`` when enabled.
    :param default_rule: Fallback rule when none provided.
    :param parse_rule_from_suffix: If ``True``, read ``| rule=...`` suffix.
    :param prefer_suffix: If ``True``, suffix overrides explicit rule per line.
    :returns: Populated :class:`CRNHyperGraph`.

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph
    >>> H = rxns_to_hypergraph(["A + B >> C", "C >> A", "2 A >> D"])
    >>> sorted(H.species) == ["A", "B", "C", "D"]
    True
    """
    H = CRNHyperGraph()
    H.parse_rxns(
        rxns,
        default_rule=default_rule,
        parse_rule_from_suffix=parse_rule_from_suffix,
        prefer_suffix=prefer_suffix,
    )
    return H


def hypergraph_to_rxn_strings(
    H: CRNHyperGraph,
    *,
    include_rule_suffix: bool = True,
    include_edge_id: bool = False,
    sort: bool = True,
) -> List[str]:
    """
    Convert a hypergraph back to human-readable reaction strings.

    Each line is printed as ``LHS >> RHS`` and, if requested,
    suffixed with ``| rule=R`` and/or ``| id=EDGEID``.

    :param H: Hypergraph to render.
    :param include_rule_suffix: If ``True``, append ``| rule=...``.
    :param include_edge_id: If ``True``, append ``| id=...``.
    :param sort: If ``True``, sort by edge id for determinism.
    :returns: List of reaction strings.

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_rxn_strings
    >>> H = CRNHyperGraph().parse_rxns(["A+B>>C | rule=R1"])
    >>> lines = hypergraph_to_rxn_strings(H, include_rule_suffix=True)
    >>> any(">>" in ln for ln in lines)
    True
    """
    out: List[str] = []
    items = sorted(H.edges.items()) if sort else list(H.edges.items())
    for eid, e in items:

        def fmt(side: RXNSide) -> str:
            if not side.data:
                return "∅"
            parts: List[str] = []
            for s in sorted(side.data.keys()):
                c = int(side.data[s])
                parts.append(f"{s}" if c == 1 else f"{c}{s}")
            return " + ".join(parts)

        left = fmt(e.reactants)
        right = fmt(e.products)
        line = f"{left} >> {right}"
        suffix_parts: List[str] = []
        if include_rule_suffix and e.rule:
            suffix_parts.append(f"rule={e.rule}")
        if include_edge_id:
            suffix_parts.append(f"id={eid}")
        if suffix_parts:
            line = f"{line} | " + " ".join(suffix_parts)
        out.append(line)
    return out


# ======================================================================
# Helpers for bipartite CRN graphs
# ======================================================================


def _as_bipartite(
    crn: Any,
    *,
    species_prefix: str = "S:",
    reaction_prefix: str = "R:",
    integer_ids: bool = True,
    include_stoich: bool = True,
) -> nx.DiGraph:
    """
    Normalize input to a bipartite species/reaction graph.

    Accepted inputs
    ---------------
    - Objects exposing ``to_bipartite(...)``:
      Called as ``crn.to_bipartite(species_prefix=..., reaction_prefix=...,
      integer_ids=..., include_stoich=...)``.
    - :class:`CRNHyperGraph` → converted via :func:`hypergraph_to_bipartite`.
    - Any NetworkX graph instance, assumed to already carry the required
      node/edge attributes (``kind`` / ``bipartite``, ``role``, ``stoich``).

    :param crn: Hypergraph-like object or NetworkX graph.
    :type crn: Any
    :param species_prefix: Prefix for species node identifiers when conversion
        from a hypergraph is required.
    :type species_prefix: str
    :param reaction_prefix: Prefix for reaction node identifiers when
        conversion from a hypergraph is required.
    :type reaction_prefix: str
    :param integer_ids: Whether the converter may use integer node IDs
        internally.
    :type integer_ids: bool
    :param include_stoich: Whether to attach stoichiometric coefficients as
        edge attributes (if available).
    :type include_stoich: bool
    :returns: Bipartite species/reaction graph.
    :rtype: networkx.DiGraph
    :raises TypeError: If the input type is unsupported or a required
        converter is unavailable.
    """
    if isinstance(crn, CRNHyperGraph):
        return hypergraph_to_bipartite(
            crn,
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )

    if isinstance(
        crn,
        (
            nx.Graph,
            nx.DiGraph,
            nx.MultiGraph,
            nx.MultiDiGraph,
        ),
    ):
        return crn if isinstance(crn, nx.DiGraph) else nx.DiGraph(crn)

    raise TypeError(
        "Expected CRNHyperGraph or NetworkX graph with bipartite species/"
        "reaction nodes."
    )


# ======================================================================
# Helpers for species graphs
# ======================================================================


def _as_species_graph(crn: Any) -> nx.DiGraph:
    """
    Normalize input to a species→species directed graph.

    Accepted inputs
    ---------------
    - Objects exposing ``to_species_graph()``:
      Called directly and converted to :class:`networkx.DiGraph` if needed.
    - :class:`CRNHyperGraph`:
      Collapsed by traversing reactants/products of each hyperedge.
    - Bipartite NetworkX graphs:
      Species nodes have ``kind='species'`` and reaction nodes
      ``kind='reaction'``; edges species→reaction are reactants and
      reaction→species are products.
    - Plain species-level NetworkX graphs:
      Returned as-is, converted to :class:`networkx.DiGraph` if necessary.

    The resulting graph may carry edge attributes:

    - ``via``        – set of reaction IDs that induce the edge.
    - ``rules``      – set of rule identifiers.
    - ``min_stoich`` – minimum stoichiometric coefficient across the pair.

    :param crn: Hypergraph-like object, or species graph.
    :type crn: Any
    :returns: Collapsed species→species directed graph.
    :rtype: networkx.DiGraph
    :raises TypeError: If the input type is unsupported.
    """

    if isinstance(crn, CRNHyperGraph):
        return hypergraph_to_species_graph(crn)

    if isinstance(
        crn,
        (
            nx.Graph,
            nx.DiGraph,
            nx.MultiGraph,
            nx.MultiDiGraph,
        ),
    ):
        kinds = {d.get("kind") for _, d in crn.nodes(data=True)}

        # Explicitly reject bipartite CRN graphs here
        if "reaction" in kinds and "species" in kinds:
            raise TypeError(
                "Bipartite CRN graph detected. "
                "Provide hypergraph-like object, or collapse "
                "the bipartite graph explicitly before calling this helper."
            )

        # Otherwise assume this is already species-level
        return crn if isinstance(crn, nx.DiGraph) else nx.DiGraph(crn)

    raise TypeError("Expected a CRNHyperGraph, or a NetworkX graph (species-level).")


# ======================================================================
# Pretty-print helpers (not paired)
# ======================================================================


def print_species_summary(
    H: CRNHyperGraph,
    *,
    species: Optional[Iterable[str]] = None,
    show_counts: bool = True,
) -> None:
    """
    Pretty-print per-species incoming/outgoing incidence.

    :param H: Hypergraph to inspect.
    :param species: Optional subset of species to print.
    :param show_counts: If ``True``, print edge counts alongside lists.
    :returns: ``None``.
    """
    species_iter = species if species is not None else sorted(H.species)
    rows = []
    for s in species_iter:
        ins = sorted(H.species_to_in_edges.get(s, []))
        outs = sorted(H.species_to_out_edges.get(s, []))
        rows.append((s, ins, outs))

    if not rows:
        print("No species.")
        return
    longest = max((len(s) for s, _, _ in rows), default=7)
    header = f"{'Species'.ljust(longest)}   In-edges          Out-edges"
    print(header)
    print("-" * len(header))
    for s, ins, outs in rows:
        ins_str = ", ".join(ins) if ins else "—"
        outs_str = ", ".join(outs) if outs else "—"
        if show_counts:
            ic = len(ins)
            oc = len(outs)
            print(
                f"{s.ljust(longest)}         "
                f"[{ic:2d}] {ins_str:<10}   "
                f"[{oc:2d}] {outs_str}"
            )
        else:
            print(
                f"{s.ljust(longest)}          " f"{ins_str:<10}        " f"{outs_str}"
            )


def print_edge_list(
    H: CRNHyperGraph,
    *,
    edge_ids: Optional[Iterable[str]] = None,
    show_stoich: bool = True,
) -> None:
    """
    Pretty-print edges in the format: ``edge_id  rule  Reactants >> Products``.

    :param H: Hypergraph to inspect.
    :param edge_ids: Optional subset of edge ids to print.
    :param show_stoich: If ``True``, show coefficients; else names only.
    :returns: ``None``.
    """
    ids = list(edge_ids) if edge_ids is not None else sorted(H.edges.keys())
    if not ids:
        print("No edges.")
        return
    print("Edge id   Rule   Reactants >> Products")
    print("-" * 60)
    for eid in ids:
        e = H.edges[eid]
        if show_stoich:

            def fmt_side(d: Dict[str, int]) -> str:
                if not d:
                    return "∅"
                parts: List[str] = []
                for s in sorted(d.keys()):
                    c = d[s]
                    parts.append(f"{s}" if c == 1 else f"{c}{s}")
                return " + ".join(parts)

            left = fmt_side(e.reactants.to_dict())
            right = fmt_side(e.products.to_dict())
        else:
            left = ", ".join(sorted(e.reactants.keys())) if e.reactants.keys() else "∅"
            right = ", ".join(sorted(e.products.keys())) if e.products.keys() else "∅"
        print(f"{eid:<8}  {e.rule:<6} {left}  >>  {right}")


def print_graph_attrs(
    G: nx.DiGraph,
    *,
    include_nodes: bool = True,
    include_edges: bool = True,
    max_rows: Optional[int] = None,
    use_labels: bool = False,
    label_attr: str = "label",
) -> None:
    """
    Pretty-print node and edge attributes of a NetworkX DiGraph.

    When ``use_labels=True``, node identifiers are *displayed* using the
    node attribute given by ``label_attr`` (default: ``"label"``), falling
    back to the raw node id if the attribute is missing. This does not
    modify the graph itself; it only affects how nodes are printed.

    :param G: Graph whose attributes should be printed.
    :type G: nx.DiGraph
    :param include_nodes: If True, print node table.
    :type include_nodes: bool
    :param include_edges: If True, print edge table.
    :type include_edges: bool
    :param max_rows: Optional maximum number of rows to print for nodes
                     and edges; extra rows are summarized with an ellipsis.
    :type max_rows: Optional[int]
    :param use_labels: If True, print nodes using the value of ``label_attr``
                       instead of the raw node id.
    :type use_labels: bool
    :param label_attr: Node attribute name to use as display label when
                       ``use_labels=True``.
    :type label_attr: str
    :returns: None (prints to stdout).
    :rtype: None
    """

    def _disp_name(n: Any) -> Any:
        if not use_labels:
            return n
        data = G.nodes[n]
        return data.get(label_attr, n)

    if include_nodes:
        print("Nodes:")
        print("-" * 40)
        node_items = sorted(G.nodes(data=True), key=lambda x: str(_disp_name(x[0])))
        for i, (n, attrs) in enumerate(node_items):
            if max_rows is not None and i >= max_rows:
                print(f"... ({len(G.nodes()) - max_rows} more)")
                break
            print(f"{_disp_name(n)}: {attrs}")
        print()

    if include_edges:
        print("Edges:")
        print("-" * 40)
        edge_items = sorted(
            G.edges(data=True),
            key=lambda x: (str(_disp_name(x[0])), str(_disp_name(x[1]))),
        )
        for i, (u, v, attrs) in enumerate(edge_items):
            if max_rows is not None and i >= max_rows:
                print(f"... ({len(G.edges()) - max_rows} more)")
                break
            print(f"{_disp_name(u)} >> {_disp_name(v)}: {attrs}")
        print()
