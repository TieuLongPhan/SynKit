# CRN/Hypergraph/conversion.py
from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional, Iterable, Set
import networkx as nx
from .rxn import RXNSide
from .hypergraph import CRNHyperGraph


# ======================================================================
# Pair 1: Hypergraph  <->  Bipartite
# ======================================================================


def hypergraph_to_bipartite(
    H: CRNHyperGraph,
    *,
    species_prefix: str = "S:",
    reaction_prefix: str = "R:",
    bipartite_values: Tuple[int, int] = (0, 1),
    include_stoich: bool = True,
    include_role: bool = True,
    include_isolated_species: bool = True,
    integer_ids: bool = True,
    include_species_attr: bool = False,
    include_edge_id_attr: bool = False,
) -> nx.DiGraph:
    """
    Export a CRN hypergraph to a **bipartite** NetworkX DiGraph
    with arcs ``species → reaction → species``.

    :param H: Hypergraph to export.
    :param species_prefix: Prefix for species node ids when ``integer_ids=False``.
    :param reaction_prefix: Prefix for reaction node ids when ``integer_ids=False``.
    :param bipartite_values: Bipartite marker values ``(species_value, reaction_value)``.
    :param include_stoich: If ``True``, add integer edge attribute ``'stoich'``.
    :param include_role: If ``True``, add edge attribute ``'role'`` in {``reactant``, ``product``}.
    :param include_isolated_species: If ``True``, keep species with no incident edges.
    :param integer_ids: If ``True``, species ids are ``1..N`` and reactions ``N+1..N+M``.
    :param include_species_attr: If ``True``, duplicate label into node attr ``'species'``.
    :param include_edge_id_attr: If ``True``, store reaction edge id in node attr ``'edge_id'``.
    :returns: Bipartite DiGraph.

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
        if include_species_attr:
            attrs["species"] = s
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
        nid = next_id if integer_ids else f"{species_prefix}{s}"
        next_id = next_id + 1 if integer_ids else next_id
        species_map[s] = nid
        if not G.has_node(nid):
            G.add_node(nid, **make_sp_attrs(s))
        return nid

    def add_rxn_node(eid: str, rule: str) -> Any:
        nonlocal next_id
        nid = next_id if integer_ids else f"{reaction_prefix}{eid}"
        next_id = next_id + 1 if integer_ids else next_id
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
    integer_ids: bool = True,
    species_label_attr: str = "label",
    reaction_edge_id_attr: str = "edge_id",
    reaction_label_attr: str = "label",
    stoich_attr: str = "stoich",
    role_attr: str = "role",
    default_rule: str = "r",
) -> CRNHyperGraph:
    """
    Reconstruct a **CRNHyperGraph** from a bipartite species→reaction→species graph.

    The function is the logical inverse of :func:`hypergraph_to_bipartite` and
    supports graphs produced by it, while attempting a best-effort reconstruction
    for general bipartite-like graphs.

    :param G: Bipartite graph (``species`` and ``reaction`` nodes).
    :param species_prefix: Prefix to detect species ids when kind/prefix absent.
    :param reaction_prefix: Prefix to detect reaction ids when kind/prefix absent.
    :param integer_ids: If ``True``, reactions/species may be numeric; labels read from attrs.
    :param species_label_attr: Node attribute holding species label (default: ``label``).
    :param reaction_edge_id_attr: Node attribute holding original edge id (default: ``edge_id``).
    :param reaction_label_attr: Node attribute holding reaction rule/label (default: ``label``).
    :param stoich_attr: Edge attribute for stoichiometry (default: ``stoich``).
    :param role_attr: Edge attribute for role (unused in reconstruction; direction is enough).
    :param default_rule: Fallback rule label when none found.
    :returns: Reconstructed hypergraph.

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_bipartite, bipartite_to_hypergraph
    >>> H0 = CRNHyperGraph().parse_rxns(["A+B>>C", "C>>A"])
    >>> B = hypergraph_to_bipartite(H0, integer_ids=False, include_edge_id_attr=True)
    >>> H1 = bipartite_to_hypergraph(B, integer_ids=False)
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

    return H


# ======================================================================
# Pair 2: Hypergraph  <->  Species Graph (collapsed)
# ======================================================================


def hypergraph_to_species_graph(H: CRNHyperGraph) -> nx.DiGraph:
    """
    Collapse hyperedges to a **species→species** DiGraph.

    Aggregated edge attributes:
      - ``via``: ``set`` of contributing hyperedge ids
      - ``rules``: ``set`` of contributing rule labels
      - ``min_stoich``: minimum of reactant/product stoichiometry encountered

    :param H: Hypergraph to collapse.
    :returns: Directed species graph.

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_species_graph
    >>> H = CRNHyperGraph().parse_rxns(["A+B>>C", "A>>D"])
    >>> S = hypergraph_to_species_graph(H)
    >>> ("A" in S) and ("C" in S) and S.has_edge("A", "C")
    True
    """
    G = nx.DiGraph()
    for s in H.species:
        G.add_node(s, label=s, kind="species")
    for eid, e in H.edges.items():
        for r, rc in e.reactants.items():
            for p, pc in e.products.items():
                if G.has_edge(r, p):
                    data = G[r][p]
                    data["via"].add(eid)
                    data["rules"].add(e.rule)
                    data["min_stoich"] = min(data["min_stoich"], min(rc, pc))
                else:
                    G.add_edge(r, p, via={eid}, rules={e.rule}, min_stoich=min(rc, pc))
    return G


def species_graph_to_hypergraph(
    G: nx.DiGraph,
    *,
    default_rule: str = "r",
    use_min_stoich: bool = True,
) -> CRNHyperGraph:
    """
    Reconstruct a **CRNHyperGraph** from a collapsed species→species graph.

    If edges expose ``'via'`` sets (carrying original hyperedge ids), arcs that
    share the same ``via`` id are grouped back into one hyperedge. If no ``via``
    is present, each species arc becomes its own hyperedge.

    :param G: Species→species DiGraph (typically from :func:`hypergraph_to_species_graph`).
    :param default_rule: Fallback rule for reconstructed edges.
    :param use_min_stoich: If ``True``, pick edge stoichiometry from ``min_stoich`` when present.
    :returns: Best-effort reconstructed hypergraph.

    **Examples**
    ----------
    >>> from synkit.CRN.Hypergraph import CRNHyperGraph
    >>> from synkit.CRN.Hypergraph.conversion import hypergraph_to_species_graph, species_graph_to_hypergraph
    >>> H0 = CRNHyperGraph().parse_rxns(["A+B>>C"])
    >>> S = hypergraph_to_species_graph(H0)
    >>> H1 = species_graph_to_hypergraph(S)
    >>> sorted(H1.species) == sorted(H0.species)
    True
    """
    H = CRNHyperGraph()

    # first pass: if 'via' present, group by eid
    eid_map: Dict[str, Dict[str, Any]] = {}
    for u, v, attrs in G.edges(data=True):
        via = attrs.get("via")
        min_st = attrs.get("min_stoich", None)
        rules = attrs.get("rules", None)
        if via:
            for eid in list(via):
                entry = eid_map.setdefault(
                    eid,
                    {
                        "reactants": set(),
                        "products": set(),
                        "min_stoich": None,
                        "rules": set(),
                    },
                )
                entry["reactants"].add(u)
                entry["products"].add(v)
                if min_st is not None:
                    entry["min_stoich"] = (
                        int(min_st)
                        if entry["min_stoich"] is None
                        else min(entry["min_stoich"], int(min_st))
                    )
                if rules:
                    if isinstance(rules, set):
                        entry["rules"].update(rules)
                    else:
                        entry["rules"].add(rules)
        else:
            # no 'via': one edge per arc
            eid = f"edge_{abs(hash((u, v))) % (10**8)}"
            entry = eid_map.setdefault(
                eid,
                {
                    "reactants": set(),
                    "products": set(),
                    "min_stoich": None,
                    "rules": set(),
                },
            )
            entry["reactants"].add(u)
            entry["products"].add(v)
            if attrs.get("min_stoich") is not None:
                ms = int(attrs["min_stoich"])
                entry["min_stoich"] = (
                    ms if entry["min_stoich"] is None else min(entry["min_stoich"], ms)
                )
            if attrs.get("rules"):
                rules_attr = attrs["rules"]
                if isinstance(rules_attr, set):
                    entry["rules"].update(rules_attr)
                else:
                    entry["rules"].add(rules_attr)

    # materialize hyperedges
    for eid, data in eid_map.items():
        coeff = (
            data["min_stoich"]
            if (use_min_stoich and data["min_stoich"] is not None)
            else 1
        )
        reactants = {s: coeff for s in sorted(data["reactants"])}
        products = {s: coeff for s in sorted(data["products"])}
        rule = next(iter(data["rules"])) if data["rules"] else default_rule
        H.add_rxn(reactants, products, rule=rule, edge_id=str(eid))

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
                f"{s.ljust(longest)}         [{ic:2d}] {ins_str:<10}   [{oc:2d}] {outs_str}"
            )
        else:
            print(f"{s.ljust(longest)}          {ins_str:<10}        {outs_str}")


def print_edge_list(
    H: CRNHyperGraph,
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
