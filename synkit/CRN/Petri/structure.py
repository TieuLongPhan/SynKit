# synkit/CRN/petri/structure.py
from __future__ import annotations

from itertools import combinations
from typing import Any, List, Set

import networkx as nx

from ..Props.utils import _split_species_reactions, _species_order
from ..Hypergraph.conversion import _as_bipartite


def _is_siphon_indices(
    G: nx.Graph,
    species_nodes_sorted: List[Any],
    reaction_nodes: List[Any],
    S_idx: Set[int],
) -> bool:
    """
    Check the **siphon** condition on a set of species indices.

    A set :math:`S` of species is a siphon if, whenever a reaction
    produces some species in :math:`S`, that reaction also consumes at
    least one species in :math:`S` (Murata, 1989).
    """
    if not S_idx:
        return False

    S_nodes = {species_nodes_sorted[i] for i in S_idx}

    for r in reaction_nodes:
        # does reaction produce any species in S?
        produces = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "product"
                and data.get("stoich", 0) > 0
            ):
                produces = True
                break
        if not produces:
            continue

        # then it must consume at least one species in S
        consumes = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "reactant"
                and data.get("stoich", 0) > 0
            ):
                consumes = True
                break
        if not consumes:
            return False
    return True


def _is_trap_indices(
    G: nx.Graph,
    species_nodes_sorted: List[Any],
    reaction_nodes: List[Any],
    S_idx: Set[int],
) -> bool:
    """
    Check the **trap** condition on a set of species indices.

    A set :math:`S` of species is a trap if, whenever a reaction
    consumes some species in :math:`S`, that reaction also produces
    at least one species in :math:`S` (Murata, 1989).
    """
    if not S_idx:
        return False

    S_nodes = {species_nodes_sorted[i] for i in S_idx}

    for r in reaction_nodes:
        # does reaction consume any species in S?
        consumes = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "reactant"
                and data.get("stoich", 0) > 0
            ):
                consumes = True
                break
        if not consumes:
            continue

        # then it must produce at least one species in S
        produces = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "product"
                and data.get("stoich", 0) > 0
            ):
                produces = True
                break
        if not produces:
            return False
    return True


def _minimal_sets(candidates: List[Set[int]]) -> List[Set[int]]:
    """
    Return inclusion-minimal sets from a list of integer subsets.

    A set :math:`S` is kept if it is not a strict superset of any other
    candidate already in the output.

    :reference: Standard minimality filtering in Petri net analysis.
    """
    out: List[Set[int]] = []
    for S in candidates:
        if any(T.issubset(S) for T in out):
            continue
        # remove supersets of S already in out
        out = [T for T in out if not S.issubset(T)]
        out.append(S)
    return out


def find_siphons(crn: Any, *, max_size: int | None = None) -> List[Set[str]]:
    """
    Enumerate **minimal siphons** via brute-force subset search.

    A siphon is a set of species with the property that any reaction that
    produces one of those species also consumes at least one of them
    (Murata, 1989). This routine enumerates all *inclusion-minimal*
    siphons up to ``max_size``.

    This is practical only for small networks (typically up to
    :math:`\\sim 10` species).

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param max_size: Optional maximum size of siphons to search for. If
        ``None``, all subset sizes are considered.
    :type max_size: int or None
    :returns: List of minimal siphons, each represented as a set of
        species labels (in the deterministic order induced by
        :func:`_species_order`).
    :rtype: list[set[str]]
    """
    G = _as_bipartite(crn)
    species_nodes_sorted, species_labels, _ = _species_order(G)
    _, reaction_nodes = _split_species_reactions(G)

    n_s = len(species_labels)
    if max_size is None:
        max_size = n_s

    all_indices = list(range(n_s))
    candidates: List[Set[int]] = []
    for k in range(1, max_size + 1):
        for combo in combinations(all_indices, k):
            S_idx = set(combo)
            if _is_siphon_indices(G, species_nodes_sorted, reaction_nodes, S_idx):
                candidates.append(S_idx)

    minimal = _minimal_sets(candidates)
    return [set(species_labels[i] for i in S_idx) for S_idx in minimal]


def find_traps(crn: Any, *, max_size: int | None = None) -> List[Set[str]]:
    """
    Enumerate **minimal traps** via brute-force subset search.

    A trap is a set of species with the property that any reaction that
    consumes one of those species also produces at least one of them
    (Murata, 1989). This routine enumerates all inclusion-minimal traps
    up to ``max_size``.

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param max_size: Optional maximum size of traps to search for. If
        ``None``, all subset sizes are considered.
    :type max_size: int or None
    :returns: List of minimal traps, each represented as a set of
        species labels.
    :rtype: list[set[str]]
    """
    G = _as_bipartite(crn)
    species_nodes_sorted, species_labels, _ = _species_order(G)
    _, reaction_nodes = _split_species_reactions(G)

    n_s = len(species_labels)
    if max_size is None:
        max_size = n_s

    all_indices = list(range(n_s))
    candidates: List[Set[int]] = []
    for k in range(1, max_size + 1):
        for combo in combinations(all_indices, k):
            S_idx = set(combo)
            if _is_trap_indices(G, species_nodes_sorted, reaction_nodes, S_idx):
                candidates.append(S_idx)

    minimal = _minimal_sets(candidates)
    return [set(species_labels[i] for i in S_idx) for S_idx in minimal]
