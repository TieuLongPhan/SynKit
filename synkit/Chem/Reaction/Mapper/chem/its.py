"""
Symmetry-distinct deduplication of atom-to-atom mappings via canonical ITS hashing.

The SLAP search can return several minimum-cost mappings that describe the *same*
reaction once symmetry-equivalent atoms are relabeled (for example, two mappings
that differ only by swapping two equivalent hydrogens or two equivalent ring
positions). Such mappings are isomorphic copies of a single optimum and should be
collapsed to one representative.

The canonical way to test "same reaction up to atom relabeling" is to build the
imaginary transition state (ITS) graph -- the union of the reactant and product
graphs with each atom identified through the mapping and each edge labeled by its
(reactant order, product order) pair -- and compare ITS graphs up to isomorphism.
Two mappings are isomorphic copies if and only if their ITS graphs are isomorphic
as node- and edge-labeled graphs. We obtain a canonical hash of each ITS graph and
deduplicate by that hash.

This module uses SynKit ITS construction and its Weisfeiler-Lehman graph hash
(:class:`synkit.Graph.Feature.wl_hash.WLHash`, a thin wrapper over
``networkx.weisfeiler_lehman_graph_hash``) for canonical hashing. The WL hash is
isomorphism-invariant by construction -- it depends only on iterative aggregation
of node/edge attributes over graph topology, not on input atom ordering -- so two
mappings that are isomorphic copies always receive the same hash.

Note that mappings which are genuinely *different* reaction centres but happen to
tie on chemical distance (e.g. an ambiguity over which of two equivalent leaving
groups departs) have *distinct* ITS graphs and are therefore preserved -- they are
separate optima, not isomorphic duplicates.
"""

from synkit.Graph.Feature.wl_hash import WLHash
from synkit.IO import rsmi_to_its

_hasher = None

# Node attributes seeding the WL colour refinement. ``atom_map`` is deliberately
# excluded so that two mappings differing only by a relabeling of equivalent
# atoms collapse to the same hash. The bond change is carried by the edge
# ``order`` attribute, an ``(reactant_order, product_order)`` tuple, which is fed
# into the refinement so that distinct reaction centres receive distinct hashes.
_WL_NODE_ATTRS = ["element", "aromatic", "charge", "hcount", "lone_pairs"]
_WL_EDGE_ATTRS = ["order"]
_WL_ITERATIONS = 5


def _get_hasher():
    global _hasher
    if _hasher is None:
        _hasher = WLHash(
            node=_WL_NODE_ATTRS,
            edge=_WL_EDGE_ATTRS,
            iterations=_WL_ITERATIONS,
        )
    return _hasher


def its_canonical_hash(mapped_rxn_smiles):
    """
    Return a canonical hash of the ITS graph of a mapped reaction SMILES.

    Parameters
    ----------
    mapped_rxn_smiles : str
        Reaction SMILES annotated with atom map numbers (``"A>>B"``).

    Returns
    -------
    str or None
        A hash that is identical for mappings describing the same reaction up to
        relabeling of symmetry-equivalent atoms, or ``None`` if the ITS graph
        could not be constructed.
    """
    try:
        its = rsmi_to_its(mapped_rxn_smiles)
        return _get_hasher().weisfeiler_lehman_graph_hash(its)
    except Exception:
        return None


def _standard_order(edge_attrs):
    if "standard_order" in edge_attrs:
        return float(edge_attrs["standard_order"])
    order = edge_attrs.get("order")
    if isinstance(order, (tuple, list)) and len(order) == 2:
        return float(order[0]) - float(order[1])
    return 0.0


def electron_balance_imbalances(its):
    """
    Return per-node signed bond-order imbalance for an ITS graph.

    The ITS edge ``standard_order`` is reactant bond order minus product bond
    order. Summing it over incident changed edges gives a compact local
    bookkeeping signal: positive values indicate net bond order lost at that
    atom, negative values indicate net bond order gained.
    """
    imbalances = {node: 0.0 for node in its.nodes}
    for u, v, attrs in its.edges(data=True):
        delta = _standard_order(attrs)
        imbalances[u] += delta
        imbalances[v] += delta
    return imbalances


def reaction_center_atom_maps(mapped_rxn_smiles, tol=1e-9):
    """
    Return atom-map numbers incident to changed ITS edges.

    The result is suitable for lightweight hydrogen expansion: map heavy atoms
    first, then display hydrogens only on atoms whose bonds changed.
    """
    try:
        its = rsmi_to_its(mapped_rxn_smiles)
    except Exception:
        return set()

    maps = set()
    for u, v, attrs in its.edges(data=True):
        if abs(_standard_order(attrs)) <= tol:
            continue
        for node in (u, v):
            data = its.nodes[node]
            atom_map = data.get("atom_map")
            if atom_map and data.get("element") != "H":
                maps.add(int(atom_map))
    return maps


def is_electron_balanced(its, tol=1e-9):  # noqa: C901
    """
    Return whether an ITS passes the bond-order electron-balance heuristic.

    This check deliberately uses explicit-H ITS graphs. Hydrogens are therefore
    present when synkit can expose them, and SynKit 1.4+ node attributes such as
    ``lone_pairs`` are available for hashing/bookkeeping. The rule is still a
    conservative bond-order heuristic rather than a full arrow-pushing proof:

    - each connected reaction-center component must have zero net changed bond
      order;
    - no atom may carry more than two units of local bond-order imbalance, and
      magnitude-2 centers must appear in opposite-sign pairs.

    The second condition allows normal terminal heteroatom source/sink behavior
    in substitutions. Magnitude-2 centers are allowed only when the same
    reaction-center component contains a matching opposite magnitude-2 center,
    which covers imine/condensation-style lone-pair bookkeeping while rejecting
    mappings that delete a C=O double bond and distribute the compensation over
    unrelated single-bond changes.
    """
    changed_edges = []
    adjacency = {}
    for u, v, attrs in its.edges(data=True):
        delta = _standard_order(attrs)
        if abs(delta) <= tol:
            continue
        changed_edges.append((u, v, delta))
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    if not changed_edges:
        return True

    imbalances = electron_balance_imbalances(its)
    if any(abs(value) > 2.0 + tol for value in imbalances.values()):
        return False

    edge_by_component = {}
    for u, v, delta in changed_edges:
        edge_by_component.setdefault(u, []).append((u, v, delta))
        edge_by_component.setdefault(v, []).append((u, v, delta))

    seen = set()
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        seen.update(component)

        total = 0.0
        used_edges = set()
        for node in component:
            for edge in edge_by_component.get(node, ()):
                key = frozenset(edge[:2])
                if key in used_edges:
                    continue
                used_edges.add(key)
                total += edge[2]
        if abs(total) > tol:
            return False

        big_pos = 0
        big_neg = 0
        for node in component:
            value = imbalances[node]
            if value > 1.0 + tol:
                big_pos += 1
            elif value < -1.0 - tol:
                big_neg += 1
        if big_pos != big_neg:
            return False

    return True


def _node_missing_on_one_side(data):
    present = data.get("present")
    return (
        isinstance(present, (tuple, list))
        and len(present) == 2
        and present[0] != present[1]
    )


def electron_balance_status(its, tol=1e-9):
    """
    Return ``True``/``False``/``None`` for electron-balance status.

    ``None`` means the mapped reaction omits a coproduct/byproduct in a changed
    component, so the displayed ITS is incomplete and a strict electron-balance
    verdict would be overconfident.
    """
    adjacency = {}
    edge_delta = {}
    for u, v, attrs in its.edges(data=True):
        delta = _standard_order(attrs)
        if abs(delta) <= tol:
            continue
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
        edge_delta[frozenset((u, v))] = delta

    seen = set()
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        seen.update(component)

        total = 0.0
        used_edges = set()
        for node in component:
            for nbr in adjacency.get(node, ()):
                key = frozenset((node, nbr))
                if key in used_edges:
                    continue
                used_edges.add(key)
                total += edge_delta[key]

        if abs(total) > tol and any(
            _node_missing_on_one_side(its.nodes[node]) for node in component
        ):
            return None

    return is_electron_balanced(its, tol=tol)


def mapped_rxn_is_electron_balanced(mapped_rxn_smiles):
    """
    Return electron-balance status for a mapped reaction SMILES.

    Returns ``None`` if synkit cannot construct the explicit-H ITS graph, so
    callers can avoid treating parser failures as proven invalid chemistry.
    """
    try:
        its = rsmi_to_its(mapped_rxn_smiles, explicit_hydrogen=True)
    except Exception:
        return None
    return electron_balance_status(its)


def dedup_mapped_rxns(results, smiles_key="smiles"):
    """
    Remove isomorphic duplicate mappings from a list of result dictionaries.

    Each result is expected to carry a ``"smiles"`` key with the mapped reaction
    SMILES. The canonical ITS hash is computed for every result and stored under
    ``"its_hash"``; only the first result seen for each distinct hash is kept, so
    the original ordering of symmetry-distinct optima is preserved.

    Results whose ITS graph cannot be built (hash ``None``) are never collapsed
    together -- they are all kept, keyed by their raw mapped SMILES instead -- so
    deduplication can only ever remove provable isomorphic copies.

    Parameters
    ----------
    results : list[dict]
        Mapping results, each containing a ``"smiles"`` entry.

    Returns
    -------
    list[dict]
        The symmetry-distinct subset of ``results``.
    """
    seen = set()
    deduped = []
    for r in results:
        h = its_canonical_hash(r.get(smiles_key))
        r["its_hash"] = h
        key = h if h is not None else ("raw", r.get(smiles_key))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped
