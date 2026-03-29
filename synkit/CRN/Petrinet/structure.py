from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Set

from .net import PetriNet


def _as_petri(crn: Any) -> PetriNet:
    """
    Convert a SynCRN-like object into a :class:`PetriNet`.

    If ``crn`` is already a :class:`PetriNet`, it is returned unchanged.
    Otherwise, the object is converted using :meth:`PetriNet.from_syncrn`.

    :param crn: SynCRN-like object or an existing :class:`PetriNet`.
    :type crn: Any
    :returns: Petri-net representation of the input object.
    :rtype: PetriNet
    """
    return crn if isinstance(crn, PetriNet) else PetriNet.from_syncrn(crn)


def _is_siphon(net: PetriNet, places: Set[str]) -> bool:
    """
    Test whether a set of places is a siphon.

    A place set ``S`` is a siphon if every transition that produces tokens into
    ``S`` also consumes at least one token from ``S``.

    :param net: Petri net to inspect.
    :type net: PetriNet
    :param places: Candidate place set.
    :type places: Set[str]
    :returns:
        ``True`` if ``places`` is a non-empty siphon, otherwise ``False``.
    :rtype: bool
    """
    if not places:
        return False
    for tid in net.transition_order:
        t = net.transitions[tid]
        produces_into_s = any(p in places and w > 0 for p, w in t.post.items())
        if not produces_into_s:
            continue
        consumes_from_s = any(p in places and w > 0 for p, w in t.pre.items())
        if not consumes_from_s:
            return False
    return True


def _is_trap(net: PetriNet, places: Set[str]) -> bool:
    """
    Test whether a set of places is a trap.

    A place set ``S`` is a trap if every transition that consumes tokens from
    ``S`` also produces at least one token back into ``S``.

    :param net: Petri net to inspect.
    :type net: PetriNet
    :param places: Candidate place set.
    :type places: Set[str]
    :returns:
        ``True`` if ``places`` is a non-empty trap, otherwise ``False``.
    :rtype: bool
    """
    if not places:
        return False
    for tid in net.transition_order:
        t = net.transitions[tid]
        consumes_from_s = any(p in places and w > 0 for p, w in t.pre.items())
        if not consumes_from_s:
            continue
        produces_into_s = any(p in places and w > 0 for p, w in t.post.items())
        if not produces_into_s:
            return False
    return True


def _minimal_sets(candidates: List[Set[str]]) -> List[Set[str]]:
    """
    Reduce a collection of candidate sets to inclusion-minimal sets.

    The result contains only those sets for which no strict subset is also
    present among the candidates.

    :param candidates: Candidate place sets.
    :type candidates: List[Set[str]]
    :returns: Inclusion-minimal subset of the input candidates.
    :rtype: List[Set[str]]
    """
    candidates = sorted(candidates, key=lambda s: (len(s), sorted(s)))
    out: List[Set[str]] = []
    for s in candidates:
        if any(t.issubset(s) for t in out):
            continue
        out = [t for t in out if not s.issubset(t)]
        out.append(s)
    return out


def _render_place_sets(
    net: PetriNet,
    sets_: List[Set[str]],
    *,
    names: str,
) -> List[Set[str]]:
    """
    Render place sets either by internal ids or by place labels.

    :param net: Petri net providing place-name lookup.
    :type net: PetriNet
    :param sets_: Place sets expressed in internal place ids.
    :type sets_: List[Set[str]]
    :param names:
        Output naming mode. Supported values are ``"id"`` and ``"label"``.
    :type names: str
    :returns:
        Place sets rendered according to the requested naming convention.
    :rtype: List[Set[str]]
    :raises ValueError: If ``names`` is not ``"id"`` or ``"label"``.
    """
    if names == "id":
        return [set(s) for s in sets_]
    if names == "label":
        return [{net.place_name(p) for p in s} for s in sets_]
    raise ValueError("names must be 'id' or 'label'")


def find_siphons(
    crn: Any,
    *,
    max_size: int | None = None,
    names: str = "label",
) -> List[Set[str]]:
    """
    Enumerate inclusion-minimal siphons of a SynCRN Petri-net view.

    The search is performed by brute-force subset enumeration up to the
    requested size bound, followed by inclusion-minimal filtering.

    :param crn: SynCRN-like object or :class:`PetriNet`.
    :type crn: Any
    :param max_size:
        Maximum siphon size to consider. ``None`` means all sizes.
    :type max_size: int | None
    :param names:
        Whether to return internal place ids or species labels.
        Supported values are ``"id"`` and ``"label"``.
    :type names: str
    :returns: Inclusion-minimal siphons.
    :rtype: List[Set[str]]
    """
    net = _as_petri(crn)
    places = net.place_order
    n = len(places)
    limit = n if max_size is None else min(max_size, n)

    candidates: List[Set[str]] = []
    for k in range(1, limit + 1):
        for combo in combinations(places, k):
            s = set(combo)
            if _is_siphon(net, s):
                candidates.append(s)
    return _render_place_sets(net, _minimal_sets(candidates), names=names)


def find_traps(
    crn: Any,
    *,
    max_size: int | None = None,
    names: str = "label",
) -> List[Set[str]]:
    """
    Enumerate inclusion-minimal traps of a SynCRN Petri-net view.

    The search is performed by brute-force subset enumeration up to the
    requested size bound, followed by inclusion-minimal filtering.

    :param crn: SynCRN-like object or :class:`PetriNet`.
    :type crn: Any
    :param max_size:
        Maximum trap size to consider. ``None`` means all sizes.
    :type max_size: int | None
    :param names:
        Whether to return internal place ids or species labels.
        Supported values are ``"id"`` and ``"label"``.
    :type names: str
    :returns: Inclusion-minimal traps.
    :rtype: List[Set[str]]
    """
    net = _as_petri(crn)
    places = net.place_order
    n = len(places)
    limit = n if max_size is None else min(max_size, n)

    candidates: List[Set[str]] = []
    for k in range(1, limit + 1):
        for combo in combinations(places, k):
            s = set(combo)
            if _is_trap(net, s):
                candidates.append(s)
    return _render_place_sets(net, _minimal_sets(candidates), names=names)


def species_transition_neighborhoods(crn: Any) -> Dict[str, Dict[str, List[str]]]:
    """
    Return per-species producer and consumer transition neighborhoods.

    This is a small structural helper that is often useful when debugging
    siphons, traps, and reachability issues.

    The returned dictionary is keyed by internal place id. For each place, the
    value contains the display label, the transitions that produce tokens into
    the place, and the transitions that consume tokens from it.

    :param crn: SynCRN-like object or :class:`PetriNet`.
    :type crn: Any
    :returns:
        Mapping from place id to a dictionary with keys ``"label"``,
        ``"producer_transitions"``, and ``"consumer_transitions"``.
    :rtype: Dict[str, Dict[str, List[str]]]
    """
    net = _as_petri(crn)
    out: Dict[str, Dict[str, List[str]]] = {}
    for p in net.place_order:
        producers: List[str] = []
        consumers: List[str] = []
        for tid in net.transition_order:
            t = net.transitions[tid]
            if p in t.post:
                producers.append(tid)
            if p in t.pre:
                consumers.append(tid)
        out[p] = {
            "label": net.place_name(p),
            "producer_transitions": producers,
            "consumer_transitions": consumers,
        }
    return out
