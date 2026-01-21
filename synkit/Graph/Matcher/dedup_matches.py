from __future__ import annotations

from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple


Sig = Tuple[Tuple, Tuple]


def _build_host_orbit_index(
    host_orbits: Iterable[FrozenSet[int]],
) -> Dict[int, int]:
    """
    Build host node -> host orbit index.

    :param host_orbits:
        Iterable of host orbits.
    :returns:
        Dict mapping host node to orbit index.
    """
    host_orbit_index: Dict[int, int] = {}
    for idx, orb in enumerate(host_orbits):
        for h in orb:
            host_orbit_index[h] = idx
    return host_orbit_index


def _make_host_repr(
    host_orbits: Optional[Iterable[FrozenSet[int]]],
) -> Callable[[int], int]:
    """
    Create a function that maps host node to its representative.

    If host orbits are provided, the representative is the host-orbit index.

    :param host_orbits:
        Host orbits or None.
    :returns:
        Callable mapping host node -> representative integer.
    :raises ValueError:
        If host orbits are provided but a host node is not covered.
    """
    if host_orbits is None:
        return lambda h: h

    host_orbit_index = _build_host_orbit_index(host_orbits)

    def _repr(h: int) -> int:
        try:
            return host_orbit_index[h]
        except KeyError as exc:
            raise ValueError(
                f"Host node {h} not present in host_orbits; cannot canonicalize."
            ) from exc

    return _repr


def _prepare_pattern_orbits(
    pattern_orbits: Optional[Iterable[FrozenSet[int]]],
    pattern_anchor: FrozenSet[int],
) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
    """
    Prepare free pattern orbits and anchored nodes.

    :param pattern_orbits:
        Pattern orbits or None.
    :param pattern_anchor:
        Anchor nodes of the pattern.
    :returns:
        (free_pattern_orbits, anchored_pattern_nodes)
    """
    if pattern_orbits is None:
        return ([], ())

    orbits_sorted = [tuple(sorted(o)) for o in pattern_orbits]
    free_orbits = [o for o in orbits_sorted if not set(o) & pattern_anchor]
    anchored_nodes = tuple(sorted(pattern_anchor))
    return (free_orbits, anchored_nodes)


def _free_sig_from_pattern_orbits(
    mapping: Dict[int, int],
    free_pattern_orbits: List[Tuple[int, ...]],
    host_repr: Callable[[int], int],
) -> Tuple:
    """
    Compute free signature using pattern orbits (partial-match safe).

    Each orbit contributes a pair:
        (present_pattern_nodes_sorted, sorted(host_repr(images)))

    :param mapping:
        Pattern -> host mapping.
    :param free_pattern_orbits:
        Pattern orbits disjoint from anchor.
    :param host_repr:
        Host representative function.
    :returns:
        Free signature tuple.
    """
    if not free_pattern_orbits:
        return ()

    m_keys = set(mapping.keys())
    parts: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    for orbit in free_pattern_orbits:
        present = [p for p in orbit if p in m_keys]
        if not present:
            continue

        present_sorted = tuple(sorted(present))
        image = tuple(sorted(host_repr(mapping[p]) for p in present_sorted))
        parts.append((present_sorted, image))

    return tuple(parts)


def _free_sig_host_only(
    mapping: Dict[int, int],
    host_repr: Callable[[int], int],
) -> Tuple:
    """
    Compute free signature using host-only symmetry (mapping values only).

    :param mapping:
        Pattern -> host mapping.
    :param host_repr:
        Host representative function.
    :returns:
        Free signature tuple.
    """
    return (tuple(sorted(host_repr(h) for h in mapping.values())),)


def _anchor_sig(
    mapping: Dict[int, int],
    anchored_pattern_nodes: Tuple[int, ...],
) -> Tuple:
    """
    Compute anchor signature: exact placement for anchored pattern nodes
    that are present in the mapping.

    Returned as (pattern_node, host_node) pairs for stability.

    :param mapping:
        Pattern -> host mapping.
    :param anchored_pattern_nodes:
        Sorted anchored pattern nodes.
    :returns:
        Anchor signature tuple.
    """
    if not anchored_pattern_nodes:
        return ()

    present = [p for p in anchored_pattern_nodes if p in mapping]
    return tuple((p, mapping[p]) for p in present)


def deduplicate_matches_with_anchor(
    matches: Iterable[Dict[int, int]],
    *,
    pattern_orbits: Optional[Iterable[FrozenSet[int]]] = None,
    pattern_anchor: Optional[FrozenSet[int]] = None,
    host_orbits: Optional[Iterable[FrozenSet[int]]] = None,
    host_anchor: Optional[FrozenSet[int]] = None,
) -> List[Dict[int, int]]:
    """
    Deduplicate pattern→host matches with optional anchor-aware symmetry
    breaking on both pattern and host sides.

    This function supports *partial* mappings: a match may map only a subset
    of pattern nodes. Orbit-based signatures are computed using only orbit
    nodes present in each mapping.

    Rules
    -----
    - Matches are always interpreted as **pattern → host** mappings.
    - If ``pattern_orbits`` is provided:
        * Pattern nodes inside ``pattern_anchor`` are fixed when present.
        * Pattern orbits disjoint from the anchor are deduplicated up to
          permutation, with host-side symmetry optionally collapsed by
          ``host_orbits``.
    - If ``pattern_orbits`` is None and ``host_orbits`` is provided:
        * Deduplicate by the multiset of host orbits hit (mapping values only).
    - If **both orbit arguments are None**, return matches unchanged.

    :param matches:
        Iterable of pattern → host mapping dictionaries.
    :param pattern_orbits:
        Automorphism orbits of the pattern graph (optional).
    :param pattern_anchor:
        Anchor component of the pattern graph (optional).
    :param host_orbits:
        Automorphism orbits of the host graph (optional).
    :param host_anchor:
        Anchor component of the host graph (optional). Kept for API symmetry;
        host anchoring is handled indirectly by orbit collapsing.
    :returns:
        Deduplicated list of pattern → host mappings, preserving input order.
    :raises ValueError:
        If ``host_orbits`` is provided but a mapping contains a host node not
        covered by any host orbit.
    """
    # silence "unused" while keeping the API you asked for
    _ = host_anchor

    if pattern_orbits is None and host_orbits is None:
        return list(matches)

    pattern_anchor = pattern_anchor or frozenset()
    host_repr = _make_host_repr(host_orbits)

    free_pattern_orbits, anchored_pattern_nodes = _prepare_pattern_orbits(
        pattern_orbits,
        pattern_anchor,
    )

    use_pattern = bool(free_pattern_orbits) or bool(anchored_pattern_nodes)
    seen: set[Sig] = set()
    unique: List[Dict[int, int]] = []

    for m in matches:
        if use_pattern:
            free_sig = _free_sig_from_pattern_orbits(
                m,
                free_pattern_orbits,
                host_repr,
            )
        else:
            # host-only symmetry case
            free_sig = _free_sig_host_only(m, host_repr)

        anchor_part = _anchor_sig(m, anchored_pattern_nodes)
        sig: Sig = (free_sig, anchor_part)

        if sig in seen:
            continue

        seen.add(sig)
        unique.append(m)

    return unique
