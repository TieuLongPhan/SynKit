"""Branch-and-bound enumeration of minimal siphons and traps.

Enumerating siphons by testing all :math:`2^n` place subsets is only viable for
toy networks: a 25-species network already takes about a minute, and the cost
multiplies by roughly eight for every three further species.

This module instead builds on the *closure* operator: for any set of places
``A`` there is a unique **maximal siphon contained in** ``A``, and it can be
computed in polynomial time by repeatedly discarding places that break the
siphon condition (:func:`max_siphon_within`). Minimal siphons are then reached
by starting from the maximal siphon of the whole net and repeatedly removing
one place and re-closing.

That descent is complete. If ``M`` is a minimal siphon and ``S`` is any siphon
with ``M`` a strict subset of ``S``, then for a place ``p`` in ``S`` but not in
``M`` the set ``S \\ {p}`` still contains ``M``; since ``M`` is itself a siphon,
it survives the closure, so ``max_siphon_within(S \\ {p})`` still contains
``M``. Every minimal siphon is therefore reachable from the top by successive
delete-and-close steps.

Traps are the exact dual (every transition consuming from the set also produces
back into it) and are handled by the same machinery.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Structure import SynCRN
    from synkit.CRN.Petrinet import PetriNet
    from synkit.CRN.Petrinet.siphon_search import minimal_siphons

    net = PetriNet.from_syncrn(SynCRN.from_reaction_strings(["A>>B", "B>>A"]))
    print(minimal_siphons(net))
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .net import PetriNet

__all__ = [
    "max_siphon_within",
    "max_trap_within",
    "minimal_siphons",
    "minimal_traps",
    "SiphonSearchLimit",
]

#: Default ceiling on explored search nodes, so a pathological network degrades
#: into a truncated result rather than an apparent hang.
DEFAULT_MAX_NODES = 200_000


class SiphonSearchLimit(RuntimeError):
    """Raised when a siphon or trap search exceeds its exploration budget."""


def _incidence(
    net: PetriNet,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Index a Petri net as ``transition -> consumed`` / ``transition -> produced``.

    :param net:
        Petri net to index.
    :type net: PetriNet

    :return:
        Pair ``(pre, post)`` mapping each transition id to the set of places it
        consumes from and produces into, with zero-weight arcs dropped.
    :rtype: Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
    """
    pre: Dict[str, Set[str]] = {}
    post: Dict[str, Set[str]] = {}

    for tid in net.transition_order:
        t = net.transitions[tid]
        pre[tid] = {p for p, w in t.pre.items() if w > 0}
        post[tid] = {p for p, w in t.post.items() if w > 0}

    return pre, post


def _max_closed_within(
    allowed: Set[str],
    *,
    trigger: Dict[str, Set[str]],
    required: Dict[str, Set[str]],
    transition_order: Sequence[str],
) -> Set[str]:
    """Compute the maximal subset of ``allowed`` closed under a incidence rule.

    A set ``S`` is closed when, for every transition ``t``, ``trigger[t]``
    meeting ``S`` implies ``required[t]`` also meets ``S``. Choosing
    ``trigger=post`` and ``required=pre`` yields siphons; swapping them yields
    traps.

    Places are removed until a fixed point is reached, which is the unique
    maximal closed subset.

    :param allowed:
        Candidate place set. Not mutated.
    :type allowed: Set[str]

    :param trigger:
        Transition-to-places map that activates the condition.
    :type trigger: Dict[str, Set[str]]

    :param required:
        Transition-to-places map that must then also be met.
    :type required: Dict[str, Set[str]]

    :param transition_order:
        Deterministic transition iteration order.
    :type transition_order: Sequence[str]

    :return:
        Maximal closed subset of ``allowed``, possibly empty.
    :rtype: Set[str]
    """
    current = set(allowed)

    changed = True
    while changed and current:
        changed = False
        for tid in transition_order:
            hit = trigger[tid] & current
            if not hit:
                continue
            if required[tid] & current:
                continue
            # This transition breaks the condition: every place it triggers on
            # must leave the set.
            current -= hit
            changed = True
            if not current:
                break

    return current


def max_siphon_within(net: PetriNet, allowed: Set[str]) -> Set[str]:
    """Return the unique maximal siphon contained in ``allowed``.

    :param net:
        Petri net to analyse.
    :type net: PetriNet

    :param allowed:
        Places the siphon may draw from.
    :type allowed: Set[str]

    :return:
        Maximal siphon inside ``allowed``, empty if there is none.
    :rtype: Set[str]

    .. rubric:: Example

    .. code-block:: python

        max_siphon_within(net, set(net.place_order))
    """
    pre, post = _incidence(net)
    return _max_closed_within(
        allowed,
        trigger=post,
        required=pre,
        transition_order=net.transition_order,
    )


def max_trap_within(net: PetriNet, allowed: Set[str]) -> Set[str]:
    """Return the unique maximal trap contained in ``allowed``.

    :param net:
        Petri net to analyse.
    :type net: PetriNet

    :param allowed:
        Places the trap may draw from.
    :type allowed: Set[str]

    :return:
        Maximal trap inside ``allowed``, empty if there is none.
    :rtype: Set[str]

    .. rubric:: Example

    .. code-block:: python

        max_trap_within(net, set(net.place_order))
    """
    pre, post = _incidence(net)
    return _max_closed_within(
        allowed,
        trigger=pre,
        required=post,
        transition_order=net.transition_order,
    )


def _enumerate_minimal(
    net: PetriNet,
    *,
    trigger: Dict[str, Set[str]],
    required: Dict[str, Set[str]],
    max_size: Optional[int],
    max_nodes: int,
    strict_limit: bool,
) -> List[Set[str]]:
    """Enumerate inclusion-minimal closed sets by delete-and-close descent.

    :param net:
        Petri net to analyse.
    :type net: PetriNet

    :param trigger:
        Transition-to-places map activating the closure condition.
    :type trigger: Dict[str, Set[str]]

    :param required:
        Transition-to-places map that must also be met.
    :type required: Dict[str, Set[str]]

    :param max_size:
        Optional ceiling on the size of returned sets.
    :type max_size: Optional[int]

    :param max_nodes:
        Maximum number of closure evaluations before giving up.
    :type max_nodes: int

    :param strict_limit:
        Whether exceeding ``max_nodes`` raises instead of truncating.
    :type strict_limit: bool

    :return:
        Inclusion-minimal closed sets.
    :rtype: List[Set[str]]

    :raises SiphonSearchLimit:
        If the budget is exhausted and ``strict_limit`` is set.
    """
    places = list(net.place_order)
    order = net.transition_order

    root = _max_closed_within(
        set(places), trigger=trigger, required=required, transition_order=order
    )
    if not root:
        return []

    seen: Set[FrozenSet[str]] = set()
    minimal: List[Set[str]] = []
    stack: List[FrozenSet[str]] = [frozenset(root)]
    seen.add(frozenset(root))
    nodes = 0
    truncated = False

    while stack:
        current = stack.pop()

        is_minimal = True
        for p in sorted(current):
            if nodes >= max_nodes:
                truncated = True
                break
            nodes += 1

            child = _max_closed_within(
                set(current) - {p},
                trigger=trigger,
                required=required,
                transition_order=order,
            )
            if not child:
                continue

            # A non-empty proper closed subset exists, so `current` is not
            # inclusion-minimal.
            is_minimal = False
            key = frozenset(child)
            if key not in seen:
                seen.add(key)
                stack.append(key)

        if truncated:
            break
        if is_minimal:
            minimal.append(set(current))

    if truncated and strict_limit:
        raise SiphonSearchLimit(
            f"Search exceeded {max_nodes} closure evaluations. Raise max_nodes, "
            f"set max_size, or pass strict_limit=False to accept a partial result."
        )

    # The descent can surface a set that strictly contains another minimal one
    # when the budget truncates a branch; filter defensively.
    result = [
        s
        for s in minimal
        if not any(t < s for t in minimal)
        and (max_size is None or len(s) <= max_size)
    ]
    result.sort(key=lambda s: (len(s), sorted(s)))
    return result


def minimal_siphons(
    net: PetriNet,
    *,
    max_size: Optional[int] = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    strict_limit: bool = False,
) -> List[Set[str]]:
    """Enumerate the inclusion-minimal siphons of a Petri net.

    :param net:
        Petri net to analyse.
    :type net: PetriNet

    :param max_size:
        Optional ceiling on siphon size. Filtering happens after enumeration,
        so the result is exactly the minimal siphons no larger than the bound.
    :type max_size: Optional[int]

    :param max_nodes:
        Maximum number of closure evaluations before the search stops.
    :type max_nodes: int

    :param strict_limit:
        Raise :class:`SiphonSearchLimit` instead of returning a partial result
        when the budget is exhausted.
    :type strict_limit: bool

    :return:
        Minimal siphons, ordered by size then lexicographically.
    :rtype: List[Set[str]]

    .. rubric:: Example

    .. code-block:: python

        minimal_siphons(net, max_size=4)
    """
    pre, post = _incidence(net)
    return _enumerate_minimal(
        net,
        trigger=post,
        required=pre,
        max_size=max_size,
        max_nodes=max_nodes,
        strict_limit=strict_limit,
    )


def minimal_traps(
    net: PetriNet,
    *,
    max_size: Optional[int] = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    strict_limit: bool = False,
) -> List[Set[str]]:
    """Enumerate the inclusion-minimal traps of a Petri net.

    :param net:
        Petri net to analyse.
    :type net: PetriNet

    :param max_size:
        Optional ceiling on trap size.
    :type max_size: Optional[int]

    :param max_nodes:
        Maximum number of closure evaluations before the search stops.
    :type max_nodes: int

    :param strict_limit:
        Raise :class:`SiphonSearchLimit` instead of returning a partial result
        when the budget is exhausted.
    :type strict_limit: bool

    :return:
        Minimal traps, ordered by size then lexicographically.
    :rtype: List[Set[str]]

    .. rubric:: Example

    .. code-block:: python

        minimal_traps(net)
    """
    pre, post = _incidence(net)
    return _enumerate_minimal(
        net,
        trigger=pre,
        required=post,
        max_size=max_size,
        max_nodes=max_nodes,
        strict_limit=strict_limit,
    )
