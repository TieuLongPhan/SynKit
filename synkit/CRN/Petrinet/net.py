# synkit/CRN/petri/net.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple

Place = str
TransitionId = str
Marking = Mapping[Place, int]
Multiset = Mapping[str, int]


@dataclass
class Transition:
    """
    Internal Petri-net transition representation.

    :param tid: Transition identifier (usually reaction / edge id).
    :type tid: str
    :param pre: Input arc weights: place -> tokens consumed.
    :type pre: Dict[str, int]
    :param post: Output arc weights: place -> tokens produced.
    :type post: Dict[str, int]
    """

    tid: TransitionId
    pre: Dict[Place, int]
    post: Dict[Place, int]

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"Transition({self.tid}, pre={self.pre}, post={self.post})"


class PetriNet:
    """
    Minimal Petri net container with marking semantics.

    This class is intentionally small and is shared between structural
    diagnostics (siphons, traps, semiflows) and pathway realizability
    utilities.

    It supports:

    * adding places and transitions,
    * checking whether a transition is enabled in a marking,
    * firing transitions to obtain successor markings,
    * deterministic tuple encoding of markings (for BFS / hashing).

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.petri.net import PetriNet

        net = PetriNet()
        net.add_place("A")
        net.add_place("B")
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        m0 = {"A": 1, "B": 0}
        assert net.enabled(m0, "t1")
        m1 = net.fire(m0, "t1")
        assert m1["A"] == 0 and m1["B"] == 1
    """

    def __init__(self) -> None:
        self.places: Set[Place] = set()
        self.transitions: Dict[TransitionId, Transition] = {}
        # deterministic order for marking_to_tuple
        self._place_index: Dict[Place, int] = {}

    # ---- construction helpers ----

    def add_place(self, p: Place) -> None:
        """
        Add a place to the net (idempotent).

        :param p: Place identifier.
        :type p: str
        """
        if p not in self.places:
            self.places.add(p)
            self._place_index[p] = len(self._place_index)

    def add_transition(
        self,
        tid: TransitionId,
        pre: Dict[Place, int],
        post: Dict[Place, int],
    ) -> None:
        """
        Add or overwrite a transition.

        All places mentioned in ``pre`` or ``post`` are automatically
        added to the net.

        :param tid: Transition identifier.
        :type tid: str
        :param pre: Input arc weights (place -> tokens consumed).
        :type pre: Dict[str, int]
        :param post: Output arc weights (place -> tokens produced).
        :type post: Dict[str, int]
        """
        for p in set(pre) | set(post):
            self.add_place(p)
        self.transitions[tid] = Transition(tid, dict(pre), dict(post))

    # ---- semantics ----

    def enabled(self, marking: Marking, tid: TransitionId) -> bool:
        """
        Check if transition ``tid`` is enabled in the given marking.

        :param marking: Current marking (place -> tokens).
        :type marking: Mapping[str, int]
        :param tid: Transition identifier.
        :type tid: str
        :returns: ``True`` if enabled, ``False`` otherwise.
        :rtype: bool
        """
        t = self.transitions[tid]
        for p, w in t.pre.items():
            if marking.get(p, 0) < w:
                return False
        return True

    def fire(self, marking: Marking, tid: TransitionId) -> Dict[Place, int]:
        """
        Fire transition ``tid`` from the given marking.

        :param marking: Current marking (place -> tokens).
        :type marking: Mapping[str, int]
        :param tid: Transition identifier.
        :type tid: str
        :returns: Successor marking as a plain ``dict``.
        :rtype: Dict[str, int]
        """
        t = self.transitions[tid]
        m = dict(marking)
        for p, w in t.pre.items():
            m[p] = m.get(p, 0) - w
        for p, w in t.post.items():
            m[p] = m.get(p, 0) + w
        return m

    def marking_to_tuple(self, m: Marking) -> Tuple[int, ...]:
        """
        Encode a marking as a deterministic tuple of integers.

        The order of places is fixed by the internal ``_place_index``
        mapping and is stable once places are added.

        :param m: Marking to encode.
        :type m: Mapping[str, int]
        :returns: Tuple representation suitable for hashing / BFS.
        :rtype: tuple[int, ...]
        """
        size = len(self._place_index)
        arr = [0] * size
        for p, idx in self._place_index.items():
            arr[idx] = int(m.get(p, 0))
        return tuple(arr)
