# synkit/CRN/explorer.py
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

from .utils import counter_key, multiset_contains
from .exceptions import SearchError  # unified error base

# Defaults (tunable)
DEFAULT_MIN_OVERLAP = 1
DEFAULT_MAX_DEPTH = 30


class Pathway:
    """
    Lightweight pathway container: reaction id sequence + visited states.

    :param reaction_ids: Ordered list of reaction ids along the path.
    :type reaction_ids: List[int] | None
    :param states: Ordered list of states (Counter) including the start state and
                   all post-step states (i.e., length = len(reaction_ids) + 1).
    :type states: List[Counter] | None
    """

    def __init__(
        self,
        reaction_ids: Optional[List[int]] = None,
        states: Optional[List[Counter]] = None,
    ) -> None:
        self.reaction_ids: List[int] = list(reaction_ids or [])
        self.states: List[Counter] = list(states or [])

    @property
    def steps(self) -> int:
        """
        Number of reaction steps in the pathway.

        :returns: Number of steps.
        :rtype: int
        """
        return len(self.reaction_ids)

    def summary(self) -> str:
        """
        Produce a short human-readable summary.

        :returns: Summary string.
        :rtype: str
        """
        return f"Pathway(steps={self.steps})"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.summary()


class ReactionPathwayExplorer:
    """
    Enumerate forward/backward pathways on a ReactionNetwork.

    The provided `network` must expose:
      - `network.reactions: Dict[int, Reaction]`
      - every Reaction has:
        * `reactants_can: Counter`
        * `products_can: Counter`
        * `apply_forward(state: Counter, matched: Counter) -> Counter`
        * `apply_backward(state: Counter, matched: Counter) -> Counter`
        * `can_fire_forward(state: Counter, min_overlap: int) -> Tuple[bool, Counter]`
        * `can_fire_backward(state: Counter, min_overlap: int) -> Tuple[bool, Counter]`
        * `original_raw` / `canonical_raw` (for display)

    :param network: ReactionNetwork-like object.
    :type network: Any
    """

    def __init__(self, network: Any) -> None:
        self.net = network
        self.pathways: List[Pathway] = []

    # ---------------------------
    # Public API
    # ---------------------------

    def find_forward(
        self,
        start: Counter,
        goal: Counter,
        *,
        min_overlap: int = DEFAULT_MIN_OVERLAP,
        allow_reuse: bool = False,
        strategy: str = "dfs",
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_pathways: int = 256,
        stop_on_goal: bool = True,
        disallow_reactions: Optional[Set[int]] = None,
        reaction_predicate: Optional[Callable[[int, Any], bool]] = None,
        enforce_stoichiometry: bool = True,
        infer_missing: bool = False,
    ) -> "ReactionPathwayExplorer":
        """
        Enumerate forward pathways from ``start`` to ``goal`` and store them in
        :pyattr:`pathways`.

        :param start: Starting state (multiset of species).
        :type start: Counter
        :param goal: Goal condition as a multiset; a state meets the goal when it
                     contains at least these counts.
        :type goal: Counter
        :param min_overlap: Minimal overlap for flexible mode (only used when
                            ``enforce_stoichiometry`` is False).
        :type min_overlap: int
        :param allow_reuse: If False, a reaction id may appear at most once per path.
        :type allow_reuse: bool
        :param strategy: 'dfs' (depth-first) or 'bfs' (breadth-first).
        :type strategy: str
        :param max_depth: Maximum path length (number of reactions).
        :type max_depth: int
        :param max_pathways: Maximum number of pathways to collect.
        :type max_pathways: int
        :param stop_on_goal: If True, stop expanding a branch after the first goal hit.
        :type stop_on_goal: bool
        :param disallow_reactions: Reaction ids to exclude from expansion.
        :type disallow_reactions: set[int] | None
        :param reaction_predicate: Optional filter `(rid, reaction) -> bool` to include/exclude.
        :type reaction_predicate: Callable[[int, Any], bool] | None
        :param enforce_stoichiometry: If True (default), require the full reactant multiset
                                      to be present before firing a reaction.
        :type enforce_stoichiometry: bool
        :param infer_missing: If True and in enforced mode, allow *guarded inference*:
                              seed only the missing co-reactants **when** the reaction
                              consumes at least one species already present in the state.
        :type infer_missing: bool
        :returns: ``self`` with results stored on :pyattr:`pathways`.
        :rtype: ReactionPathwayExplorer
        :raises SearchError: On invalid strategy value.
        """
        self._validate_strategy(strategy)

        forbidden = disallow_reactions or set()
        push, pop = self._make_fringe_ops(strategy)
        visited: Dict[int, Set[Tuple[Tuple[str, int], ...]]] = {}
        results: List[Pathway] = []

        # fringe entries: (state, rids, hist, used)
        fringe: Deque = deque()
        push(fringe, (start.copy(), [], [start.copy()], set()))

        while fringe and len(results) < max_pathways:
            state, rids, hist, used = pop(fringe)
            if len(rids) > max_depth:
                continue
            if self._seen(visited, len(rids), state):
                continue

            if self._meets_goal(state, goal):
                results.append(Pathway(list(rids), list(hist)))
                if stop_on_goal:
                    continue

            for rid, ns in self._forward_successors(
                state=state,
                used=used,
                allow_reuse=allow_reuse,
                forbidden=forbidden,
                predicate=reaction_predicate,
                enforce=enforce_stoichiometry,
                infer_missing=infer_missing,
                min_overlap=min_overlap,
            ):
                nrids = rids + [rid]
                nhist = hist + [ns.copy()]
                nused = used if allow_reuse else (used | {rid})
                push(fringe, (ns, nrids, nhist, nused))

        self.pathways = results
        return self

    def find_reverse(
        self,
        start: Counter,
        goal: Counter,
        *,
        min_overlap: int = DEFAULT_MIN_OVERLAP,
        allow_reuse: bool = False,
        strategy: str = "dfs",
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_pathways: int = 256,
        stop_on_start: bool = True,
        disallow_reactions: Optional[Set[int]] = None,
        reaction_predicate: Optional[Callable[[int, Any], bool]] = None,
        enforce_stoichiometry: bool = True,
        infer_missing: bool = False,
    ) -> "ReactionPathwayExplorer":
        """
        Enumerate backward pathways from ``goal`` towards ``start`` and store them
        on :pyattr:`pathways`. Returned reaction ids/states are oriented in the
        forward direction.

        :param start: Target start state for the backward search.
        :type start: Counter
        :param goal: Initial state from which to expand backward.
        :type goal: Counter
        :param min_overlap: Minimal overlap for flexible backward mode (only used
                            when ``enforce_stoichiometry`` is False).
        :type min_overlap: int
        :param allow_reuse: If False, a reaction id may appear at most once per path.
        :type allow_reuse: bool
        :param strategy: 'dfs' (depth-first) or 'bfs' (breadth-first).
        :type strategy: str
        :param max_depth: Maximum path length (number of reactions).
        :type max_depth: int
        :param max_pathways: Maximum number of pathways to collect.
        :type max_pathways: int
        :param stop_on_start: If True, stop expanding a branch after the first start hit.
        :type stop_on_start: bool
        :param disallow_reactions: Reaction ids to exclude from expansion.
        :type disallow_reactions: set[int] | None
        :param reaction_predicate: Optional filter `(rid, reaction) -> bool` to include/exclude.
        :type reaction_predicate: Callable[[int, Any], bool] | None
        :param enforce_stoichiometry: If True (default), require full product multiset
                                      to be present before a backward step.
        :type enforce_stoichiometry: bool
        :param infer_missing: If True and in enforced mode, allow guarded inference of
                              missing products only if the reaction shares at least one
                              product already present in the state.
        :type infer_missing: bool
        :returns: ``self`` with results stored on :pyattr:`pathways`.
        :rtype: ReactionPathwayExplorer
        :raises SearchError: On invalid strategy value.
        """
        self._validate_strategy(strategy)

        forbidden = disallow_reactions or set()
        push, pop = self._make_fringe_ops(strategy)
        visited: Dict[int, Set[Tuple[Tuple[str, int], ...]]] = {}
        results: List[Pathway] = []

        # fringe entries: (state_back, rids_back, hist_back, used)
        fringe: Deque = deque()
        push(fringe, (goal.copy(), [], [goal.copy()], set()))

        while fringe and len(results) < max_pathways:
            state_b, rids_b, hist_b, used = pop(fringe)
            if len(rids_b) > max_depth:
                continue
            if self._seen(visited, len(rids_b), state_b):
                continue

            if self._meets_goal(state_b, start):
                # convert collected backward to forward orientation
                f_rids = list(reversed(rids_b))
                f_hist = list(reversed(hist_b))
                results.append(Pathway(f_rids, f_hist))
                if stop_on_start:
                    continue

            for rid, ns in self._backward_successors(
                state=state_b,
                used=used,
                allow_reuse=allow_reuse,
                forbidden=forbidden,
                predicate=reaction_predicate,
                enforce=enforce_stoichiometry,
                infer_missing=infer_missing,
                min_overlap=min_overlap,
            ):
                nrids_b = rids_b + [rid]
                nhist_b = hist_b + [ns.copy()]
                nused = used if allow_reuse else (used | {rid})
                push(fringe, (ns, nrids_b, nhist_b, nused))

        self.pathways = results
        return self

    # ---------------------------
    # Small helpers (keep methods simple -> flake8 C901 friendly)
    # ---------------------------

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        if strategy not in {"dfs", "bfs"}:
            raise SearchError("strategy must be 'dfs' or 'bfs'")

    @staticmethod
    def _make_fringe_ops(strategy: str):
        def push(dq: Deque, item) -> None:
            dq.append(item)

        def pop_dfs(dq: Deque):
            return dq.pop()

        def pop_bfs(dq: Deque):
            return dq.popleft()

        return (push, pop_dfs if strategy == "dfs" else pop_bfs)

    @staticmethod
    def _meets_goal(state: Counter, goal: Counter) -> bool:
        return all(state.get(k, 0) >= v for k, v in goal.items())

    @staticmethod
    def _seen(
        visited: Dict[int, Set[Tuple[Tuple[str, int], ...]]], depth: int, state: Counter
    ) -> bool:
        key = counter_key(state)
        bucket = visited.setdefault(depth, set())
        if key in bucket:
            return True
        bucket.add(key)
        return False

    @staticmethod
    def _shares_present(state: Counter, need: Counter) -> bool:
        return any(state.get(k, 0) > 0 for k in need.keys())

    @staticmethod
    def _seed_missing_copy(state: Counter, need: Counter) -> Counter:
        ns = state.copy()
        for k, v in need.items():
            deficit = v - ns.get(k, 0)
            if deficit > 0:
                ns[k] = ns.get(k, 0) + deficit
        return ns

    def _iter_reactions(
        self,
        *,
        used: Set[int],
        allow_reuse: bool,
        forbidden: Set[int],
        predicate: Optional[Callable[[int, Any], bool]],
    ) -> Iterable[Tuple[int, Any]]:
        for rid, rx in self.net.reactions.items():
            if rid in forbidden:
                continue
            if (not allow_reuse) and (rid in used):
                continue
            if predicate and not predicate(rid, rx):
                continue
            yield rid, rx

    # ---- successor generators ----

    def _forward_successors(
        self,
        *,
        state: Counter,
        used: Set[int],
        allow_reuse: bool,
        forbidden: Set[int],
        predicate: Optional[Callable[[int, Any], bool]],
        enforce: bool,
        infer_missing: bool,
        min_overlap: int,
    ) -> Iterable[Tuple[int, Counter]]:
        """
        Yield (rid, next_state) for forward expansion under the configured policy.
        """
        for rid, rx in self._iter_reactions(
            used=used, allow_reuse=allow_reuse, forbidden=forbidden, predicate=predicate
        ):
            if enforce:
                need = rx.reactants_can
                if multiset_contains(state, need):
                    matched = need.copy()
                    yield rid, rx.apply_forward(state, matched)
                    continue
                if infer_missing and self._shares_present(state, need):
                    tmp = self._seed_missing_copy(state, need)
                    matched = need.copy()
                    yield rid, rx.apply_forward(tmp, matched)
                    continue
                # else: cannot fire
            else:
                ok, matched = rx.can_fire_forward(state, min_overlap)
                if ok:
                    yield rid, rx.apply_forward(state, matched)

    def _backward_successors(
        self,
        *,
        state: Counter,
        used: Set[int],
        allow_reuse: bool,
        forbidden: Set[int],
        predicate: Optional[Callable[[int, Any], bool]],
        enforce: bool,
        infer_missing: bool,
        min_overlap: int,
    ) -> Iterable[Tuple[int, Counter]]:
        """
        Yield (rid, next_state) for backward expansion under the configured policy.
        """
        for rid, rx in self._iter_reactions(
            used=used, allow_reuse=allow_reuse, forbidden=forbidden, predicate=predicate
        ):
            if enforce:
                need = rx.products_can
                if multiset_contains(state, need):
                    matched = need.copy()
                    yield rid, rx.apply_backward(state, matched)
                    continue
                if infer_missing and self._shares_present(state, need):
                    tmp = self._seed_missing_copy(state, need)
                    matched = need.copy()
                    yield rid, rx.apply_backward(tmp, matched)
                    continue
            else:
                ok, matched = rx.can_fire_backward(state, min_overlap)
                if ok:
                    yield rid, rx.apply_backward(state, matched)
