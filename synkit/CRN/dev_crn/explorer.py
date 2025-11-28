from __future__ import annotations

from collections import Counter
from typing import Any, Callable, List, Optional, Set

from .utils import multiset_contains
from .exceptions import SearchError


class Pathway:
    """Lightweight pathway container: ordered reaction ids + states."""

    def __init__(
        self,
        reaction_ids: Optional[List[int]] = None,
        states: Optional[List[Counter]] = None,
    ) -> None:
        self.reaction_ids: List[int] = list(reaction_ids or [])
        self.states: List[Counter] = list(states or [])

    @property
    def steps(self) -> int:
        return len(self.reaction_ids)

    def __repr__(self) -> str:
        return f"Pathway(steps={self.steps})"


class ReactionPathwayExplorer:
    """
    Deterministic DFS enumerator of forward/backward pathways.

    - This implementation uses recursive backtracking to enumerate all ordered
      feasible sequences (each reaction used at most once per path when
      allow_reuse=False). It avoids over-aggressive pruning by not relying on
      a compressed "seen" multiset key; instead it enumerates with a `used` set
      and checks stoichiometry at each choice.
    - Expected Reaction API (present in your repo):
        - rx.reactants_can : Counter
        - rx.products_can  : Counter
        - rx.apply_forward(state, matched) -> Counter
        - rx.apply_backward(state, matched) -> Counter
        - rx.can_fire_forward(state, min_overlap) -> (bool, matched)
        - rx.can_fire_backward(state, min_overlap) -> (bool, matched)
        - rx.original_raw / rx.canonical_raw
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
        min_overlap: int = 1,
        allow_reuse: bool = False,
        max_depth: int = 30,
        max_pathways: int = 256,
        disallow_reactions: Optional[Set[int]] = None,
        reaction_predicate: Optional[Callable[[int, Any], bool]] = None,
        enforce_stoichiometry: bool = True,
        infer_missing: bool = False,
    ) -> "ReactionPathwayExplorer":
        """
        Enumerate forward pathways from `start` to `goal`.

        Results are stored in `self.pathways` (list of Pathway).
        """
        if max_depth < 0:
            raise SearchError("max_depth must be >= 0")

        forbidden = disallow_reactions or set()
        self.pathways = []

        # Precompute ordered reaction ids to iterate deterministically
        candidate_rids = [rid for rid in sorted(self.net.reactions.keys())]

        def _recurse(
            state: Counter, path: List[int], states_hist: List[Counter], used: Set[int]
        ) -> None:
            # stop conditions
            if len(self.pathways) >= max_pathways:
                return
            if len(path) > max_depth:
                return
            # goal test
            if all(state.get(k, 0) >= v for k, v in goal.items()):
                self.pathways.append(Pathway(list(path), list(states_hist)))
                return

            # Try all candidate reactions in deterministic order
            for rid in candidate_rids:
                if rid in forbidden:
                    continue
                if (not allow_reuse) and (rid in used):
                    continue
                rx = self.net.reactions[rid]
                if reaction_predicate and not reaction_predicate(rid, rx):
                    continue

                # handle Source activation specially (optional) — but stoichiometry check below will catch it
                if enforce_stoichiometry:
                    need = rx.reactants_can
                    if multiset_contains(state, need):
                        # apply forward
                        matched = need.copy()
                        newstate = rx.apply_forward(state, matched)
                    elif infer_missing and any(
                        state.get(k, 0) > 0 for k in need.keys()
                    ):
                        tmp = state.copy()
                        for k, v in need.items():
                            if tmp.get(k, 0) < v:
                                tmp[k] = tmp.get(k, 0) + (v - tmp.get(k, 0))
                        matched = need.copy()
                        newstate = rx.apply_forward(tmp, matched)
                    else:
                        continue  # cannot fire this reaction now
                else:
                    ok, matched = rx.can_fire_forward(state, min_overlap)
                    if not ok:
                        continue
                    # ensure matched is a Counter-like if necessary
                    newstate = rx.apply_forward(state, matched)

                # proceed with this reaction
                path.append(rid)
                states_hist.append(newstate.copy())
                used_added = False
                if not allow_reuse:
                    used.add(rid)
                    used_added = True

                _recurse(newstate, path, states_hist, used)

                # backtrack
                if used_added:
                    used.remove(rid)
                states_hist.pop()
                path.pop()

                if len(self.pathways) >= max_pathways:
                    return

        # initial call
        _recurse(start.copy(), [], [start.copy()], set())

        return self

    def find_reverse(
        self,
        start: Counter,
        goal: Counter,
        *,
        min_overlap: int = 1,
        allow_reuse: bool = False,
        max_depth: int = 30,
        max_pathways: int = 256,
        disallow_reactions: Optional[Set[int]] = None,
        reaction_predicate: Optional[Callable[[int, Any], bool]] = None,
        enforce_stoichiometry: bool = True,
        infer_missing: bool = False,
    ) -> "ReactionPathwayExplorer":
        """
        Backward enumeration: expand from `goal` backwards. Collected pathways
        are returned oriented forward (reaction ids in forward order).
        """
        if max_depth < 0:
            raise SearchError("max_depth must be >= 0")

        forbidden = disallow_reactions or set()
        self.pathways = []

        candidate_rids = [rid for rid in sorted(self.net.reactions.keys())]

        def _recurse_back(
            state: Counter,
            path_back: List[int],
            states_hist_back: List[Counter],
            used: Set[int],
        ) -> None:
            if len(self.pathways) >= max_pathways:
                return
            if len(path_back) > max_depth:
                return
            # goal test: have we reached the desired start?
            if all(state.get(k, 0) >= v for k, v in start.items()):
                # orient forward before recording
                f_rids = list(reversed(path_back))
                f_states = list(reversed(states_hist_back))
                self.pathways.append(Pathway(f_rids, f_states))
                return

            for rid in candidate_rids:
                if rid in forbidden:
                    continue
                if (not allow_reuse) and (rid in used):
                    continue
                rx = self.net.reactions[rid]
                if reaction_predicate and not reaction_predicate(rid, rx):
                    continue

                if enforce_stoichiometry:
                    need = rx.products_can
                    if multiset_contains(state, need):
                        matched = need.copy()
                        newstate = rx.apply_backward(state, matched)
                    elif infer_missing and any(
                        state.get(k, 0) > 0 for k in need.keys()
                    ):
                        tmp = state.copy()
                        for k, v in need.items():
                            if tmp.get(k, 0) < v:
                                tmp[k] = tmp.get(k, 0) + (v - tmp.get(k, 0))
                        matched = need.copy()
                        newstate = rx.apply_backward(tmp, matched)
                    else:
                        continue
                else:
                    ok, matched = rx.can_fire_backward(state, min_overlap)
                    if not ok:
                        continue
                    newstate = rx.apply_backward(state, matched)

                path_back.append(rid)
                states_hist_back.append(newstate.copy())
                used_added = False
                if not allow_reuse:
                    used.add(rid)
                    used_added = True

                _recurse_back(newstate, path_back, states_hist_back, used)

                if used_added:
                    used.remove(rid)
                states_hist_back.pop()
                path_back.pop()

                if len(self.pathways) >= max_pathways:
                    return

        _recurse_back(goal.copy(), [], [goal.copy()], set())
        return self
