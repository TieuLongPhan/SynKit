from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Set, Tuple, Iterable, Callable

from .constants import DEFAULT_MIN_OVERLAP, DEFAULT_MAX_DEPTH
from .exceptions import SearchError
from .network import ReactionNetwork
from .pathway import Pathway
from .reaction import Reaction
from .utils import counter_key, parse_state, inflow_outflow, multiset_contains


class ReactionPathwayExplorer:
    """
    Pathway enumeration over a :class:`crn.network.ReactionNetwork`.

    Fluent API
    ----------
    Search methods return ``self`` and store results internally in ``.pathways``.
    Example:

    >>> explorer = ReactionPathwayExplorer(net).find_forward(start, goal).filter_by_flow(require_outflow="E")
    >>> results = explorer.pathways

    :param network: Reaction network to explore.
    """

    def __init__(self, network: ReactionNetwork) -> None:
        self.net = network
        self._pathways: List[Pathway] = []

    # ---- Results accessors ----
    @property
    def pathways(self) -> List[Pathway]:
        """Last search results."""
        return list(self._pathways)

    def _store(self, paths: List[Pathway]) -> "ReactionPathwayExplorer":
        self._pathways = paths
        return self

    # ---- Flow helpers ----
    @staticmethod
    def reaction_net_change(rxn: Reaction) -> Counter:
        """Net change for one reaction."""
        return rxn.net_change

    @staticmethod
    def compute_state_flow(before: Counter, after: Counter) -> Tuple[Counter, Counter]:
        """(inflow, outflow) between states."""
        return inflow_outflow(before, after)

    def compute_pathway_flow(self, pathway: Pathway) -> Tuple[Counter, Counter]:
        """(inflow, outflow) for a pathway."""
        if not pathway.states:
            return Counter(), Counter()
        return self.compute_state_flow(pathway.states[0], pathway.states[-1])

    # ---- Filtering on stored results (fluent) ----
    def filter_by_flow(
        self,
        *,
        min_net_gain: Optional[int] = None,
        require_inflow: Optional[Iterable[str] | str | Dict[str, int]] = None,
        require_outflow: Optional[Iterable[str] | str | Dict[str, int]] = None,
        inflow_exact: bool = False,
        inflow_any: bool = False,
        outflow_exact: bool = False,
        outflow_any: bool = False,
    ) -> "ReactionPathwayExplorer":
        """
        Filter the stored pathways by inflow/outflow properties.

        :param min_net_gain: Keep if total inflow >= this threshold.
        :param require_inflow: Required inflow multiset (None to ignore).
        :param require_outflow: Required outflow multiset (None to ignore).
        :param inflow_exact: Require exact match for inflow.
        :param inflow_any: Accept if any of the required inflow keys appear.
        :param outflow_exact: Require exact match for outflow.
        :param outflow_any: Accept if any of the required outflow keys appear.
        :returns: ``self``.
        """
        req_in = parse_state(require_inflow)
        req_out = parse_state(require_outflow)

        def _ok(
            actual: Counter, req: Optional[Counter], *, exact: bool, any_of: bool
        ) -> bool:
            if req is None:
                return True
            if any_of:
                return any(actual.get(k, 0) >= max(1, v) for k, v in req.items())
            if exact:
                return actual == req
            return multiset_contains(actual, req)

        kept: List[Pathway] = []
        for p in self._pathways:
            inflow_c, outflow_c = self.compute_pathway_flow(p)
            if min_net_gain is not None and sum(inflow_c.values()) < min_net_gain:
                continue
            if not _ok(inflow_c, req_in, exact=inflow_exact, any_of=inflow_any):
                continue
            if not _ok(outflow_c, req_out, exact=outflow_exact, any_of=outflow_any):
                continue
            kept.append(p)
        return self._store(kept)

    # ---- Forward enumeration ----
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
        reaction_predicate: Optional[Callable[[int, Reaction], bool]] = None,
    ) -> "ReactionPathwayExplorer":
        """
        Enumerate forward pathways from ``start`` to ``goal``.

        :param start: Start state.
        :param goal: Required goal (subset condition).
        :param min_overlap: Minimal overlap to fire a reaction.
        :param allow_reuse: Allow reusing reaction ids.
        :param strategy: 'dfs' or 'bfs'.
        :param max_depth: Depth limit.
        :param max_pathways: Maximum number of pathways to collect.
        :param stop_on_goal: Stop expanding a branch once goal is reached.
        :param disallow_reactions: Reaction ids to exclude.
        :param reaction_predicate: Custom filter (rid, Reaction) -> bool.
        :returns: ``self`` with results stored in :pyattr:`pathways`.
        """
        if strategy not in {"dfs", "bfs"}:
            raise SearchError("strategy must be 'dfs' or 'bfs'")

        forbidden = disallow_reactions or set()
        fringe: Deque = deque()
        push = fringe.append
        pop = fringe.pop if strategy == "dfs" else fringe.popleft
        push((start.copy(), [], [start.copy()], set()))
        visited: Dict[int, set] = {}
        out: List[Pathway] = []

        while fringe and len(out) < max_pathways:
            state, rids, hist, used = pop()
            depth = len(rids)
            if depth > max_depth:
                continue
            key = counter_key(state)
            if key in visited.setdefault(depth, set()):
                continue
            visited[depth].add(key)

            # goal check (subset)
            if all(state.get(k, 0) >= v for k, v in goal.items()):
                out.append(Pathway(reaction_ids=list(rids), states=list(hist)))
                if stop_on_goal:
                    continue

            for rid, rx in self.net.reactions.items():
                if rid in forbidden or ((not allow_reuse) and (rid in used)):
                    continue
                if reaction_predicate and not reaction_predicate(rid, rx):
                    continue
                ok, matched = rx.can_fire_forward(state, min_overlap)
                if not ok:
                    continue
                ns = rx.apply_forward(state, matched)
                nr = rids + [rid]
                nh = hist + [ns.copy()]
                nu = set(used)
                if not allow_reuse:
                    nu.add(rid)
                push((ns, nr, nh, nu))

        return self._store(out)

    # ---- Reverse enumeration ----
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
    ) -> "ReactionPathwayExplorer":
        """
        Enumerate backward pathways from ``goal`` down to ``start`` (returned in forward order).

        :returns: ``self``.
        """
        if strategy not in {"dfs", "bfs"}:
            raise SearchError("strategy must be 'dfs' or 'bfs'")

        fringe: Deque = deque()
        push = fringe.append
        pop = fringe.pop if strategy == "dfs" else fringe.popleft
        push((goal.copy(), [], [goal.copy()], set()))
        visited: Dict[int, set] = {}
        out: List[Pathway] = []

        while fringe and len(out) < max_pathways:
            state, rids_back, hist_back, used = pop()
            depth = len(rids_back)
            if depth > max_depth:
                continue
            key = counter_key(state)
            if key in visited.setdefault(depth, set()):
                continue
            visited[depth].add(key)

            if all(state.get(k, 0) >= v for k, v in start.items()):
                forward_rids = list(reversed(rids_back))
                forward_states = list(reversed(hist_back))
                out.append(Pathway(reaction_ids=forward_rids, states=forward_states))
                if stop_on_start:
                    continue

            for rid, rx in self.net.reactions.items():
                if (not allow_reuse) and (rid in used):
                    continue
                ok, matched = rx.can_fire_backward(state, min_overlap)
                if not ok:
                    continue
                ns = rx.apply_backward(state, matched)
                push(
                    (
                        ns,
                        rids_back + [rid],
                        hist_back + [ns.copy()],
                        used | ({rid} if not allow_reuse else set()),
                    )
                )

        return self._store(out)

    # ---- A* (fewest steps) ----
    def find_a_star(
        self,
        start: Counter,
        goal: Counter,
        *,
        min_overlap: int = DEFAULT_MIN_OVERLAP,
        allow_reuse: bool = False,
        max_nodes: int = 200000,
        max_pathways: int = 1,
    ) -> "ReactionPathwayExplorer":
        """
        A* search minimizing number of steps.

        :returns: ``self``.
        """
        max_goal_prod = 0
        for rx in self.net.reactions.values():
            inter = rx.products_can & goal
            if sum(inter.values()) > max_goal_prod:
                max_goal_prod = sum(inter.values())

        def h(s: Counter) -> int:
            missing = sum((goal - s).values())
            if missing <= 0:
                return 0
            denom = max(1, max_goal_prod)
            return math.ceil(missing / denom)

        start_key = counter_key(start)
        open_heap: List[tuple] = []
        heapq.heappush(
            open_heap,
            (h(start), 0, 0, start_key, start.copy(), [], [start.copy()], set()),
        )
        best_g: Dict[tuple, int] = {start_key: 0}
        results: List[Pathway] = []
        tie = 1
        explored = 0

        while open_heap and len(results) < max_pathways and explored < max_nodes:
            f, g, _, key, state, rids, hist, used = heapq.heappop(open_heap)
            explored += 1
            if best_g.get(key, 1e9) < g:
                continue
            if sum((goal - state).values()) == 0:
                results.append(Pathway(reaction_ids=list(rids), states=list(hist)))
                continue

            for rid, rx in self.net.reactions.items():
                if (not allow_reuse) and (rid in used):
                    continue
                ok, matched = rx.can_fire_forward(state, min_overlap)
                if not ok:
                    continue
                ns = rx.apply_forward(state, matched)
                nk = counter_key(ns)
                ng = g + 1
                if ng < best_g.get(nk, 1e9):
                    best_g[nk] = ng
                    nf = ng + h(ns)
                    heapq.heappush(
                        open_heap,
                        (
                            nf,
                            ng,
                            tie,
                            nk,
                            ns.copy(),
                            rids + [rid],
                            hist + [ns.copy()],
                            used | ({rid} if not allow_reuse else set()),
                        ),
                    )
                    tie += 1

        return self._store(results)
