from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Set, Tuple, Iterable, Any, Union

from .network import ReactionNetwork
from .pathway import Pathway
from .reaction import Reaction
from .utils import counter_key


class ReactionPathwayExplorer:
    """
    High-level explorer for reaction pathway enumeration and shortest-path search.

    Matching model
    --------------
    A reaction is applicable if its reactants (forward) or products (backward) have
    a multiset intersection with the current state of size at least ``min_overlap``.
    Only the matched instances are consumed; other required co-reactants are assumed
    to be externally available.

    :param network: Network to search over.
    :type network: ReactionNetwork
    """

    def __init__(self, network: ReactionNetwork) -> None:
        self.net = network

    # ---------------------------------------------------------------------
    # Internal helpers (apply transitions)
    # ---------------------------------------------------------------------
    @staticmethod
    def _can_fire_forward(
        rxn: Reaction, state: Counter[str], min_overlap: int
    ) -> Tuple[bool, Counter[str]]:
        matched = rxn.reactants_can & state
        return (sum(matched.values()) >= min_overlap, matched)

    @staticmethod
    def _apply_forward(
        state: Counter[str], matched: Counter[str], products: Counter[str]
    ) -> Counter[str]:
        new_state = state - matched
        new_state += products
        for k in list(new_state.keys()):
            if new_state[k] <= 0:
                del new_state[k]
        return new_state

    @staticmethod
    def _can_fire_backward(
        rxn: Reaction, state: Counter[str], min_overlap: int
    ) -> Tuple[bool, Counter[str]]:
        matched = rxn.products_can & state
        return (sum(matched.values()) >= min_overlap, matched)

    @staticmethod
    def _apply_backward(
        state: Counter[str], matched_products: Counter[str], reactants: Counter[str]
    ) -> Counter[str]:
        new_state = state - matched_products
        new_state += reactants
        for k in list(new_state.keys()):
            if new_state[k] <= 0:
                del new_state[k]
        return new_state

    # ---------------------------------------------------------------------
    # Inflow / Outflow analytics
    # ---------------------------------------------------------------------
    def reaction_net_change(self, rxn: Reaction) -> Counter[str]:
        """
        Net change for a single reaction (products − reactants).

        Positive counts => net production; negative counts => net consumption.
        """
        net: Counter[str] = Counter()
        for p, c in rxn.products_can.items():
            net[p] += c
        for r, c in rxn.reactants_can.items():
            net[r] -= c
        return net

    @staticmethod
    def compute_state_flow(
        before: Counter[str], after: Counter[str]
    ) -> Tuple[Counter[str], Counter[str]]:
        """
        Compute (inflow, outflow) between two states.

        - inflow = (after - before)  [only positive entries kept]
        - outflow = (before - after) [only positive entries kept]
        """
        inflow = Counter(after - before)
        outflow = Counter(before - after)
        return inflow, outflow

    def compute_pathway_flow(
        self, pathway: Pathway
    ) -> Tuple[Counter[str], Counter[str]]:
        """
        Compute (inflow, outflow) for a full pathway w.r.t. its initial state.
        """
        if not pathway.states:
            return Counter(), Counter()
        return self.compute_state_flow(pathway.states[0], pathway.states[-1])

    def flow_summary(self, pathway: Pathway) -> str:
        """
        String summary of a pathway's inflow/outflow.
        """
        inflow, outflow = self.compute_pathway_flow(pathway)

        def _fmt(c: Counter[str]) -> str:
            return "-" if not c else ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))

        return f"inflow: {_fmt(inflow)}    outflow: {_fmt(outflow)}"

    # ----- flexible requirements parsing / checks -----
    def _parse_requirement(
        self, req: Optional[Union[str, List[str], Dict[str, int]]]
    ) -> Optional[Counter[str]]:
        """
        Parse requirement into a Counter (multiset).

        Accepts:
          - None                -> None
          - "A.C"               -> Counter({'A':1,'C':1})
          - "A"                 -> Counter({'A':1})
          - ["A","C"]           -> Counter({'A':1,'C':1})
          - {"A":2,"C":1}       -> Counter({'A':2,'C':1})
        """
        if req is None:
            return None
        if isinstance(req, dict):
            return Counter({str(k): int(v) for k, v in req.items()})
        if isinstance(req, list):
            return Counter(str(x) for x in req)
        if isinstance(req, str):
            parts = [p.strip() for p in req.split(".") if p.strip()]
            return Counter(parts)
        return Counter([str(req)])  # last-resort coercion

    def _satisfy_requirement(
        self,
        actual: Counter[str],
        required: Optional[Counter[str]],
        *,
        exact: bool = False,
        any_of: bool = False,
    ) -> bool:
        """
        Check if `actual` (inflow/outflow) satisfies `required`.

        Rules:
          - required is None  -> True
          - any_of=True       -> at least one key in required appears in actual (count >= required[k] or >=1)
          - exact=True        -> exact multiset equality
          - default           -> actual contains required (actual[k] >= required[k] for all k)
        """
        if required is None:
            return True
        if any_of:
            for k, v in required.items():
                need = max(1, v)
                if actual.get(k, 0) >= need:
                    return True
            return False
        if exact:
            return actual == required
        # containment
        for k, v in required.items():
            if actual.get(k, 0) < v:
                return False
        return True

    def filter_pathways_by_flow(
        self,
        pathways: Iterable[Pathway],
        *,
        min_net_gain: Optional[int] = None,
        require_inflow: Optional[Union[str, List[str], Dict[str, int]]] = None,
        require_outflow: Optional[Union[str, List[str], Dict[str, int]]] = None,
        require_inflow_exact: bool = False,
        require_inflow_any: bool = False,
        require_outflow_exact: bool = False,
        require_outflow_any: bool = False,
    ) -> List[Pathway]:
        """
        Filter pathways using inflow/outflow criteria.

        :param min_net_gain: numeric threshold on total inflow (sum of counts).
        :param require_inflow: required inflow multiset (str/list/dict). "A.C" means at least 1 A and 1 C.
        :param require_outflow: required outflow multiset (str/list/dict).
        :param require_inflow_exact: inflow must equal the multiset exactly.
        :param require_inflow_any: inflow must contain *any* of the listed molecules.
        :param require_outflow_exact: outflow must equal the multiset exactly.
        :param require_outflow_any: outflow must contain *any* of the listed molecules.
        :return: filtered pathways (order preserved).
        """
        req_in = self._parse_requirement(require_inflow)
        req_out = self._parse_requirement(require_outflow)

        kept: List[Pathway] = []
        for p in pathways:
            inflow, outflow = self.compute_pathway_flow(p)
            if min_net_gain is not None and sum(inflow.values()) < min_net_gain:
                continue
            if not self._satisfy_requirement(
                inflow, req_in, exact=require_inflow_exact, any_of=require_inflow_any
            ):
                continue
            if not self._satisfy_requirement(
                outflow,
                req_out,
                exact=require_outflow_exact,
                any_of=require_outflow_any,
            ):
                continue
            kept.append(p)
        return kept

    def find_forward_filter_by_flow(
        self,
        start: Counter[str],
        goal: Counter[str],
        *,
        min_overlap: int = 1,
        allow_reuse: bool = False,
        strategy: str = "dfs",
        stop_on_goal: bool = True,
        must_include_last_reaction: bool = True,
        max_depth: Optional[int] = 10,
        max_pathways: int = 1000,
        keep_shortest_only: bool = False,
        # flow filters:
        min_net_gain: Optional[int] = None,
        require_inflow: Optional[Union[str, List[str], Dict[str, int]]] = None,
        require_outflow: Optional[Union[str, List[str], Dict[str, int]]] = None,
        require_inflow_exact: bool = False,
        require_inflow_any: bool = False,
        require_outflow_exact: bool = False,
        require_outflow_any: bool = False,
    ) -> List[Pathway]:
        """
        Run :meth:`find_forward` then apply flow-based filtering.
        """
        paths = self.find_forward(
            start=start,
            goal=goal,
            min_overlap=min_overlap,
            allow_reuse=allow_reuse,
            strategy=strategy,
            stop_on_goal=stop_on_goal,
            must_include_last_reaction=must_include_last_reaction,
            max_depth=max_depth,
            max_pathways=max_pathways,
            keep_shortest_only=keep_shortest_only,
        )
        return self.filter_pathways_by_flow(
            paths,
            min_net_gain=min_net_gain,
            require_inflow=require_inflow,
            require_outflow=require_outflow,
            require_inflow_exact=require_inflow_exact,
            require_inflow_any=require_inflow_any,
            require_outflow_exact=require_outflow_exact,
            require_outflow_any=require_outflow_any,
        )

    # ---------------------------------------------------------------------
    # Forward search (DFS/BFS)
    # ---------------------------------------------------------------------
    def find_forward(
        self,
        start: Counter[str],
        goal: Counter[str],
        *,
        min_overlap: int = 1,
        allow_reuse: bool = False,
        strategy: str = "dfs",
        stop_on_goal: bool = True,
        must_include_last_reaction: bool = True,
        max_depth: Optional[int] = 10,
        max_pathways: int = 1000,
        keep_shortest_only: bool = False,
    ) -> List[Pathway]:
        """
        Enumerate forward pathways from ``start`` to ``goal``.
        """
        if max_depth is None:
            max_depth = max(8, len(self.net.reactions) * (2 if allow_reuse else 1))

        last_rid = max(self.net.reactions.keys()) if self.net.reactions else None
        found: List[Pathway] = []

        if strategy not in {"dfs", "bfs"}:
            raise ValueError('strategy must be "dfs" or "bfs"')

        fringe: Deque = deque()
        push = fringe.append
        pop = fringe.pop if strategy == "dfs" else fringe.popleft

        push((start.copy(), [], [start.copy()], set()))
        visited_by_depth: Dict[int, Set[Tuple[Tuple[str, int], ...]]] = {}

        while fringe:
            state, rids, history, used = pop()
            depth = len(rids)
            if depth > max_depth:
                continue

            key = counter_key(state)
            if key in visited_by_depth.setdefault(depth, set()):
                continue
            visited_by_depth[depth].add(key)

            # goal satisfied (subset)
            if all(state.get(m, 0) >= c for m, c in goal.items()):
                if (not must_include_last_reaction) or (
                    last_rid in rids if last_rid is not None else True
                ):
                    found.append(Pathway(reaction_ids=list(rids), states=list(history)))
                    if len(found) >= max_pathways:
                        break
                if stop_on_goal:
                    continue

            for rid, rxn in self.net.reactions.items():
                if (not allow_reuse) and (rid in used):
                    continue
                ok, matched = self._can_fire_forward(rxn, state, min_overlap)
                if not ok:
                    continue
                new_state = self._apply_forward(state, matched, rxn.products_can)
                new_rids = rids + [rid]
                new_history = history + [new_state.copy()]
                new_used = set(used)
                if not allow_reuse:
                    new_used.add(rid)
                push((new_state, new_rids, new_history, new_used))

        if keep_shortest_only and found:
            found.sort(key=lambda p: len(p.reaction_ids))
            kept: List[Pathway] = []
            seen_final: Set[Tuple[Tuple[str, int], ...]] = set()
            for p in found:
                fk = counter_key(p.states[-1])
                if fk in seen_final:
                    continue
                seen_final.add(fk)
                kept.append(p)
            found = kept

        return found

    # ---------------------------------------------------------------------
    # Reverse search (DFS/BFS)
    # ---------------------------------------------------------------------
    def find_reverse(
        self,
        start: Counter[str],
        goal: Counter[str],
        *,
        min_overlap: int = 1,
        allow_reuse: bool = False,
        strategy: str = "dfs",
        stop_on_start: bool = True,
        must_include_first_reaction: bool = True,
        max_depth: Optional[int] = None,
        max_pathways: int = 1000,
        keep_shortest_only: bool = False,
    ) -> List[Pathway]:
        """
        Enumerate backward pathways from ``goal`` down to ``start`` (returned forward-ordered).
        """
        if max_depth is None:
            max_depth = max(8, len(self.net.reactions) * (2 if allow_reuse else 1))

        first_rid = min(self.net.reactions.keys()) if self.net.reactions else None
        found: List[Pathway] = []

        if strategy not in {"dfs", "bfs"}:
            raise ValueError('strategy must be "dfs" or "bfs"')

        fringe: Deque = deque()
        pop = fringe.pop if strategy == "dfs" else fringe.popleft
        push = fringe.append

        push((goal.copy(), [], [goal.copy()], set()))
        visited_by_depth: Dict[int, Set[Tuple[Tuple[str, int], ...]]] = {}

        while fringe:
            state, rids_back, hist_back, used = pop()
            depth = len(rids_back)
            if depth > max_depth:
                continue

            key = counter_key(state)
            if key in visited_by_depth.setdefault(depth, set()):
                continue
            visited_by_depth[depth].add(key)

            if all(state.get(m, 0) >= c for m, c in start.items()):
                forward_rids = list(reversed(rids_back))
                forward_states = list(reversed(hist_back))
                if (not must_include_first_reaction) or (
                    first_rid in forward_rids if first_rid is not None else True
                ):
                    found.append(
                        Pathway(reaction_ids=forward_rids, states=forward_states)
                    )
                    if len(found) >= max_pathways:
                        break
                if stop_on_start:
                    continue

            for rid, rxn in self.net.reactions.items():
                if (not allow_reuse) and (rid in used):
                    continue
                ok, matched = self._can_fire_backward(rxn, state, min_overlap)
                if not ok:
                    continue
                new_state = self._apply_backward(state, matched, rxn.reactants_can)
                new_rids_back = rids_back + [rid]
                new_hist_back = hist_back + [new_state.copy()]
                new_used = set(used)
                if not allow_reuse:
                    new_used.add(rid)
                push((new_state, new_rids_back, new_hist_back, new_used))

        if keep_shortest_only and found:
            found.sort(key=lambda p: len(p.reaction_ids))
            kept: List[Pathway] = []
            seen_final: Set[Tuple[Tuple[str, int], ...]] = set()
            for p in found:
                fk = counter_key(p.states[-1])
                if fk in seen_final:
                    continue
                seen_final.add(fk)
                kept.append(p)
            found = kept

        return found

    # ---------------------------------------------------------------------
    # Bidirectional BFS with relaxed meeting (subset/superset)
    # ---------------------------------------------------------------------
    def find_bidirectional(
        self,
        start: Counter[str],
        goal: Counter[str],
        *,
        min_overlap: int = 1,
        allow_reuse: bool = False,
        max_depth_each: Optional[int] = None,
        max_pathways: int = 1000,
        max_entries_per_key: int = 3,
    ) -> List[Pathway]:
        """
        Meet-in-the-middle BFS where forward and backward frontiers "meet"
        when either multiset contains the other (⊇ or ⊆).
        """
        if max_depth_each is None:
            max_depth_each = max(6, len(self.net.reactions))

        def key_to_counter(key: Tuple[Tuple[str, int], ...]) -> Counter[str]:
            return Counter(dict(key))

        def add_entry(
            front: Dict[
                Tuple[Tuple[str, int], ...],
                List[Tuple[List[int], List[Counter[str]], Set[int]]],
            ],
            key: Tuple[Tuple[str, int], ...],
            rids: List[int],
            history: List[Counter[str]],
            used: Set[int],
        ) -> None:
            lst = front.setdefault(key, [])
            lst.append((rids, history, used))
            lst.sort(key=lambda t: len(t[0]))
            if len(lst) > max_entries_per_key:
                del lst[max_entries_per_key:]

        f_front: Dict[
            Tuple[Tuple[str, int], ...],
            List[Tuple[List[int], List[Counter[str]], Set[int]]],
        ] = {}
        b_front: Dict[
            Tuple[Tuple[str, int], ...],
            List[Tuple[List[int], List[Counter[str]], Set[int]]],
        ] = {}

        f_key0 = counter_key(start)
        b_key0 = counter_key(goal)
        add_entry(f_front, f_key0, [], [start.copy()], set())
        add_entry(b_front, b_key0, [], [goal.copy()], set())

        found_paths: List[Pathway] = []
        depth = 0

        while depth < max_depth_each and len(found_paths) < max_pathways:
            depth += 1

            # expand forward
            new_f: Dict[
                Tuple[Tuple[str, int], ...],
                List[Tuple[List[int], List[Counter[str]], Set[int]]],
            ] = {}
            for _, entries in list(f_front.items()):
                for f_rids, f_hist, f_used in entries:
                    s = f_hist[-1]
                    for rid, rxn in self.net.reactions.items():
                        if (not allow_reuse) and (rid in f_used):
                            continue
                        ok, matched = self._can_fire_forward(rxn, s, min_overlap)
                        if not ok:
                            continue
                        ns = self._apply_forward(s, matched, rxn.products_can)
                        nk = counter_key(ns)
                        nr = f_rids + [rid]
                        nh = f_hist + [ns.copy()]
                        nu = set(f_used)
                        if not allow_reuse:
                            nu.add(rid)
                        add_entry(new_f, nk, nr, nh, nu)
            for k, v in new_f.items():
                f_front.setdefault(k, []).extend(v)

            # expand backward
            new_b: Dict[
                Tuple[Tuple[str, int], ...],
                List[Tuple[List[int], List[Counter[str]], Set[int]]],
            ] = {}
            for _, entries in list(b_front.items()):
                for b_rids, b_hist, b_used in entries:
                    s = b_hist[-1]
                    for rid, rxn in self.net.reactions.items():
                        if (not allow_reuse) and (rid in b_used):
                            continue
                        ok, matched = self._can_fire_backward(rxn, s, min_overlap)
                        if not ok:
                            continue
                        ns = self._apply_backward(s, matched, rxn.reactants_can)
                        nk = counter_key(ns)
                        nr = b_rids + [rid]
                        nh = b_hist + [ns.copy()]
                        nu = set(b_used)
                        if not allow_reuse:
                            nu.add(rid)
                        add_entry(new_b, nk, nr, nh, nu)
            for k, v in new_b.items():
                b_front.setdefault(k, []).extend(v)

            # relaxed meeting: one contains the other
            f_items = list(f_front.items())
            b_items = list(b_front.items())
            outer, inner = (
                (f_items, b_items)
                if len(f_items) <= len(b_items)
                else (b_items, f_items)
            )
            for o_key, o_entries in outer:
                o_cnt = key_to_counter(o_key)
                for i_key, i_entries in inner:
                    i_cnt = key_to_counter(i_key)
                    if not (o_cnt >= i_cnt or i_cnt >= o_cnt):
                        continue
                    if outer is f_items:
                        f_entries, b_entries = o_entries, i_entries
                    else:
                        f_entries, b_entries = i_entries, o_entries
                    for f_rids, f_hist, _ in f_entries:
                        for b_rids, b_hist, _ in b_entries:
                            if (not allow_reuse) and (set(f_rids) & set(b_rids)):
                                continue
                            forward_rids = f_rids + list(reversed(b_rids))
                            forward_states = f_hist + list(reversed(b_hist))[1:]
                            found_paths.append(
                                Pathway(
                                    reaction_ids=forward_rids, states=forward_states
                                )
                            )
                            if len(found_paths) >= max_pathways:
                                return found_paths

        return found_paths

    # ---------------------------------------------------------------------
    # A* (fewest steps)
    # ---------------------------------------------------------------------
    def find_a_star(
        self,
        start: Counter[str],
        goal: Counter[str],
        *,
        min_overlap: int = 1,
        allow_reuse: bool = False,
        stop_on_goal: bool = True,
        max_nodes: Optional[int] = 200000,
        max_pathways: int = 1,
    ) -> List[Pathway]:
        """
        A* search for a minimal-step pathway from ``start`` to ``goal``.
        """
        max_goal_prod = 0
        for rxn in self.net.reactions.values():
            inter = rxn.products_can & goal
            max_goal_prod = max(max_goal_prod, sum(inter.values()))

        def heuristic(state: Counter[str]) -> int:
            missing = sum((goal - state).values())
            if missing <= 0:
                return 0
            denom = max(1, max_goal_prod)
            return math.ceil(missing / denom)

        start_key = counter_key(start)
        h0 = heuristic(start)
        # elements: (f, g, tie, key, state, rids, history, used)
        open_heap: List[
            Tuple[
                int,
                int,
                int,
                Tuple[Tuple[str, int], ...],
                Counter[str],
                List[int],
                List[Counter[str]],
                Set[int],
            ]
        ] = []
        tie = 0
        heapq.heappush(
            open_heap, (h0, 0, tie, start_key, start.copy(), [], [start.copy()], set())
        )
        tie += 1

        best_g: Dict[Tuple[Tuple[str, int], ...], int] = {start_key: 0}
        found: List[Pathway] = []
        explored = 0

        while open_heap:
            f, g, _, key, state, rids, history, used = heapq.heappop(open_heap)
            explored += 1
            if max_nodes is not None and explored > max_nodes:
                break
            if best_g.get(key, float("inf")) < g:
                continue

            if sum((goal - state).values()) == 0:
                found.append(Pathway(reaction_ids=list(rids), states=list(history)))
                if len(found) >= max_pathways:
                    return found
                if stop_on_goal:
                    continue

            for rid, rxn in self.net.reactions.items():
                if (not allow_reuse) and (rid in used):
                    continue
                ok, matched = self._can_fire_forward(rxn, state, min_overlap)
                if not ok:
                    continue
                ns = self._apply_forward(state, matched, rxn.products_can)
                nk = counter_key(ns)
                ng = g + 1
                if ng < best_g.get(nk, float("inf")):
                    best_g[nk] = ng
                    nh = heuristic(ns)
                    nf = ng + nh
                    nr = rids + [rid]
                    nhist = history + [ns.copy()]
                    nused = set(used)
                    if not allow_reuse:
                        nused.add(rid)
                    heapq.heappush(
                        open_heap, (nf, ng, tie, nk, ns.copy(), nr, nhist, nused)
                    )
                    tie += 1

        return found
