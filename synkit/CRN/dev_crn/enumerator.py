from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, List, Optional, Set

from synkit.dev_crn import ReactionNetwork
from synkit.dev_crn.explorer import Pathway  # reuse the Pathway container if present
from .exceptions import NoPathwaysError
from .helpers import (
    dedupe_pathways_by_canonical,
    pretty_print_pathway,
    replay_pathway_and_collect_inferred,
)


class MotifEnumerator:
    """
    Enumerate CRN motif pathways.

    Behaviour (matches the debug cell semantics you validated):
      - Build start state from config["sources"] (honours 'initial').
      - Convert all `Source.X` tokens into `X` counts **up-front** so stoichiometry is natural.
      - Forbid re-activation of the Source reaction and record one logical Source activation
        as a prefix for display.
      - Perform a deterministic search (BFS by canonical reaction order) that:
        * enforces stoichiometry (or uses infer_missing if requested),
        * forbids reusing the same reaction id within a single path (used set),
        * distinguishes different ordered reaction histories (seen key includes path tuple).
    """

    def __init__(self, *, max_depth: int = 50, max_pathways: int = 2000) -> None:
        """
        :param max_depth: max reaction steps in a path
        :param max_pathways: cap on collected pathways
        """
        self.max_depth = int(max_depth)
        self.max_pathways = int(max_pathways)

    def __repr__(self) -> str:
        return f"MotifEnumerator(max_depth={self.max_depth}, max_pathways={self.max_pathways})"

    # ---------------------- small helpers ----------------------
    def _build_start(
        self, reactions: List[str], config: Optional[Dict[str, Any]]
    ) -> Counter:
        """
        Construct the start Counter honoring config["sources"] if present.
        """
        start = Counter()
        if config and isinstance(config.get("sources"), dict) and config["sources"]:
            for token, spec in config["sources"].items():
                try:
                    if isinstance(spec, dict):
                        typ = str(spec.get("type", "limited")).lower()
                        if typ in {"unlimited", "infinite"}:
                            start[token] = int(self.max_depth)
                        else:
                            start[token] = int(spec.get("initial", 1))
                    else:
                        start[token] = 1
                except Exception:
                    start[token] = 1
        else:
            for r in reactions:
                if isinstance(r, str) and r.startswith("Source."):
                    start[r.split(">>", 1)[0]] += 1
        return start

    # ---------------------- main API ----------------------
    def enumerate_motif(
        self,
        name: str,
        reactions: List[str],
        config: Optional[Dict[str, Any]] = None,
        *,
        infer_missing: bool = False,
        show_n: int = 3,
    ) -> Dict[str, Any]:
        """
        Enumerate forward pathways for a motif.

        :param name: motif name for printing
        :param reactions: list of raw reaction strings
        :param config: motif config (may contain 'sources' and 'sinks')
        :param infer_missing: if True, allow guarded inference of missing co-reactants
        :param show_n: how many examples to pretty-print
        :returns: summary dict with keys name, n_raw, n_unique, examples, net, paths
        :raises: NoPathwaysError if no pathways are found
        """
        print(f"\n=== Enumerating {name} (len={len(reactions)}) ===")

        # Build network
        net = ReactionNetwork.from_raw_list(
            reactions, standardizer=None, remove_aam=True
        )

        # Build start counter
        start = self._build_start(reactions, config)

        # Detect Source.* reactions: map Source.Token -> rid and rid -> product
        source_token_to_rid: Dict[str, int] = {}
        source_rid_to_product: Dict[int, str] = {}
        for rid, rx in net.reactions.items():
            lhs = getattr(rx, "reactants_can", {})
            rhs = getattr(rx, "products_can", {})
            if len(lhs) == 1 and len(rhs) == 1:
                ((lk, lv),) = lhs.items()
                ((rk, rv),) = rhs.items()
                if (
                    isinstance(lk, str)
                    and lk.startswith("Source.")
                    and lv == 1
                    and rv == 1
                ):
                    source_token_to_rid[lk] = rid
                    source_rid_to_product[rid] = rk

        # PRE-CONVERT ALL Source tokens -> product counts (so P:50 is available stoichiometrically)
        start_after = start.copy()
        disallow_rids: Set[int] = set()
        prefix_rids: List[int] = []
        prefix_states: List[Counter] = [start_after.copy()]

        for source_token, count in list(start.items()):
            if count <= 0:
                continue
            if source_token in source_token_to_rid:
                rid = source_token_to_rid[source_token]
                prod = source_rid_to_product.get(rid)
                if prod is None:
                    continue
                # convert all Source token counts into product counts
                start_after[prod] = start_after.get(prod, 0) + count
                start_after[source_token] = 0
                # record one logical activation for display and forbid re-activation
                prefix_rids.append(rid)
                disallow_rids.add(rid)
                prefix_states.append(start_after.copy())

        # Sinks/default goal
        sink_set = {"Removed"}
        if config and isinstance(config.get("sinks"), dict) and config["sinks"]:
            sink_set = set(config["sinks"].keys())
        # goal = Counter({t: 1 for t in sink_set})

        # -------------------------
        # Deterministic BFS enumerator (fringe), matching the debug cell behaviour
        # -------------------------
        MAX_DEPTH = self.max_depth
        MAX_FOUND = self.max_pathways

        def rx_fireable_and_matched(rx, state: Counter) -> Optional[Counter]:
            """
            Return a matched Counter if rx can fire under strict stoichiometry
            (or using infer_missing semantics), otherwise None.
            """
            need = rx.reactants_can
            # strict check
            if all(state.get(k, 0) >= v for k, v in need.items()):
                return need.copy()
            # infer_missing guarded behavior
            if infer_missing and any(state.get(k, 0) > 0 for k in need.keys()):
                tmp = state.copy()
                for k, v in need.items():
                    if tmp.get(k, 0) < v:
                        tmp[k] = tmp.get(k, 0) + (v - tmp.get(k, 0))
                return need.copy()
            return None

        # canonical reaction order
        candidate_rids = [rid for rid in sorted(net.reactions.keys())]

        fringe = deque()
        # each fringe element: (state Counter, path [rids], used set of rids)
        fringe.append((start_after.copy(), [], set(prefix_rids)))
        seen = set()
        found_paths: List[Pathway] = []

        while fringe and len(found_paths) < MAX_FOUND:
            state, path, used = fringe.popleft()

            # goal check
            if any(state.get(s, 0) > 0 for s in sink_set):
                found_paths.append(
                    Pathway(list(path), [None] * (len(path) + 1))
                )  # states will be fixed later if needed
                if len(found_paths) >= MAX_FOUND:
                    break
                continue

            if len(path) >= MAX_DEPTH:
                continue

            progressed = False
            for rid in candidate_rids:
                if rid in disallow_rids:
                    continue
                if rid in used:
                    continue
                rx = net.reactions[rid]
                matched = rx_fireable_and_matched(rx, state)
                if matched is None:
                    continue

                progressed = True
                # apply the reaction to get newstate
                newstate = rx.apply_forward(state.copy(), matched)
                newpath = path + [rid]
                newused = set(used)
                newused.add(rid)
                # key must include ordered history so different orders are preserved
                key = (tuple(sorted(newstate.items())), tuple(newpath))
                if key in seen:
                    continue
                seen.add(key)
                fringe.append((newstate, newpath, newused))

            if not progressed:
                # dead end
                continue

        # Build Pathway objects with actual states by replaying them (so states are accurate)
        raw_pathways: List[Pathway] = []
        for path in found_paths:
            # path.reaction_ids was populated earlier as list(path) - but we used placeholder states
            # reconstruct states by replaying from start_after
            seq = path.reaction_ids
            if seq is None:
                seq = []
            s = start_after.copy()
            states = [s.copy()]
            for rid in seq:
                rx = net.reactions[rid]
                s = rx.apply_forward(s, rx.reactants_can.copy())
                states.append(s.copy())
            raw_pathways.append(Pathway(seq, states))

        # Prepend prefix_rids/state if any (so the displayed sequence includes logical Source activation)
        if prefix_rids:
            prefixed = []
            for p in raw_pathways:
                combined_states = (
                    list(prefix_states) + list(p.states[1:])
                    if p.states
                    else list(prefix_states)
                )
                combined_rids = list(prefix_rids) + list(p.reaction_ids)
                prefixed.append(Pathway(combined_rids, combined_states))
            raw_pathways = prefixed

        unique_paths = dedupe_pathways_by_canonical(net, raw_pathways)

        if not unique_paths:
            msg = (
                f"No pathways found for motif '{name}' (raw_paths={len(raw_pathways)}). "
                "Hint: check for unproduced reactants or missing sources."
            )
            raise NoPathwaysError(msg)

        print(
            f"  total raw: {len(raw_pathways)}   unique canonical: {len(unique_paths)}"
        )

        # Pretty print examples and show inferred inflows if any
        for i, p in enumerate(unique_paths[:show_n], 1):
            print(f"\n  Pathway #{i}:")
            pretty_print_pathway(net, p, show_original=True)
            inferred = replay_pathway_and_collect_inferred(net, p, start=start)
            if inferred:
                print("   inferred inflows:", dict(inferred))
            else:
                print("   inferred inflows: (none)")

        seqs = [
            " | ".join(
                net.reactions[r].canonical_raw or net.reactions[r].original_raw
                for r in p.reaction_ids
            )
            for p in unique_paths
        ]

        return {
            "name": name,
            "n_raw": len(raw_pathways),
            "n_unique": len(unique_paths),
            "examples": seqs[:show_n],
            "net": net,
            "paths": unique_paths,
        }
