from __future__ import annotations

from collections import Counter
from typing import Any, List, Set, Tuple


def replay_pathway_and_collect_inferred(
    net: Any, pathway: Any, *, start: Counter | None = None
) -> Counter:
    """
    Replay ``pathway`` from ``start`` (defaults to seeding one copy of each declared
    Source.* reaction in the network) and return a Counter of species that had to
    be inferred (seeded) to satisfy full stoichiometry at each step.

    :param net: ReactionNetwork instance.
    :param pathway: Pathway-like object with .reaction_ids attribute.
    :param start: Optional starting Counter (if None seeds Source.* tokens once).
    :returns: Counter of inferred species -> counts.
    """
    if start is None:
        start = Counter()
        for rx in net.reactions.values():
            raw = getattr(rx, "original_raw", None)
            if isinstance(raw, str) and raw.startswith("Source."):
                token = raw.split(">>", 1)[0]
                start[token] += 1

    state = start.copy()
    inferred = Counter()
    for rid in pathway.reaction_ids:
        rx = net.reactions[rid]
        # compute missing required reactants
        missing = Counter(
            {
                k: v - state.get(k, 0)
                for k, v in rx.reactants_can.items()
                if v > state.get(k, 0)
            }
        )
        if missing:
            for k, v in missing.items():
                state[k] = state.get(k, 0) + v
                inferred[k] += v
        matched = rx.reactants_can.copy()
        state = rx.apply_forward(state, matched)
    return inferred


def dedupe_pathways_by_canonical(net: Any, pathways: List[Any]) -> List[Any]:
    """
    Deduplicate Pathway objects by canonical reaction sequence. Keeps first-seen.

    :param net: ReactionNetwork instance.
    :param pathways: list of Pathway objects.
    :returns: list of unique Pathway objects.
    """
    seen: Set[Tuple[str, ...]] = set()
    uniq: List[Any] = []
    for p in pathways:
        seq = tuple(
            net.reactions[r].canonical_raw or net.reactions[r].original_raw
            for r in p.reaction_ids
        )
        if seq in seen:
            continue
        seen.add(seq)
        uniq.append(p)
    return uniq


def pretty_print_pathway(net: Any, p: Any, *, show_original: bool = True) -> None:
    """
    Print a pathway to stdout: reaction lines and state progression.

    :param net: ReactionNetwork instance.
    :param p: Pathway object with .reaction_ids and .states.
    :param show_original: If True, print rx.original_raw; otherwise use canonical_raw.
    """
    for rid in p.reaction_ids:
        rx = net.reactions[rid]
        line = (
            rx.original_raw if show_original else (rx.canonical_raw or rx.original_raw)
        )
        print("  ", line)
    print("  states:")
    for i, st in enumerate(p.states):
        print("   ", f"state[{i}]:", dict(st))
