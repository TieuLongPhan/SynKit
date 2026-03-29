from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple

from synkit.Synthesis.Reactor.syn_reactor import SynReactor


def apply_rule_worker(
    args: Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]],
) -> Tuple[int, Tuple[str, ...], List[str]]:
    """
    Apply one rule to one substrate mixture.

    This intentionally preserves the behavior of the validated monolith:
    - automorphism=True
    - order-preserving deduplication of raw output strings
    """
    idx, rule, substrate, explicit_h, implicit_temp, strategy, reactant_keys = args

    kwargs = dict(
        smiles=substrate,
        template=rule,
        invert=False,
        explicit_h=explicit_h,
        implicit_temp=implicit_temp,
        automorphism=True,
    )
    if strategy is not None:
        kwargs["strategy"] = strategy

    reactor = SynReactor.from_smiles(**kwargs)

    out: List[str] = []
    seen: Set[str] = set()
    for s in reactor.smiles_list:
        val = (s or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)

    return idx, reactant_keys, out
