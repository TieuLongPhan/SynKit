from __future__ import annotations

from typing import Optional, Tuple


def make_dedup_key(
    *,
    dedup_across_rules: bool,
    rule_index: int,
    r_keep_keys: Tuple[str, ...],
    p_keep_keys: Tuple[str, ...],
) -> Tuple[Optional[int], Tuple[str, ...], Tuple[str, ...]]:
    ridx: Optional[int] = None if dedup_across_rules else int(rule_index)
    return (ridx, r_keep_keys, p_keep_keys)
