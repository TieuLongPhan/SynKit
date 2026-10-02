from __future__ import annotations

from typing import Any, Optional


def count_lhs_components(text: str) -> Optional[int]:
    if not text:
        return None
    lhs = text.split(">>", 1)[0].strip() if ">>" in text else text.strip()
    parts = [p for p in lhs.split(".") if p.strip()]
    return len(parts) if parts else None


def infer_rule_arity(rule: Any) -> int:
    ar: Optional[int] = None
    if isinstance(rule, str):
        ar = count_lhs_components(rule)
    else:
        for attr in ("smarts", "smirks", "template"):
            if not hasattr(rule, attr):
                continue
            try:
                ar = count_lhs_components(str(getattr(rule, attr)))
                if ar is not None:
                    break
            except Exception:
                continue
        if ar is None:
            try:
                ar = count_lhs_components(repr(rule))
            except Exception:
                ar = None

    if ar is None or ar < 1:
        ar = 2
    return int(ar)
