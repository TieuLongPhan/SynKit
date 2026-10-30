"""Reaction-SMILES fields without interpreting endpoint chemistry."""

from __future__ import annotations

import re

_REACTION_SEPARATOR = re.compile(r"(?<!-)>")


def split_reaction_fields(
    text: str, *, error_message: str = "Expected reactants>agents>products"
) -> tuple[str, str, str]:
    """Return exactly three fields, preserving dative ``->`` bond text.

    A right angle bracket belongs to a dative bond when preceded by ``-``.
    All other right angle brackets delimit reaction fields. Empty fields are
    retained; each caller decides whether empty endpoints or agents are valid.
    This function does not parse, normalize or rewrite any SMILES field.
    """
    parts = _REACTION_SEPARATOR.split(text)
    if len(parts) != 3:
        raise ValueError(error_message)
    return parts[0], parts[1], parts[2]


__all__ = ["split_reaction_fields"]
