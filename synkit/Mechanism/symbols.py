"""Canonical electron-locus symbols and external-input aliases.

External formats may use words, ASCII abbreviations, or Unicode orbital
symbols.  SynKit normalizes them at the adapter boundary and stores only the
compact internal alphabet ``lp``, ``σ``, ``π``, and ``∙``.
"""

from __future__ import annotations

from typing import Final, Literal

LONE_PAIR: Final = "lp"
SIGMA: Final = "σ"
PI: Final = "π"
RADICAL: Final = "∙"

CanonicalLocus = Literal["lp", "σ", "π", "∙"]

_ALIASES: dict[str, CanonicalLocus] = {
    "lp": LONE_PAIR,
    "lonepair": LONE_PAIR,
    "lone_pair": LONE_PAIR,
    "sigma": SIGMA,
    SIGMA: SIGMA,
    "pi": PI,
    PI: PI,
    "rad": RADICAL,
    "radical": RADICAL,
    RADICAL: RADICAL,
    "·": RADICAL,
    "•": RADICAL,
    ".": RADICAL,
}

_ASCII: dict[CanonicalLocus, str] = {
    LONE_PAIR: "lp",
    SIGMA: "sigma",
    PI: "pi",
    RADICAL: "rad",
}

_LEGACY_EPD: dict[CanonicalLocus, str] = {
    LONE_PAIR: "LP",
    SIGMA: "Sigma",
    PI: "Pi",
    RADICAL: "Rad",
}


def normalize_locus_symbol(value: str) -> CanonicalLocus:
    """Normalize an external locus name to SynKit's internal alphabet.

    Accepted primary aliases are ``LP``/``lp``, ``sigma``/``σ``,
    ``pi``/``π``, and ``rad``/``∙``. Common radical-dot variants are accepted
    because datasets do not consistently use the same Unicode code point.
    """
    if not isinstance(value, str):
        raise TypeError(f"Electron locus must be text, got {type(value).__name__}.")
    token = value.strip()
    normalized = _ALIASES.get(token) or _ALIASES.get(token.casefold())
    if normalized is None:
        accepted = "lp/LP, sigma/σ, pi/π, rad/∙"
        raise ValueError(f"Unsupported electron locus {value!r}; expected {accepted}.")
    return normalized


def external_locus_symbol(
    value: str, *, style: Literal["unicode", "ascii", "legacy_epd"] = "unicode"
) -> str:
    """Render a locus using a requested external notation."""
    canonical = normalize_locus_symbol(value)
    if style == "unicode":
        return canonical
    if style == "ascii":
        return _ASCII[canonical]
    if style == "legacy_epd":
        return _LEGACY_EPD[canonical]
    raise ValueError(f"Unsupported locus rendering style: {style!r}")


def internal_action_label(source: str, target: str) -> str:
    """Return a compact internal move label such as ``lp-/σ+``."""
    return f"{normalize_locus_symbol(source)}-/{normalize_locus_symbol(target)}+"


def legacy_action_label(source: str, target: str) -> str:
    """Return a legacy typed EPD label such as ``LP-/Sigma+``."""
    return (
        f"{external_locus_symbol(source, style='legacy_epd')}-/"
        f"{external_locus_symbol(target, style='legacy_epd')}+"
    )
