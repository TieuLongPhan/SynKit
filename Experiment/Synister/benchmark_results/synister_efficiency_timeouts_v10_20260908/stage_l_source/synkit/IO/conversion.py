"""High-level conversion entry points.

This module provides I/O-facing access to conversions whose implementation
belongs to a domain-specific SynKit package.
"""

from __future__ import annotations

from typing import Any, Optional


def ef_smirks_to_epd(
    ef_smirks: str,
    orbital_class: Optional[str] = None,
    strict_bond_lookup: bool = True,
) -> dict[str, Any]:
    """Complete AAM and convert EF-SMIRKS text into generic and typed EPD.

    See :func:`synkit.Graph.Mech.conversion.ef_smirks_to_epd` for the
    EF-SMIRKS input format and returned fields.
    """
    from synkit.Graph.Mech.conversion import ef_smirks_to_epd as _convert

    return _convert(
        ef_smirks,
        orbital_class=orbital_class,
        strict_bond_lookup=strict_bond_lookup,
    )


def epd_to_ef_smirks(complete_aam: str, epd: list[list[Any]]) -> str:
    """Convert complete atom-mapped RSMI and EPD records into EF-SMIRKS.

    See :func:`synkit.Graph.Mech.conversion.epd_to_ef_smirks` for accepted
    generic and typed EPD records.
    """
    from synkit.Graph.Mech.conversion import epd_to_ef_smirks as _convert

    return _convert(complete_aam, epd)


__all__ = ["ef_smirks_to_epd", "epd_to_ef_smirks"]
