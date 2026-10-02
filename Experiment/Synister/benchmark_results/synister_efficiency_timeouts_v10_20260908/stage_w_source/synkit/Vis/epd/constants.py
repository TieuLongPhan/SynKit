from __future__ import annotations

"""Visual constants for mechanism plotting."""

from typing import Dict

ACS_ATOM_COLORS: Dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#303030",
    "N": "#3B5BDB",
    "O": "#D94841",
    "F": "#2B8A3E",
    "Cl": "#2B8A3E",
    "Br": "#8C4A2F",
    "I": "#6741D9",
    "P": "#C77D00",
    "S": "#C9A227",
    "B": "#B3541E",
    "Si": "#6C757D",
    "Na": "#4DABF7",
    "K": "#845EF7",
    "Li": "#9775FA",
    "Mg": "#37B24D",
    "Ca": "#51CF66",
}

DEFAULT_ATOM_COLOR = "#ADB5BD"

BOND_SYMBOLS = {
    None: "∅",
    0: "∅",
    1: "—",
    2: "=",
    3: "≡",
    1.5: ":",
}

KIND_FAMILY: Dict[str, str] = {
    "LP-/B+": "LP-/B+",
    "B-/LP+": "B-/LP+",
    "B-/B+": "B-/B+",
    "LP-/H+": "LP-/H+",
    "H-/LP+": "H-/LP+",
    "H-/B+": "H-/B+",
    "LP-/Sigma+": "LP-/B+",
    "LP-/Pi+": "LP-/B+",
    "Sigma-/LP+": "B-/LP+",
    "Pi-/LP+": "B-/LP+",
    "Sigma-/Sigma+": "B-/B+",
    "Sigma-/Pi+": "B-/B+",
    "Pi-/Sigma+": "B-/B+",
    "Pi-/Pi+": "B-/B+",
    "Sigma-/H+": "B-/H+",
    "Pi-/H+": "B-/H+",
    "H-/Sigma+": "H-/B+",
    "H-/Pi+": "H-/B+",
}


def transition_family(kind: str) -> str:
    """Return the generic rendering family for a typed EPD action."""
    return KIND_FAMILY.get(kind, kind)


class NatureStyle:
    """Default restrained styling for two-panel mechanism figures."""

    broken_color = "#C92A2A"
    forming_color = "#2B8A3E"
    shift_color = "#2563EB"
    proton_color = "#B7791F"
    arrow_color = "#111111"
    neutral_bond_color = "#2A2A2A"
    faded_bond_color = "#8A8F98"
    bond_label_color = "#4A4F57"
    map_text_color = "#5C6770"
    charge_badge_color = "#6D1F1F"
    electron_badge_color = "#1F3A5F"
    panel_title_color = "#111111"
    background_color = "white"

    main_title_fontsize = 18
    panel_title_fontsize = 15
