#!/usr/bin/env python3
"""Render publication figures directly from frozen Synister evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from summarize_synister_evidence import summarize, _verify_embedded_digest

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = (
    REPOSITORY
    / "paper"
    / "synister"
    / "evidence"
    / "pilot100_v4"
    / "derived_findings.json"
)
DEFAULT_CASE = (
    REPOSITORY
    / "paper"
    / "synister"
    / "evidence"
    / "alternative_its_case_v1"
    / "record.json"
)
DEFAULT_OUTPUT = REPOSITORY / "paper" / "synister" / "figures" / "pilot_spectrum.pdf"

# Local, portable adaptation of the thesis style; TeX reads generated aliases.
TOKENS_PATH = REPOSITORY / "paper/synister/figures/style_tokens.json"
TOKENS = json.loads(TOKENS_PATH.read_text())
COLORS = TOKENS["colors"]
BLUE, GREEN, ORANGE = (COLORS[key] for key in ("blue", "teal", "gold"))
VERMILLION, INK, MUTED = (COLORS[key] for key in ("vermilion", "text", "muted"))
HAIR, WASH, NAVY = (COLORS[key] for key in ("rule", "panel", "navy"))


def write_palette() -> None:
    """Generate the TeX palette from the same tokens as the Python figures."""
    aliases = dict(
        npgBlue="blue",
        npgGreen="teal",
        npgOrange="gold",
        npgVermillion="vermilion",
        npgPurple="purple",
        npgSky="sky",
        figInk="text",
        figMuted="muted",
        figHair="rule",
        figWash="panel",
        figNavy="navy",
        figBroken="broken",
    )
    lines = ["% Generated from style_tokens.json; do not edit."]
    lines.extend(
        rf"\definecolor{{{name}}}{{HTML}}{{{COLORS[key][1:]}}}"
        for name, key in aliases.items()
    )
    TOKENS_PATH.with_name("palette.tex").write_text("\n".join(lines) + "\n")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _panel_badge(axis: mpl.axes.Axes, letter: str, title: str) -> None:
    axis.text(
        -0.075,
        1.075,
        letter,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
        fontfamily="DejaVu Serif",
        bbox={"boxstyle": "circle,pad=0.22", "fc": NAVY, "ec": "none"},
        clip_on=False,
    )
    axis.text(
        -0.025,
        1.075,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        color=INK,
        fontsize=8.5,
        fontweight="bold",
        fontfamily="DejaVu Serif",
        clip_on=False,
    )


def _cohort_panel(axis: mpl.axes.Axes, pilot: dict[str, Any]) -> None:
    """Paired bars expose the numerator and keep each denominator explicit."""
    modes = pilot["modes"]
    minimum, reference = modes["minimal"], modes["reference_cd"]
    rows = [
        ("Closed shell", "cases", 100),
        ("Multiple ITS", "multiple_exact_its_classes", None),
        ("Varying center", "unstable_reaction_centres", None),
    ]
    for i, (label, field, denominator) in enumerate(rows):
        y = 6.6 - 2.7 * i
        axis.text(0, y + 0.48, label, fontsize=8.2, color=INK)
        for j, (mode, color) in enumerate(((minimum, BLUE), (reference, ORANGE))):
            d = denominator or mode["cases"]
            n = mode[field]
            yy = y - 0.40 * j
            axis.barh(yy, 100, height=0.25, color=WASH, edgecolor="none")
            axis.barh(yy, 100 * n / d, height=0.25, color=color, edgecolor="none")
            axis.text(103, yy, f"{n}/{d}", fontsize=7.6, va="center", color=color)
    axis.set(
        xlim=(0, 128),
        ylim=(0.1, 8.3),
        yticks=[],
        xticks=[0, 50, 100],
        xlabel="Reactions (%)",
    )
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_bounds(0, 100)
    axis.tick_params(axis="y", length=0)
    for color, label in ((BLUE, "Minimum"), (ORANGE, "Reference CD")):
        axis.plot([], [], color=color, lw=4, label=label)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),
        ncol=2,
        frameon=False,
        fontsize=7.6,
        handlelength=1.0,
        columnspacing=1.4,
    )
    _panel_badge(axis, "A", "100-reaction pilot")


def _exact_ladder(case: dict[str, Any]) -> tuple[list[float], list[int], list[int]]:
    by_distance: dict[float, set[tuple[int, int, tuple[str, ...]]]] = {}
    for query in case["queries"]:
        shell = query["result"]["shell"]
        if not shell["complete"]:
            raise ValueError("the publication ladder requires complete exact shells")
        target = query["requested_target"]
        if target == "minimal":
            distance = float(shell["minimum_cost"])
        elif target == "reference":
            distance = float(shell["reference_cd"])
        else:
            distance = float(target)
        counts = (
            int(shell["shell_labeled_mapping_count"]),
            int(shell["shell_its_class_count"]),
            tuple(sorted(item["its_class_id"] for item in shell["alternatives"])),
        )
        by_distance.setdefault(distance, set()).add(counts)
    inconsistent = {key: value for key, value in by_distance.items() if len(value) != 1}
    if inconsistent:
        raise ValueError(f"seed modes disagree on exact shell results: {inconsistent}")
    distances = sorted(by_distance)
    counts = [next(iter(by_distance[distance])) for distance in distances]
    return distances, [value[0] for value in counts], [value[1] for value in counts]


def _ladder_panel(axis: mpl.axes.Axes, case: dict[str, Any]) -> None:
    """Separate numeric shells, with no interpolation between queried targets."""
    distance, mappings, classes = _exact_ladder(case)
    axis.axvspan(5.55, 6.45, color=GREEN, alpha=0.07, lw=0)
    axis.axvspan(7.55, 8.45, color=VERMILLION, alpha=0.05, lw=0)
    for dx, values, color, label in (
        (-0.22, mappings, BLUE, "Maps"),
        (0.22, classes, GREEN, "ITS"),
    ):
        xs = np.asarray(distance) + dx
        axis.vlines(xs, 0, values, color=color, lw=2.5, zorder=3)
        axis.scatter(xs, values, color=color, s=18, zorder=4, label=label)
        for x, value in zip(xs, values):
            if value:
                axis.annotate(
                    str(value),
                    (x, value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=color,
                )
    axis.text(4, 0.12, "0", ha="center", fontsize=8, color=MUTED)
    axis.set_yscale("symlog", linthresh=1, linscale=0.5)
    axis.set(
        xlim=(3.4, 12.7),
        ylim=(-0.13, 650),
        xticks=distance,
        xlabel="Chemical distance",
        ylabel="Count (symlog)",
    )
    axis.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    axis.grid(axis="y", color=HAIR, lw=0.5)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),
        ncol=2,
        frameon=False,
        fontsize=7.6,
        handletextpad=0.2,
        columnspacing=1.0,
    )
    axis.text(6, 300, "min", ha="center", fontsize=7.5, color=GREEN)
    axis.text(8, 300, "ref", ha="center", fontsize=7.5, color=VERMILLION)
    _panel_badge(axis, "B", "FlowER 84:1")


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": TOKENS["metrics"]["font"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 8.0,
            "axes.titlesize": TOKENS["metrics"]["title"],
            "axes.edgecolor": MUTED,
            "axes.linewidth": TOKENS["metrics"]["axis"],
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.solid_capstyle": "round",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render(pilot_path: Path, case_path: Path, output: Path) -> None:
    """Render the two-panel pilot figure after validating frozen evidence."""
    # Recompute statistics from digest-checked cases, not a cached report.
    pilot = summarize(pilot_path.parent)
    case = _load(case_path)
    _verify_embedded_digest(case, "record_sha256", context="alternative ITS case")
    if pilot.get("case_records") != 100:
        raise ValueError("the publication figure requires the frozen 100-case pilot")
    if case.get("reaction_id") != "84:1" or case.get("source_line") != 109:
        raise ValueError("the publication figure requires frozen FlowER record 84:1")
    _configure_style()
    write_palette()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(TOKENS["metrics"]["width_inches"], 3.65),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.43},
    )
    _cohort_panel(axes[0], pilot)
    _ladder_panel(axes[1], case)
    figure.subplots_adjust(left=0.065, right=0.975, top=0.79, bottom=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 8, 28, tzinfo=timezone.utc)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches=None,
        metadata={
            "Title": "Synister exact-shell evidence",
            "Author": "Tieu-Long Phan",
            "Creator": "scripts/plot_synister_figures.py",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    plt.close(figure)


def main() -> None:
    """Parse command-line arguments and render the evidence figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.pilot, args.case, args.output)


if __name__ == "__main__":
    main()
