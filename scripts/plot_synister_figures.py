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

# Canonical palette from ../Style/style.tex.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#E64B35"
INK = "#1A1A1A"
MUTED = "#666666"
HAIR = "#D9D9D9"
WASH = "#F7F7F7"


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
        bbox={"boxstyle": "circle,pad=0.22", "fc": INK, "ec": "none"},
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
    modes = pilot["modes"]
    total = int(pilot["case_records"])
    minimal = modes["minimal"]
    reference = modes["reference_cd"]
    rows = [
        (
            "Shell closed",
            minimal["cases"],
            total,
            reference["cases"],
            total,
        ),
        (
            ">1 exact ITS class",
            minimal["multiple_exact_its_classes"],
            minimal["cases"],
            reference["multiple_exact_its_classes"],
            reference["cases"],
        ),
        (
            "Map-dependent centre",
            minimal["unstable_reaction_centres"],
            minimal["cases"],
            reference["unstable_reaction_centres"],
            reference["cases"],
        ),
        (
            "Reference ITS observed",
            minimal["reference_its_class_observed"],
            minimal["cases"],
            reference["reference_its_class_observed"],
            reference["cases"],
        ),
    ]
    y = np.arange(len(rows), dtype=float)[::-1]
    offset = 0.115
    for position, (_, min_n, min_d, ref_n, ref_d) in zip(y, rows):
        min_rate = 100.0 * min_n / min_d
        ref_rate = 100.0 * ref_n / ref_d
        axis.plot(
            [min_rate, ref_rate],
            [position + offset, position - offset],
            color=HAIR,
            linewidth=1.1,
            zorder=1,
        )
        axis.scatter(
            min_rate,
            position + offset,
            s=24,
            color=BLUE,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.scatter(
            ref_rate,
            position - offset,
            s=27,
            marker="s",
            color=ORANGE,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.annotate(
            f"{min_n}/{min_d}",
            (min_rate, position + offset),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
            color=BLUE,
        )
        axis.annotate(
            f"{ref_n}/{ref_d}",
            (ref_rate, position - offset),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
            color=ORANGE,
        )

    axis.set_yticks(y, [row[0] for row in rows])
    axis.set_xlim(0, 112)
    axis.set_ylim(-0.55, 3.55)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("fraction of the stated denominator (%)")
    axis.grid(axis="x", color=HAIR, linewidth=0.55, zorder=0)
    axis.tick_params(axis="y", length=0, pad=5)
    min_classes = minimal["alternative_its_classes_relative_to_reference"]
    ref_classes = reference["alternative_its_classes_relative_to_reference"]
    axis.scatter(
        [], [], s=24, color=BLUE, label=f"Minimal CD · {min_classes} alternatives"
    )
    axis.scatter(
        [],
        [],
        s=27,
        marker="s",
        color=ORANGE,
        label=f"Reference CD · {ref_classes} alternatives",
    )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    _panel_badge(axis, "A", "Closed-shell pilot outcomes")


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
    distance, mappings, classes = _exact_ladder(case)
    axis.axvspan(5.78, 6.22, color=GREEN, alpha=0.075, linewidth=0)
    axis.axvspan(7.78, 8.22, color=VERMILLION, alpha=0.07, linewidth=0)
    axis.plot(
        distance,
        mappings,
        color=BLUE,
        marker="o",
        markersize=4.2,
        linewidth=1.45,
        label="Labeled maps",
        zorder=3,
    )
    axis.plot(
        distance,
        classes,
        color=GREEN,
        marker="s",
        markersize=4.0,
        linewidth=1.45,
        label="Exact ITS classes",
        zorder=3,
    )
    for x_value, map_count, class_count in zip(distance, mappings, classes):
        if map_count == 0:
            axis.annotate(
                "0",
                (x_value, 0),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6.3,
                color=MUTED,
            )
            continue
        axis.annotate(
            str(map_count),
            (x_value, map_count),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=6.3,
            color=BLUE,
        )
        axis.annotate(
            str(class_count),
            (x_value, class_count),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            fontsize=6.3,
            color=GREEN,
        )

    axis.set_yscale("symlog", linthresh=1, linscale=0.35, base=10)
    axis.set_xlim(3.55, 12.45)
    axis.set_ylim(-0.18, 290)
    axis.set_xticks(distance, [str(int(value)) for value in distance])
    axis.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    axis.set_xlabel("chemical distance, CD")
    axis.set_ylabel("exact count (symmetric log scale)")
    axis.grid(axis="y", color=HAIR, linewidth=0.55, zorder=0)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(-0.01, 0.94),
        frameon=False,
        handlelength=1.6,
        labelspacing=0.35,
    )
    axis.text(
        5.92,
        230,
        "global minimum",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=GREEN,
    )
    axis.text(
        8.08,
        230,
        "reference CD",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=VERMILLION,
    )
    _panel_badge(axis, "B", "Record 84:1 exact spectrum")


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.55,
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.solid_capstyle": "round",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render(pilot_path: Path, case_path: Path, output: Path) -> None:
    """Render the two-panel pilot figure after validating frozen evidence."""
    pilot = _load(pilot_path)
    case = _load(case_path)
    if pilot.get("case_records") != 100:
        raise ValueError("the publication figure requires the frozen 100-case pilot")
    if case.get("reaction_id") != "84:1" or case.get("source_line") != 109:
        raise ValueError("the publication figure requires frozen FlowER record 84:1")
    _configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        gridspec_kw={"width_ratios": [1.13, 1.0], "wspace": 0.40},
    )
    _cohort_panel(axes[0], pilot)
    _ladder_panel(axes[1], case)
    figure.subplots_adjust(left=0.16, right=0.985, top=0.83, bottom=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 8, 28, tzinfo=timezone.utc)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
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
