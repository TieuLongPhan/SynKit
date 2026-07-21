#!/usr/bin/env python3
"""Plot LLG versus atom-bond graph rewriting populations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results" / "bidirectional-replay"
PAPER_FIGURE = ROOT / "paper" / "lwg" / "fig" / "graph_rewriting_comparison.png"

COLORS = {"tuple": "#0E7C86", "typesGH": "#D9822B"}
LABELS = {
    "tuple": r"$\mathtt{LLG}$",
    "typesGH": r"$\mathtt{AtomBond}$",
}
INK = "#232323"
MUTED = "#6C737A"
GRID = "#E6E9EC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "graph-rewriting-comparison",
        help="Output stem; PDF and PNG are written.",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        default=PAPER_FIGURE,
        help="PNG copied into the LWG figure directory.",
    )
    return parser.parse_args()


def aggregate(path: Path) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"mappings": 0, "unique": 0, "recovered": 0}
    )
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            rows += 1
            for direction in ("forward", "backward"):
                result = case["directions"][direction]
                totals[direction]["mappings"] += int(result["mapping_count"])
                totals[direction]["unique"] += int(result["unique_reaction_count"])
                totals[direction]["recovered"] += result["status"] == "PASS"
    if rows != 39_732:
        raise ValueError(f"Expected 39,732 cases in {path}, found {rows}")
    for direction in ("forward", "backward"):
        if totals[direction]["recovered"] != rows:
            raise ValueError(f"Incomplete {direction} recovery in {path}")
    return dict(totals)


def style_axis(axis, title: str, label: str, xlim: tuple[float, float]) -> None:
    axis.axvline(0, color=COLORS["typesGH"], linewidth=1.4, alpha=0.9, zorder=1)
    axis.set_xlim(*xlim)
    axis.set_ylim(-0.62, 1.62)
    axis.set_yticks((0, 1), ("Forward", "Inverse"))
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
    axis.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    axis.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#B7BCC1")
    axis.tick_params(axis="y", length=0, pad=8, colors=INK, labelsize=9.5)
    axis.tick_params(axis="x", length=3, colors=MUTED, labelsize=8.5)
    axis.set_xlabel("change vs baseline", fontsize=8.6, color=MUTED)
    axis.set_title(title, fontsize=10, fontweight="bold", color=INK, pad=16)
    axis.text(
        -0.12,
        1.14,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def plot_panel(
    axis,
    data: dict[str, dict[str, dict[str, int]]],
    metric: str,
    title: str,
    label: str,
    xlim: tuple[float, float],
) -> None:
    style_axis(axis, title, label, xlim)
    span = xlim[1] - xlim[0]
    for position, direction in enumerate(("forward", "backward")):
        llg = data["tuple"][direction][metric]
        atom_bond = data["typesGH"][direction][metric]
        change = (llg / atom_bond - 1.0) * 100
        removed = atom_bond - llg
        # thin connector between baseline and LLG marker
        axis.plot(
            (0, change),
            (position, position),
            color="#C4C9CE",
            linewidth=2.0,
            solid_capstyle="round",
            zorder=2,
        )
        # AtomBond baseline (hollow) and LLG (filled)
        axis.scatter(
            0,
            position,
            s=64,
            facecolor="white",
            edgecolor=COLORS["typesGH"],
            linewidth=1.7,
            zorder=4,
        )
        axis.scatter(
            change,
            position,
            s=76,
            color=COLORS["tuple"],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        # counts above each marker
        axis.text(
            change,
            position - 0.30,
            f"{llg:,}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color=COLORS["tuple"],
        )
        axis.text(
            0.012 * span,
            position - 0.30,
            f"{atom_bond:,}",
            ha="left",
            va="bottom",
            fontsize=8.6,
            color=COLORS["typesGH"],
        )
        # reduction below the connector, plain (no box)
        axis.text(
            change / 2,
            position + 0.24,
            f"−{removed:,}  ({change:.2f}%)",
            ha="center",
            va="top",
            fontsize=8.2,
            color=MUTED,
            zorder=5,
        )


def main() -> int:
    args = parse_args()
    data = {
        representation: aggregate(args.results / f"{representation}-cases.jsonl.gz")
        for representation in ("tuple", "typesGH")
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    plot_panel(axes[0], data, "mappings", "All structural mappings", "A", (-2.35, 0.42))
    plot_panel(
        axes[1], data, "unique", "Unique standardized reactions", "B", (-1.02, 0.20)
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["tuple"],
            markeredgecolor="white",
            markersize=8,
            label=LABELS["tuple"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["typesGH"],
            markeredgewidth=1.6,
            markersize=8,
            label=LABELS["typesGH"],
        ),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
        fontsize=9.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.paper_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(args.paper_output, dpi=320, bbox_inches="tight")
    plt.close(figure)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))
    print(args.paper_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
