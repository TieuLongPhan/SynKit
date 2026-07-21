#!/usr/bin/env python3
"""Plot four-method expansion results for general and radical corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
PAPER_FIGURE = ROOT / "paper" / "lwg" / "fig" / "partial_aam_comparison.png"
METHODS = ("synkit", "gm", "rb1", "rb2")
LABELS = {
    "synkit": r"$\mathtt{SynKit}$",
    "gm": r"$\mathtt{GM}$",
    "rb1": r"$\mathtt{RB1}$",
    "rb2": r"$\mathtt{RB2}$",
}
# Validated colourblind-safe categorical palette (Okabe--Ito); identity is also
# carried by the direct method labels on every row.
COLORS = {
    "synkit": "#0072B2",
    "gm": "#009E73",
    "rb1": "#E69F00",
    "rb2": "#CC79A7",
}
# Display order (fastest general expansion first); shared across both panels.
ORDER = ("synkit", "rb1", "rb2", "gm")
INK = "#232323"
MUTED = "#6B6B6B"
GRID = "#E7E9EB"
SPINE = "#B7BCC1"
RADICAL_RECORDS = 5_426

# The external radical counts were retained in sprint/SS_LOG.md before the
# redundant per-case files were removed. SynKit is loaded from its current
# minimal-path five-run aggregate below.
EXTERNAL_RADICAL_VALID_COUNTS = (3_869, 5_326, 5_326)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synkit-results",
        type=Path,
        default=RESULTS / "synkit-normal-expansion-5x" / "aggregate.json",
    )
    parser.add_argument(
        "--external-results",
        type=Path,
        default=RESULTS / "gm-rb-normal-expansion-5x",
    )
    parser.add_argument(
        "--synkit-radical-results",
        type=Path,
        default=RESULTS / "synkit-radical-expansion-5x" / "aggregate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "expansion-comparison-general-radical",
        help="Output stem; PDF and PNG are written.",
    )
    return parser.parse_args()


def load_synkit(path: Path) -> list[float]:
    payload = json.loads(path.read_text())
    aggregate = next(
        item for item in payload["aggregates"] if item["method"] == "synkit"
    )
    values = aggregate["metrics"]["mean_generation_seconds_per_attempt"]["values"]
    return [float(value) * 1000 for value in values]


def load_synkit_radical_coverage(path: Path) -> int:
    payload = json.loads(path.read_text())
    aggregate = next(
        item for item in payload["aggregates"] if item["method"] == "synkit"
    )
    metrics = aggregate["metrics"]
    return round(float(metrics["accepted_coverage"]["mean"]) * RADICAL_RECORDS)


def load_external(directory: Path) -> dict[str, list[float]]:
    timings = {method: [] for method in METHODS[1:]}
    for path in sorted(directory.glob("general-*-generation.json")):
        payload = json.loads(path.read_text())
        method = payload["methods"][0]
        name = str(method["method"])
        if name in timings:
            timings[name].append(float(method["generation_seconds"]["mean"]) * 1000)
    missing = [name for name, values in timings.items() if len(values) != 5]
    if missing:
        raise ValueError(f"Expected five retained runs for: {', '.join(missing)}")
    return timings


def _style_open_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, length=3, width=0.8)
    axis.set_axisbelow(True)


def _panel_key(axis, label: str) -> None:
    axis.text(
        -0.02,
        1.07,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def _method_yaxis(axis, order) -> None:
    axis.set_yticks(list(range(len(order))), [LABELS[m] for m in order])
    axis.set_ylim(-0.6, len(order) - 0.4)
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0, pad=8)
    _style_open_axis(axis)
    axis.spines["left"].set_visible(False)


def _general_dot_panel(axis, samples: dict, label: str) -> None:
    """Cleveland dot plot: mean +/- sd generation time, five runs per method."""
    _method_yaxis(axis, ORDER)
    for y, method in enumerate(ORDER):
        values = samples[method]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        color = COLORS[method]
        axis.scatter(
            values,
            [y] * len(values),
            s=15,
            color=color,
            alpha=0.30,
            edgecolor="none",
            zorder=2,
        )
        axis.errorbar(
            mean,
            y,
            xerr=sd,
            fmt="o",
            markersize=8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.2,
            zorder=4,
        )
        axis.text(
            mean + sd + 0.035,
            y,
            f"{mean:.3f} \u00b1 {sd:.3f}",
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK,
        )
    axis.set_xlim(1.70, 2.74)
    axis.set_xlabel("Time per input (ms)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_title(
        "General-corpus expansion time",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    _panel_key(axis, label)


def _coverage_bar_panel(axis, coverage: dict, label: str) -> None:
    """Radical valid-completion coverage as bars; 100% General ITS reference."""
    _method_yaxis(axis, ORDER)
    base = 60.0
    for y, method in enumerate(ORDER):
        pct = coverage[method] / RADICAL_RECORDS * 100
        axis.barh(
            y,
            pct - base,
            left=base,
            height=0.54,
            color=COLORS[method],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        inside = pct - base > 8
        axis.text(
            pct - 0.9 if inside else pct + 0.9,
            y,
            f"{pct:.1f}%",
            ha="right" if inside else "left",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="white" if inside else INK,
            zorder=4,
        )
    axis.axvline(100, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
    axis.text(
        100,
        -0.52,
        "General ITS 100%",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=MUTED,
    )
    axis.set_xlim(base, 109)
    axis.set_xlabel("Valid radical completion (%)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_title(
        "Radical completion coverage",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    _panel_key(axis, label)


def plot(
    synkit: list[float],
    external: dict[str, list[float]],
    radical_synkit_accepted: int,
    output: Path,
) -> None:
    samples = {"synkit": synkit, **external}
    coverage = {
        "synkit": radical_synkit_accepted,
        "gm": EXTERNAL_RADICAL_VALID_COUNTS[0],
        "rb1": EXTERNAL_RADICAL_VALID_COUNTS[1],
        "rb2": EXTERNAL_RADICAL_VALID_COUNTS[2],
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.3,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": 160,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    _general_dot_panel(axes[0], samples, "A")
    _coverage_bar_panel(axes[1], coverage, "B")
    figure.tight_layout(w_pad=3.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    figure.savefig(PAPER_FIGURE, bbox_inches="tight", dpi=300)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    radical_synkit_accepted = load_synkit_radical_coverage(args.synkit_radical_results)
    plot(
        load_synkit(args.synkit_results),
        load_external(args.external_results),
        radical_synkit_accepted,
        args.output,
    )
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
