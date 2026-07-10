"""
Visualize typed-action statistics for generated polar mechanistic JSON.

Usage:
  python debug/visualize_polar_json.py
  python debug/visualize_polar_json.py --stats debug/data/mech/polar_stats.json
  python debug/visualize_polar_json.py --json debug/data/mech/polar.json --top 20

Outputs typed-action PNG figures to debug/data/mech/figures by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_polar_json import analyze

FIGSIZE = (11, 6.5)
BAR_COLOR = "#3b82f6"
BAR_COLOR_ALT = "#14b8a6"
GRID_COLOR = "#d9dee7"
HEATMAP_CMAP = "Blues"


def _load_rows(json_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected top-level JSON list in {json_path}")
    return rows


def _load_summary(stats_path: Path | None, json_path: Path, top: int) -> dict[str, Any]:
    if stats_path is not None and stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    return analyze(json_path, top=top)


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_pngs(path: Path) -> None:
    if not path.exists():
        return
    for png in path.glob("*.png"):
        png.unlink()


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _bar_labels(ax: plt.Axes, bars, values: list[int]) -> None:
    if not values:
        return
    offset = max(values) * 0.01
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=9,
        )


def plot_vertical_bar(
    rows: list[dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    color: str = BAR_COLOR,
) -> None:
    values = [str(row["value"]) for row in rows]
    counts = [int(row["count"]) for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(values, counts, color=color, edgecolor="#1f2937", linewidth=0.5)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, out_path)


def plot_horizontal_bar(
    rows: list[dict[str, Any]],
    title: str,
    out_path: Path,
    top: int | None = None,
    color: str = BAR_COLOR,
) -> None:
    selected = rows[:top] if top else rows
    labels = [str(row["value"]) for row in selected][::-1]
    counts = [int(row["count"]) for row in selected][::-1]

    height = max(5.5, min(14, 0.42 * len(labels) + 1.8))
    fig, ax = plt.subplots(figsize=(12, height))
    bars = ax.barh(labels, counts, color=color, edgecolor="#1f2937", linewidth=0.5)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("Count")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    _bar_labels(ax, bars, counts)
    ax.margins(x=0.14)

    _save(fig, out_path)


def plot_summary_card(summary: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    lines = [
        ("Rows", f"{summary['total_rows']:,}"),
        ("Arrow steps", f"{summary['total_arrow_steps']:,}"),
        ("Typed actions", f"{summary['unique_epd_lw_actions']:,}"),
        ("Typed sequences", f"{summary['unique_epd_lw_sequences']:,}"),
    ]

    ax.text(0.02, 0.94, "Polar Typed-Action Summary", fontsize=18, weight="bold")
    y = 0.80
    for label, value in lines:
        ax.text(0.08, y, label, fontsize=13, color="#374151")
        ax.text(0.62, y, value, fontsize=13, weight="bold", color="#111827")
        y -= 0.10

    _save(fig, out_path)


def _sorted_orbital_classes(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(row["orbital_class"] for row in rows)
    return [value for value, _count in counts.most_common()]


def _sorted_actions(rows: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(step[0] for row in rows for step in row[key])
    return [value for value, _count in counts.most_common()]


def _matrix(
    row_labels: list[Any],
    col_labels: list[Any],
    counts: dict[tuple[Any, Any], int],
    normalize_rows: bool = False,
) -> list[list[float]]:
    matrix = []
    for row_label in row_labels:
        values = [
            float(counts.get((row_label, col_label), 0)) for col_label in col_labels
        ]
        if normalize_rows:
            total = sum(values)
            values = [(value / total * 100.0) if total else 0.0 for value in values]
        matrix.append(values)
    return matrix


def plot_heatmap(
    matrix: list[list[float]],
    row_labels: list[Any],
    col_labels: list[Any],
    title: str,
    out_path: Path,
    value_fmt: str = ".0f",
    colorbar_label: str = "Count",
) -> None:
    width = max(8, 0.65 * len(col_labels) + 4)
    height = max(5.5, 0.48 * len(row_labels) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(label) for label in col_labels], rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([str(label) for label in row_labels])

    max_value = max((value for row in matrix for value in row), default=0)
    threshold = max_value * 0.55
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value == 0:
                continue
            color = "white" if value >= threshold else "#111827"
            ax.text(
                j,
                i,
                format(value, value_fmt),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    _save(fig, out_path)


def plot_stacked_number_arrow_by_orbital(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    orbital_classes = _sorted_orbital_classes(rows)
    arrow_counts = sorted({row["number_arrow"] for row in rows})

    counts = defaultdict(int)
    totals = Counter()
    for row in rows:
        orbital = row["orbital_class"]
        counts[(orbital, row["number_arrow"])] += 1
        totals[orbital] += 1

    fig, ax = plt.subplots(figsize=(12, 7))
    left = [0.0] * len(orbital_classes)
    palette = plt.get_cmap("tab20")

    for idx, number_arrow in enumerate(arrow_counts):
        values = [
            (
                counts[(orbital, number_arrow)] / totals[orbital] * 100.0
                if totals[orbital]
                else 0.0
            )
            for orbital in orbital_classes
        ]
        ax.barh(
            orbital_classes,
            values,
            left=left,
            label=str(number_arrow),
            color=palette(idx),
            edgecolor="white",
            linewidth=0.5,
        )
        left = [old + value for old, value in zip(left, values)]

    ax.set_title(
        "Number of Arrows by Orbital Class (Row %)", fontsize=15, weight="bold"
    )
    ax.set_xlabel("Percent within orbital class")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title="number_arrow", ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.20)
    )
    _save(fig, out_path)


def plot_number_arrow_by_orbital_heatmap(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    orbital_classes = _sorted_orbital_classes(rows)
    arrow_counts = sorted({row["number_arrow"] for row in rows})
    counts = defaultdict(int)
    for row in rows:
        counts[(row["orbital_class"], row["number_arrow"])] += 1

    matrix = _matrix(orbital_classes, arrow_counts, counts)
    plot_heatmap(
        matrix,
        orbital_classes,
        arrow_counts,
        "Number of Arrows by Orbital Class (Counts)",
        out_path,
        value_fmt=".0f",
    )


def plot_typed_action_by_orbital_heatmap(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    orbital_classes = _sorted_orbital_classes(rows)
    actions = _sorted_actions(rows, "epd_lw")
    counts = defaultdict(int)
    for row in rows:
        for step in row["epd_lw"]:
            counts[(row["orbital_class"], step[0])] += 1

    matrix = _matrix(orbital_classes, actions, counts, normalize_rows=True)
    plot_heatmap(
        matrix,
        orbital_classes,
        actions,
        "Typed Action Mix by Orbital Class (Row-Normalized %)",
        out_path,
        value_fmt=".1f",
        colorbar_label="Percent",
    )


def plot_action_by_position_heatmap(
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    out_path: Path,
    normalize: bool = True,
) -> None:
    max_position = max((len(row[key]) for row in rows), default=0)
    positions = list(range(1, max_position + 1))
    actions = _sorted_actions(rows, key)
    counts = defaultdict(int)
    position_totals = Counter()

    for row in rows:
        for position, step in enumerate(row[key], start=1):
            counts[(step[0], position)] += 1
            position_totals[position] += 1

    matrix = []
    for action in actions:
        row = []
        for position in positions:
            total = position_totals[position]
            value = (
                counts[(action, position)] / total * 100.0
                if normalize and total
                else counts[(action, position)]
            )
            row.append(value)
        matrix.append(row)

    col_labels = [
        f"{position}\n(n={position_totals[position]:,})" for position in positions
    ]
    plot_heatmap(
        matrix,
        actions,
        col_labels,
        title,
        out_path,
        value_fmt=".1f" if normalize else ".0f",
        colorbar_label="Percent at position" if normalize else "Count at position",
    )


def plot_top_typed_sequences_by_arrow_count(
    rows: list[dict[str, Any]], out_path: Path, top_per_group: int = 3
) -> None:
    labels = []
    counts = []
    for number_arrow in sorted({row["number_arrow"] for row in rows}):
        sequence_counts = Counter(
            " ; ".join(step[0] for step in row["epd_lw"])
            for row in rows
            if row["number_arrow"] == number_arrow
        )
        for sequence, count in sequence_counts.most_common(top_per_group):
            labels.append(f"{number_arrow} arrows | {sequence}")
            counts.append(count)

    selected = [
        {"value": label, "count": count, "percent": 0.0}
        for label, count in zip(labels, counts)
    ]
    plot_horizontal_bar(
        selected,
        f"Top {top_per_group} Typed Sequences Within Each Arrow Count",
        out_path,
        color="#f97316",
    )


def make_figures(
    summary: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path, top: int
) -> list[Path]:
    _prepare_output_dir(out_dir)
    _clean_pngs(out_dir)
    outputs = []

    plots = [
        (
            "summary.png",
            lambda path: plot_summary_card(summary, path),
        ),
        (
            "epd_lw_action_counts.png",
            lambda path: plot_horizontal_bar(
                summary["epd_lw_action_counts"],
                "Typed EPD Action Counts",
                path,
                color=BAR_COLOR_ALT,
            ),
        ),
        (
            "top_epd_lw_sequences.png",
            lambda path: plot_horizontal_bar(
                summary["top_epd_lw_sequences"],
                f"Top {top} Typed EPD Sequences",
                path,
                top=top,
            ),
        ),
        (
            "typed_action_by_orbital_percent.png",
            lambda path: plot_typed_action_by_orbital_heatmap(rows, path),
        ),
        (
            "typed_action_by_position_percent.png",
            lambda path: plot_action_by_position_heatmap(
                rows,
                "epd_lw",
                "Typed Action Distribution by Arrow Position (% among rows with that position)",
                path,
                normalize=True,
            ),
        ),
        (
            "typed_action_by_position_counts.png",
            lambda path: plot_action_by_position_heatmap(
                rows,
                "epd_lw",
                "Typed Action Counts by Arrow Position",
                path,
                normalize=False,
            ),
        ),
        (
            "top_typed_sequences_by_arrow_count.png",
            lambda path: plot_top_typed_sequences_by_arrow_count(rows, path),
        ),
    ]

    for filename, plotter in plots:
        out_path = out_dir / filename
        plotter(out_path)
        outputs.append(out_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", default="debug/data/mech/polar.json", help="Input polar JSON file."
    )
    parser.add_argument(
        "--stats",
        default="debug/data/mech/polar_stats.json",
        help="Optional stats JSON produced by analyze_polar_json.py.",
    )
    parser.add_argument(
        "--out-dir", default="debug/data/mech/figures", help="Output figure directory."
    )
    parser.add_argument(
        "--top", type=int, default=15, help="Top sequence/code count to plot."
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    stats_path = Path(args.stats) if args.stats else None
    summary = _load_summary(stats_path, json_path, top=args.top)
    rows = _load_rows(json_path)
    outputs = make_figures(summary, rows, Path(args.out_dir), top=args.top)

    print("wrote figures:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
