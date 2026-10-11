#!/usr/bin/env python
"""Render the actual overlay-collision report as a standalone vector figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render(report_path, output):
    report = json.loads(report_path.read_text())
    if report["status"] != "PASS":
        raise ValueError("A passing comparison report is required.")
    first, second = report["formal_query_separation"]
    if first["overlays"][0]["overlay"] != second["overlays"][0]["overlay"]:
        raise ValueError("The recorded published overlays must actually coincide.")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.7, 2.6), constrained_layout=True)
    for row, color, marker, name in (
        (first, "#0072B2", "o", "A: reaches order three"),
        (second, "#D55E00", "s", "B: never reaches order three"),
    ):
        values = row["mtg_bond_order_history"]
        ax.plot(
            range(len(values)),
            values,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5,
            label=name,
            linestyle="-" if name.startswith("A") else "--",
        )
    ax.set(
        xlabel="Event prefix (formal history)",
        ylabel="Bond order",
        xticks=range(5),
        yticks=(1, 2, 3),
        ylim=(0.8, 3.5),
        xlim=(-0.1, 4.1),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(
        "Same published overlay aggregate, different query answers", fontsize=10
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(json.dumps({"figure": str(output), "status": "PASS"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.report, args.output)


if __name__ == "__main__":
    main()
