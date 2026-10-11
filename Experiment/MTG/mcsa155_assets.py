#!/usr/bin/env python
"""Render a vector figure from passing chemical-case and network reports."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render(case, network, output):
    assert case["status"] == network["status"] == "PASS"
    assert network["summary"]["configuration_worlds_checked"] == 98
    assert network["summary"]["coexistence"] == [True, False]
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.4, 1, 1]}
    )
    colors = ("#777777", "#166a91")
    for row, color in zip(case["candidates"], colors):
        values = row["thioester_motif_at_cuts"]
        axes[0].plot(
            range(len(values)),
            values,
            "o-",
            color=color,
            label=f"Source ID {row['proposal']}",
        )
    axes[0].set(
        title="A  Strict intermediate motif",
        xlabel="Completed events",
        ylabel="Motif present",
    )
    axes[0].set_yticks([0, 1], ["No", "Yes"])
    axes[0].set_xticks(range(9))
    axes[0].set_ylim(-0.15, 1.35)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].text(4, 1.12, "Asp–C(=O)–S–CoA", ha="center", fontsize=8)
    for ax, independent in zip(axes[1:], (True, False)):
        prefixes = (
            [(a, b) for a in range(9) for b in range(9)]
            if independent
            else [(a, 0) for a in range(9)] + [(8, b) for b in range(1, 9)]
        )
        ax.scatter(*zip(*prefixes), s=15, color="#a6b5bf")
        ax.set(
            xlim=(-0.7, 8.7),
            ylim=(-0.7, 8.7),
            xlabel="Copy 1 prefix",
            ylabel="Copy 2 prefix",
        )
        ax.set_xticks([0, 4, 8])
        ax.set_yticks([0, 4, 8])
        ax.set_aspect("equal")
        ax.scatter(
            [4],
            [4],
            marker="*" if independent else "x",
            s=130,
            color="#b9442d",
            zorder=3,
        )
        ax.set_title(
            (
                "B  Two separate sites\n81 checked cuts"
                if independent
                else "C  One site reused\n17 checked cuts"
            ),
            fontsize=10,
            pad=8,
        )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Title": "M-CSA155 configuration construction", "Creator": __file__},
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=Path("Experiment/MTG/evidence/mcsa155_case.json")
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("Experiment/MTG/evidence/mcsa155_network.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        json.loads(args.case.read_text()),
        json.loads(args.network.read_text()),
        args.output,
    )
    print("PASS: vector figure rendered from verified reports")


if __name__ == "__main__":
    main()
