#!/usr/bin/env python3
"""Build exact toy diagrams and pilot structure plots, with their source data.

The toy graphs are pedagogical examples, not measured chemical reactions.
All pilot observations are read from digest-verified frozen case records.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle

from plot_synister_figures import (
    BLUE,
    HAIR,
    MUTED,
    ORANGE,
    _configure_style,
    _panel_badge,
    write_palette,
)
from summarize_synister_evidence import _verified_records
from synister_diagrams import draw_toy, draw_hydrogen, draw_workflow

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/synister"
FIGURES = PAPER / "figures"
SOURCES = FIGURES / "source_data"


def exact_toy() -> dict:
    """Exhaust all 120 maps and all 120 canonical relabelings on five vertices."""
    n = 5
    pairs = list(combinations(range(n), 2))
    maps = list(permutations(range(n)))
    reactant = {(0, 1), (1, 2), (2, 3), (3, 4)}
    product = {(0, 1), (1, 2), (1, 3), (3, 4)}

    def state(mapping):
        return {
            (i, j): (
                int((i, j) in reactant),
                int(tuple(sorted((mapping[i], mapping[j]))) in product),
            )
            for i, j in pairs
        }

    states = {p: state(p) for p in maps}
    costs = {p: sum(abs(a - b) for a, b in s.values()) for p, s in states.items()}
    minimum = min(costs.values())
    optimal = [p for p in maps if costs[p] == minimum]
    classes = {}
    for p in optimal:
        s = states[p]
        canonical = min(
            tuple(s[tuple(sorted((q[i], q[j])))] for i, j in pairs) for q in maps
        )
        classes.setdefault(canonical, []).append(p)
    automorphisms = [
        p
        for p in maps
        if all(
            ((i, j) in product) == (tuple(sorted((p[i], p[j]))) in product)
            for i, j in pairs
        )
    ]
    orbits = {
        min(tuple(h[p[i]] for i in range(n)) for h in automorphisms) for p in optimal
    }

    # Independent full paired-edge isomorphism validates the canonical partition.
    def graph(p):
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(
            (i, j, {"state": s}) for (i, j), s in states[p].items() if s != (0, 0)
        )
        return g

    representatives = [items[0] for items in classes.values()]
    for first, second in combinations(representatives, 2):
        if nx.is_isomorphic(
            graph(first),
            graph(second),
            edge_match=lambda a, b: a["state"] == b["state"],
        ):
            raise ValueError("Canonical classes disagree with graph isomorphism")
    for items in classes.values():
        if not all(
            nx.is_isomorphic(
                graph(items[0]),
                graph(p),
                edge_match=lambda a, b: a["state"] == b["state"],
            )
            for p in items
        ):
            raise ValueError("A canonical class contains inequivalent maps")
    if (minimum, len(optimal), len(orbits), len(classes)) != (2, 16, 8, 4):
        raise ValueError("The illustrated exact toy result changed")
    return {
        "kind": "exhaustive_synthetic_graph_example",
        "vertices": n,
        "reactant_edges": sorted(reactant),
        "product_edges": sorted(product),
        "minimum_cd": minimum,
        "labeled_maps": len(optimal),
        "product_group_order": len(automorphisms),
        "product_orbits": len(orbits),
        "its_classes": len(classes),
        "all_optimal_maps": optimal,
        "classes": [
            {
                "representative": p,
                "multiplicity": len(items),
                "edge_states": [
                    [i, j, *s] for (i, j), s in states[p].items() if s != (0, 0)
                ],
            }
            for items in classes.values()
            for p in [items[0]]
        ],
    }


def hydrogen_oracle() -> dict:
    """Literal six-vertex permutations verify the conditional-lift example."""
    heavy_edges = {(0, 1), (1, 2)}
    r = heavy_edges | {(0, 3), (0, 4), (0, 5)}
    p = heavy_edges | {(1, 3), (1, 4), (1, 5)}
    rows = []
    for heavy in permutations(range(3)):
        values = []
        for hydrogen in permutations(range(3, 6)):
            mapping = heavy + hydrogen
            values.append(
                sum(
                    ((i, j) in r) != (tuple(sorted((mapping[i], mapping[j]))) in p)
                    for i, j in combinations(range(6), 2)
                )
            )
        hcd = sum(
            ((i, j) in heavy_edges)
            != (tuple(sorted((heavy[i], heavy[j]))) in heavy_edges)
            for i, j in combinations(range(3), 2)
        )
        penalty = sum(
            abs(a - b) for a, b in zip((3, 0, 0), [((0, 3, 0))[j] for j in heavy])
        )
        if min(values) != hcd + penalty:
            raise ValueError(
                "Hydrogen decomposition disagrees with literal enumeration"
            )
        rows.append(
            dict(
                heavy_map=heavy,
                heavy_cd=hcd,
                hydrogen_penalty=penalty,
                full_minimum=min(values),
                optimal_hydrogen_maps=values.count(min(values)),
            )
        )
    if rows[0]["full_minimum"] != 6 or min(row["full_minimum"] for row in rows) != 2:
        raise ValueError("Hydrogen counterexample changed")
    return {
        "kind": "synthetic_pendant_hydrogen_oracle",
        "enumerated_maps": 36,
        "rows": rows,
    }


def pilot_structure() -> dict:
    campaign = PAPER / "evidence/pilot100_v4"
    manifest, records = _verified_records(campaign)
    if len(records) != 100 or len({r["source_line"] for r in records}) != 100:
        raise ValueError("Expected 100 distinct pilot records")
    rows = []
    for record in records:
        shell = record.get("shells", {}).get("minimal", {})
        if not (shell.get("complete") and shell.get("structure", {}).get("complete")):
            continue
        rc = shell["reaction_center"]
        varying = (
            rc["bond_union"] != rc["bond_intersection"]
            or rc["atom_union"] != rc["atom_intersection"]
        )
        rows.append(
            dict(
                source_line=record["source_line"],
                reaction_id=record["reaction_id"],
                labeled_maps=int(shell["labeled_solution_count"]),
                product_orbits=int(shell["representative_solution_count"]),
                its_classes=int(shell["structure"]["observed_its_class_count"]),
                varying_centre=int(varying),
            )
        )
    rows.sort(key=lambda r: r["source_line"])
    matrix = np.zeros((2, 2), dtype=int)
    for row in rows:
        matrix[int(row["its_classes"] > 1), row["varying_centre"]] += 1
    if len(rows) != 46 or sum(r["its_classes"] > 1 for r in rows) != 16:
        raise ValueError("Pilot denominator/class count changed")
    with (SOURCES / "pilot_structure.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _configure_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 3.35), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.18, wspace=0.62)
    ax = axes[0]
    for varied, marker, color, label in [
        (0, "o", BLUE, "Invariant center"),
        (1, "s", ORANGE, "Varying center"),
    ]:
        counts = Counter(
            (r["labeled_maps"], r["its_classes"])
            for r in rows
            if r["varying_centre"] == varied
        )
        for (labeled, classes), count in counts.items():
            # No jitter: exact coordinates, count encoded by marker area.
            ax.scatter(
                labeled,
                classes,
                s=22 + 12 * count,
                marker=marker,
                facecolors="none" if not varied else color,
                edgecolors=color,
                linewidths=1.0,
                alpha=0.80,
                zorder=3,
            )
        ax.scatter(
            [],
            [],
            s=32,
            marker=marker,
            facecolors="none" if not varied else color,
            edgecolors=color,
            label=label,
        )
    ax.plot([1, 256], [1, 256], color=HAIR, ls="--", lw=0.8, zorder=0)
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(1.5, 256),
        ylim=(0.7, 22),
        xlabel="Labeled maps, $L$",
        ylabel="ITS classes, $K$",
    )
    ax.set_xticks([2, 8, 32, 128], ["2", "8", "32", "128"])
    ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.grid(color=HAIR, linewidth=0.55)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    _panel_badge(ax, "A", "46 closed reactions")
    ax = axes[1]
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for (i, j), number in np.ndenumerate(matrix):
        color = BLUE if j == 0 else ORANGE
        ax.add_patch(
            Rectangle(
                (j - 0.46, i - 0.46),
                0.92,
                0.92,
                facecolor=color if number else HAIR,
                edgecolor="none",
                alpha=0.07,
            )
        )
        if (i, j) == (0, 1):
            ax.add_patch(
                Rectangle(
                    (j - 0.46, i - 0.46),
                    0.92,
                    0.92,
                    fill=False,
                    edgecolor=ORANGE,
                    linewidth=1.0,
                )
            )
        for k in range(int(number)):
            ax.scatter(
                j - 0.30 + 0.20 * (k % 4),
                i - 0.04 + 0.13 * (k // 4),
                s=16,
                marker="o" if j == 0 else "s",
                color=color,
                linewidths=0,
            )
        ax.text(
            j - 0.30,
            i - 0.29,
            str(number),
            va="center",
            ha="left",
            fontsize=14,
            color=color if number else MUTED,
        )
    ax.set_xticks([0, 1], ["Invariant", "Varying"])
    ax.set_yticks([0, 1], ["1 ITS", ">1 ITS"])
    ax.set_xlabel("Center")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_badge(ax, "B", "Center versus ITS")
    ax.text(
        0.5,
        -0.24,
        "1 symbol = 1 reaction",
        fontsize=7.5,
        ha="center",
        transform=ax.transAxes,
        color=MUTED,
    )
    fig.savefig(
        FIGURES / "pilot_structure.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return dict(
        case_records=100,
        minimum_structure_complete=len(rows),
        cross_tabulation=matrix.tolist(),
        row_labels=["single_its", "multiple_its"],
        column_labels=["invariant_centre", "varying_centre"],
        campaign_manifest_sha256=manifest["manifest_sha256"],
    )


def main() -> None:
    SOURCES.mkdir(parents=True, exist_ok=True)
    write_palette()
    toy = exact_toy()
    draw_toy(toy)
    draw_hydrogen()
    draw_workflow(toy)
    provenance = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "drawing_sha256": hashlib.sha256(
            (ROOT / "scripts/synister_diagrams.py").read_bytes()
        ).hexdigest(),
        "drawing_style_sha256": hashlib.sha256(
            (FIGURES / "figure_style.tex").read_bytes()
        ).hexdigest(),
        "worked_record_sha256": json.loads(
            (PAPER / "evidence/worked_unmapped_flower84_v1/record.json").read_text()
        )["record_sha256"],
        "style_sha256": hashlib.sha256(
            (FIGURES / "style_tokens.json").read_bytes()
        ).hexdigest(),
        "toy": toy,
        "hydrogen": hydrogen_oracle(),
        "pilot": pilot_structure(),
    }
    provenance["pilot_source_table_sha256"] = hashlib.sha256(
        (SOURCES / "pilot_structure.csv").read_bytes()
    ).hexdigest()
    (SOURCES / "figure_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "toy": {
                    k: toy[k]
                    for k in (
                        "minimum_cd",
                        "labeled_maps",
                        "product_orbits",
                        "its_classes",
                    )
                },
                "pilot": provenance["pilot"],
                "hydrogen_maps_checked": 36,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
