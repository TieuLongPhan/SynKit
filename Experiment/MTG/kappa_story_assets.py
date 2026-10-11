"""Draw actual native story vertices/edges for the first prespecified pulse seed."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RULES = ("a", "b", "c", "d", "pulse_up", "pulse_down")


def panel(ax, story, title):
    nodes = {n: value for n, value in story["nodes"] if value[0] == 1}
    edges = {
        kind: [
            (e["from"], e["to"])
            for e in story[kind]
            if e["from"] in nodes and e["to"] in nodes
        ]
        for kind in ("cause", "inhibit")
    }
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(edges["cause"] + edges["inhibit"])
    assert nx.is_directed_acyclic_graph(graph)
    generations = list(nx.topological_generations(graph))
    positions = {
        n: ((i - (len(layer) - 1) / 2) * 1.8, -depth)
        for depth, layer in enumerate(generations)
        for i, n in enumerate(sorted(layer))
    }
    counts = {name: 0 for name in RULES}
    labels = {}
    for n in nx.lexicographical_topological_sort(graph):
        name = RULES[nodes[n][1]]
        counts[name] += 1
        labels[n] = (
            name
            if not name.startswith("pulse")
            else (f"p{'+' if name == 'pulse_up' else '−'} {counts[name]}")
        )
    for kind, color, style in (
        ("cause", "#255b93", "solid"),
        ("inhibit", "#ae5b1c", "dashed"),
    ):
        for start, end in edges[kind]:
            long_edge = positions[start][1] - positions[end][1] > 1
            radius = (0.24 if positions[start][0] < 0 else -0.24) if long_edge else 0
            nx.draw_networkx_edges(
                graph,
                positions,
                ax=ax,
                edgelist=[(start, end)],
                edge_color=color,
                style=style,
                connectionstyle=f"arc3,rad={radius}",
                arrowsize=16,
                node_size=1000,
                width=1.5,
            )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=1000,
        node_color=[
            "#ffe6cc" if RULES[nodes[n][1]].startswith("pulse") else "#e2ecf6"
            for n in graph
        ],
        edgecolors="#59636d",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(graph, positions, labels, ax=ax, font_size=11)
    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-len(generations) + 0.2, 0.65)
    ax.axis("off")
    return {"rule_vertices": len(nodes), **{k: len(v) for k, v in edges.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "Experiment/MTG/evidence/kappa_history_comparison.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "paper/mtg/fig/kappa_native_stories.pdf"
    )
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "PASS"
    row = report["pulse_panel"]["runs"][0]
    assert row["seed"] == 1 and row["goal_reached"]
    assert (
        row["stories"]["none"]["source_program"]["final_world"]
        == row["stories"]["weak"]["source_program"]["final_world"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 5.7))
    summaries = [
        panel(
            axes[0],
            row["stories"]["none"]["native_story"],
            "Native causal past: 8 events",
        ),
        panel(
            axes[1],
            row["stories"]["weak"]["native_story"],
            "Native weak compression: 4 events",
        ),
    ]
    fig.legend(
        [
            Line2D([0], [0], color="#255b93", lw=1.5),
            Line2D([0], [0], color="#ae5b1c", lw=1.5, linestyle="--"),
        ],
        ["Native cause edge", "Native inhibit edge"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(top=0.9, bottom=0.09, left=0.04, right=0.96, wspace=0.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        metadata={
            "Creator": "MTG native story adapter",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    if args.preview:
        fig.savefig(args.preview, dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "PASS",
                "seed": 1,
                "native_rule_graphs": summaries,
                "report_sha256": sha256(args.report.read_bytes()).hexdigest(),
                "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "output_sha256": sha256(args.output.read_bytes()).hexdigest(),
                "omitted": "Agent introductions, observer and their incident edges.",
            }
        )
    )


if __name__ == "__main__":
    main()
