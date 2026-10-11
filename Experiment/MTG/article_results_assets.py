"""Render Article result figures from frozen MTG reports, without rerunning chemistry.

Every plotted value is exported with its source-report hash. Timing medians are
recomputed from retained individual rounds. Corpus bars have a common denominator
and are overlapping analyses, not a partition or an accuracy measurement.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#697782"
REPORTS = (
    "synepd_post_repair_summary",
    "grouping_dag_post_repair_summary",
    "mcsa155_case",
    "mcsa155_network",
    "observation_network_audit",
    "selenophosphate_isotope_case",
    "phosphoenzyme_refinement",
    "component_factorization",
    "component_analytic_reference",
    "component_observation",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def save(fig, name, output, preview):
    target = output / (name + ".pdf")
    fig.savefig(
        target,
        bbox_inches="tight",
        pad_inches=0.1,
        metadata={"Creator": "MTG frozen-result renderer", "CreationDate": None, "ModDate": None},
    )
    if preview:
        fig.savefig(preview / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"file": str(target.relative_to(ROOT)), "sha256": digest(target)}


def draw_arrow_state(ax, state, center):
    positions = {3: (-0.64, 0), 1: (-0.12, 0), 4: (0.42, 0), 2: (0.96, 0)}
    for left, right, order in state["bonds"]:
        assert order == 1
        x, y = positions[left], positions[right]
        ax.plot([center + x[0], center + y[0]], [x[1], y[1]], color=GREY, lw=1.5, zorder=1)
    for atom in state["atoms"]:
        text = {"C": r"CH$_3$", "N": r"NH$_3$"}.get(atom["element"], atom["element"])
        if atom["charge"]:
            assert abs(atom["charge"]) == 1
            text += r"$^{+}$" if atom["charge"] > 0 else r"$^{-}$"
        x, y = positions[atom["atom_map"]]
        ax.text(
            center + x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color=ORANGE if atom["element"] == "H" else "black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            zorder=2,
        )


def construction(data, output, preview):
    corpus, grouping = data[REPORTS[0]], data[REPORTS[1]]
    summary, groups = corpus["summary"], grouping["summary"]
    n = summary["records"]
    assert n == groups["records"] == 1926
    sample = next(e for e in corpus["examples"] if e["id"] == 1)
    states = sample["ordered"]["states"]
    assert len(states) == 3 and sample["ordered"]["transitions"] == 2
    assert sample["one_group"]["transitions"] == 1
    counts = [
        summary["whole_state_constructed"],
        groups["records_with_multiple_valid_groupings"],
        groups["records_with_multiple_hydrogen_duet_admitted_groupings"],
        groups["records_with_possible_hydrogen_boundary_violation"],
    ]
    labels = [
        "Certified ordered histories",
        "Multiple admitted groupings",
        "Multiple groupings with H-duet filter",
        "A grouping permits H-duet excess",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), gridspec_kw={"height_ratios": [1.05, 1.2]})
    ax = axes[0]
    centers = (0, 2.45, 4.9)
    for i, (state, center) in enumerate(zip(states, centers)):
        draw_arrow_state(ax, state, center)
        ax.text(center + 0.16, 0.7, f"State {i}", ha="center", fontsize=10)
    for i in range(2):
        ax.annotate(
            "",
            (centers[i + 1] - 0.8, 0),
            (centers[i] + 1.15, 0),
            arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.6},
        )
        ax.text(centers[i] + 1.4, 0.23, f"Arrow {i + 1}", ha="center", fontsize=9)
    ax.annotate("", (5.04, -0.65), (0.16, -0.65), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.8})
    ax.text(2.6, -0.87, "One common-prestate event groups both arrows", ha="center", fontsize=9, color=GREEN)
    ax.text(2.6, -0.29, "Internal H has two bonds", ha="center", fontsize=9, color=ORANGE)
    ax.set(xlim=(-0.95, 6.2), ylim=(-1.05, 1.0))
    ax.set_title("a  Event grouping changes the internal boundary", loc="left", weight="bold", fontsize=10)
    ax.axis("off")
    ax = axes[1]
    y = np.arange(4)
    ax.barh(y, [n] * 4, color="#edf1f4", height=0.6)
    ax.barh(y, counts, color=[BLUE, GREEN, GREEN, ORANGE], height=0.6)
    for i, value in enumerate(counts):
        ax.text(value + 30, i, f"{value:,} ({100 * value / n:.1f}%)", va="center", fontsize=9)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlim=(0, 2390), xlabel=f"Records (common denominator: {n:,})")
    ax.set_xticks([0, 500, 1000, 1500, 1926])
    ax.set_title("b  Construction coverage and grouping ambiguity", loc="left", weight="bold", fontsize=10, pad=10)
    style(ax)
    fig.tight_layout(h_pad=1.2)
    plotted = {
        "records": n,
        "bars": dict(zip(labels, counts)),
        "example_id": 1,
        "example_states": states,
        "admitted_groupings": groups["total_admitted_groupings"],
        "largest_grouping_record": grouping["most_groupings"][0],
    }
    return save(fig, "article_construction", output, preview), plotted


def transient_network(data, output, preview):
    case, network, observation = (data[n] for n in ("mcsa155_case", "mcsa155_network", "observation_network_audit"))
    assert case["initial_worlds_equal"] and case["final_anchored_worlds_equal"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0, 0]
    for row, color in zip(case["candidates"], (GREY, BLUE)):
        values = row["thioester_motif_at_cuts"]
        ax.plot(range(len(values)), values, "o-", color=color, ms=4, label=f"Proposal {row['proposal']}")
    ax.set(xlabel="Completed events", ylabel="Asp–CoA thioester motif", ylim=(-0.12, 1.35))
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 1], ["Absent", "Present"])
    ax.set_title("a  Same endpoint, different history", loc="left", fontsize=10, weight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax)
    ax = axes[0, 1]
    rows = [(model, probe) for model in observation["models"] for probe in model["probes"]]
    for i, (model, probe) in enumerate(rows):
        ax.barh(i, model["full_ideals"], color="#dfe5ea", height=0.64)
        ax.barh(i, probe["reduced_ideals"], color=BLUE, height=0.4)
        ax.text(
            model["full_ideals"] + 2,
            i,
            f"{model['full_ideals']} → {probe['reduced_ideals']}",
            va="center",
            fontsize=8.5,
        )
    ax.set_yticks(range(4), ["Two sites / bond", "Two sites / motif", "Reuse / bond", "Reuse / motif"], fontsize=8.5)
    ax.set(xlim=(0, 117), xlabel="Full → observer configurations")
    ax.set_xticks([0, 25, 50, 75])
    ax.invert_yaxis()
    ax.set_title("b  Exact observation domains", loc="left", fontsize=10, weight="bold")
    style(ax)
    curves = {r["proposal"]: r["thioester_motif_at_cuts"] for r in case["candidates"]}
    motifs = curves[2]
    bond = next(r for r in case["candidates"] if r["proposal"] == 2)["covalent_Asp_CoA_at_cuts"]
    grids = []
    for ax, row in zip(axes[1], network["models"]):
        separate = row["model"] == "two_sites"
        legal = (
            {(i, j) for i in range(9) for j in range(9)}
            if separate
            else {(i, 0) for i in range(9)} | {(8, j) for j in range(1, 9)}
        )
        grid = np.zeros((9, 9), dtype=int)
        for i, j in legal:
            grid[j, i] = 3 if motifs[i] and motifs[j] else 2 if bond[i] and bond[j] else 1
        assert len(legal) == row["configuration_worlds_checked"]
        model = next(v for v in observation["models"] if v["model"] == row["model"])
        assert int((grid >= 2).sum()) == model["probes"][0]["full_positive_ideals"]
        assert int((grid == 3).sum()) == model["probes"][1]["full_positive_ideals"]
        ax.imshow(
            grid,
            origin="lower",
            cmap=ListedColormap(["white", "#dfe5ea", "#83bad9", ORANGE]),
            vmin=0,
            vmax=3,
            extent=(-0.5, 8.5, -0.5, 8.5),
            interpolation="nearest",
        )
        ax.set(xlabel="Turnover 1: completed events", ylabel="Turnover 2: completed events")
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 2, 4, 6, 8])
        letter, title = ("c", "Separate-site model") if separate else ("d", "Site-reuse model")
        schedule_label = "schedule" if row["schedule_count"] == 1 else "schedules"
        ax.set_title(
            f"{letter}  {title}\n{len(legal)} cuts; {row['schedule_count']:,} {schedule_label}",
            loc="left",
            fontsize=10,
            weight="bold",
        )
        grids.append({"model": row["model"], "configuration_codes": grid.tolist()})
        style(ax)
    legend = [Line2D([0], [0], marker="s", lw=0, color=c, ms=8) for c in ("#dfe5ea", "#83bad9", ORANGE)]
    fig.legend(
        legend,
        ["Legal cut", "Both anchored bonds", "Both thioester motifs"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(h_pad=2, w_pad=1.5, rect=(0, 0.065, 1, 1))
    plotted = {
        "candidates": [
            {
                k: r[k]
                for k in ("proposal", "native_closed_steps", "thioester_motif_at_cuts", "covalent_Asp_CoA_at_cuts")
            }
            for r in case["candidates"]
        ],
        "deposited_connections": case["observation_connections"],
        "network_models": network["models"],
        "grids": grids,
        "observer_domains": [
            {
                "model": m["model"],
                "predicate": p["predicate"],
                "full_ideals": m["full_ideals"],
                **{
                    k: p[k]
                    for k in (
                        "reduced_ideals",
                        "full_positive_ideals",
                        "reduced_positive_ideals",
                        "fiber_cardinalities",
                    )
                },
            }
            for m, p in rows
        ],
    }
    return save(fig, "article_transient_network", output, preview), plotted


def isotope_controls(data, output, preview):
    solvent, phospho = data["selenophosphate_isotope_case"], data["phosphoenzyme_refinement"]
    cases = [r for r in solvent["cases"] if r["isotopes"] == [18, 18]]
    assert len(cases) == 4
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), gridspec_kw={"height_ratios": [1.05, 1.2]})
    ax = axes[0, 0]
    ax.set_title("a  Solvent attack alternatives", loc="left", fontsize=10, weight="bold")
    ax.axis("off")
    positions = {"PA": (0, 0), "O3A": (0.7, 0), "PB": (1.4, 0), "O3B": (2.1, 0), "PG": (2.8, 0)}
    for left, right in (("PA", "O3A"), ("O3A", "PB"), ("PB", "O3B"), ("O3B", "PG")):
        x, y = positions[left], positions[right]
        ax.plot([x[0], y[0]], [0, 0], color=GREY, lw=1.8)
    for key, text in (("PA", "Pα"), ("O3A", "O"), ("PB", "Pβ"), ("O3B", "O"), ("PG", "Pγ")):
        x, y = positions[key]
        ax.text(
            x,
            y,
            text,
            va="center",
            ha="center",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    ax.text(0.7, 0.95, "H₂¹⁸O / α–β cleavage", ha="center", fontsize=9, color=BLUE)
    ax.text(2.1, -0.95, "H₂¹⁸O / β–γ cleavage", ha="center", fontsize=9, color=ORANGE)
    for key in ("PA", "PB"):
        ax.annotate(
            "", (positions[key][0], 0.12), (0.7, 0.7), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2}
        )
    for key in ("PB", "PG"):
        ax.annotate(
            "", (positions[key][0], -0.12), (2.1, -0.7), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.2}
        )
    ax.text(1.4, -1.42, "Local phosphate connectivity only", ha="center", fontsize=8, color=GREY)
    ax.set(xlim=(-0.35, 3.15), ylim=(-1.65, 1.45))
    ax = axes[0, 1]
    grid = np.zeros((2, 2), dtype=int)
    for row in cases:
        i = {"PA": 0, "PB": 1}[row["targets"]["ab"]]
        j = {"PB": 0, "PG": 1}[row["targets"]["bg"]]
        grid[i, j] = int(row["qualitative_observation_compatible"])
    assert grid.tolist() == [[0, 0], [1, 1]]
    ax.imshow(grid, cmap=ListedColormap(["#f5e0d4", "#d9eee5"]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                "¹⁸O in AMP\nContradicted" if not grid[i, j] else "No ¹⁸O in AMP\nSatisfied",
                ha="center",
                va="center",
                fontsize=8,
                color=ORANGE if not grid[i, j] else GREEN,
            )
    ax.set_xticks([0, 1], ["Pβ", "Pγ"])
    ax.set_yticks([0, 1], ["Pα", "Pβ"])
    ax.set(xlabel="β–γ attack at", ylabel="α–β attack at")
    ax.set_title("b  Published AMP-position condition", loc="left", fontsize=10, weight="bold", pad=9)
    style(ax)
    ax = axes[1, 0]
    rows = [
        r
        for r in phospho["pix"]
        if r["family"] == "mono_gamma" and r["label"] == "gamma18" and r["start_bridge"] == "O3B"
    ]
    assert len(rows) == 3
    ax.axis("off")
    ax.set_title("c  Return of captured gamma group", loc="left", fontsize=10, weight="bold", pad=9)
    ax.text(0.02, 0.81, "Originally: Pβ — ¹⁸O3B — Pγ", transform=ax.transAxes, fontsize=9, color=BLUE)
    ax.text(0.02, 0.66, "Captured Pγ–X returns to:", transform=ax.transAxes, fontsize=9)
    pix_rows = []
    for i, row in enumerate(rows):
        positive = "O3B" in row["cuts"][-1]["observation"]["beta"]["labelled_terminal"]
        pix_rows.append({"return_bridge": row["return_bridge"], "label_becomes_beta_terminal": positive})
        y = 0.48 - i * 0.17
        ax.text(0.08, y, row["return_bridge"], transform=ax.transAxes, fontsize=10)
        ax.annotate(
            "",
            (0.45, y + 0.015),
            (0.29, y + 0.015),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": GREEN if positive else GREY, "lw": 1.3},
        )
        ax.text(
            0.5,
            y,
            "¹⁸O is β-terminal" if positive else "¹⁸O stays bridge",
            transform=ax.transAxes,
            fontsize=9,
            color=GREEN if positive else GREY,
        )
    ax.text(0.02, -0.1, "Reachability; no 2/3 population inference", transform=ax.transAxes, fontsize=8, color=GREY)
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("d  Unassigned attachment control", loc="left", fontsize=10, weight="bold", pad=9)
    for y, has_port in ((0.72, True), (0.25, False)):
        ax.text(0.03, y, "P — ¹⁸O", transform=ax.transAxes, fontsize=11, color=BLUE)
        ax.text(
            0.62,
            y,
            "P — X" if has_port else "P",
            transform=ax.transAxes,
            fontsize=11,
            color=ORANGE if has_port else GREY,
        )
        ax.text(0.05, y - 0.16, "T = 1, L = 1" if has_port else "T = 2, L = 1", transform=ax.transAxes, fontsize=10)
        ax.text(
            0.61,
            y - 0.16,
            "Port retained" if has_port else "Port erased",
            transform=ax.transAxes,
            fontsize=9,
            color=ORANGE,
        )
    ax.text(0.02, -0.1, "Component sketches; other atoms omitted", transform=ax.transAxes, fontsize=8, color=GREY)
    retained = next(r for r in phospho["formal_retention_control"] if r["labels"] == [18, 18])
    assert retained["after_beta_retained_port"] == {"total": 1, "labelled": 1}
    assert retained["after_beta_erased_port"] == {"total": 2, "labelled": 1}
    fig.tight_layout(h_pad=2.1, w_pad=1.6)
    plotted = {
        "solvent_rows": [
            {
                "targets": r["targets"],
                "compatible": r["qualitative_observation_compatible"],
                "final_roles": r["final_roles"],
            }
            for r in cases
        ],
        "solvent_position_grid": grid.tolist(),
        "pix_rows": pix_rows,
        "retained_port_inventory": retained["after_beta_retained_port"],
        "erased_port_inventory": retained["after_beta_erased_port"],
        "retention_control_is_measured_intermediate": False,
        "reachability_implies_population": False,
    }
    return save(fig, "article_isotope_controls", output, preview), plotted


def scaling(data, output, preview):
    factor, analytic = data["component_factorization"], data["component_analytic_reference"]
    assert factor["all_exact_inventory_sets_agree"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), sharey=True)
    exported = []
    for retained, ax in zip((False, True), axes):
        rows = [r for r in factor["cases"] if r["retention_control"] == retained]
        xs = [r["copies"] for r in rows]
        assert xs == [1, 2, 4, 8, 16]
        for method, color, label, marker in (
            ("full_atom", GREY, "Full-atom factors", "o"),
            ("quotient", BLUE, "Invariant-block factors", "s"),
        ):
            medians, mins, maxs = [], [], []
            for row in rows:
                values = [r["complete_query_seconds"] for r in row["rounds"] if r["method"] == method]
                assert len(values) == 3
                value = median(values)
                assert value == row["complete_query_median_seconds"][method]
                medians.append(value)
                mins.append(value - min(values))
                maxs.append(max(values) - value)
                exported.append(
                    {
                        "retention_control": retained,
                        "copies": row["copies"],
                        "method": method,
                        "raw_seconds": values,
                        "median_seconds": value,
                    }
                )
            ax.errorbar(
                xs, medians, yerr=[mins, maxs], fmt=marker + "-", color=color, lw=1.5, ms=4, capsize=2, label=label
            )
        medians, mins, maxs = [], [], []
        for k in xs:
            row = next(r for r in analytic["cases"] if r["copies"] == k and r["retention_control"] == retained)
            values = [r["seconds"] for r in row["rounds"]]
            value = median(values)
            assert value == row["median_preparation_plus_two_queries_seconds"]
            medians.append(value)
            mins.append(value - min(values))
            maxs.append(max(values) - value)
            exported.append(
                {
                    "retention_control": retained,
                    "copies": k,
                    "method": "known_coefficient_linear",
                    "raw_seconds": values,
                    "median_seconds": value,
                }
            )
        ax.errorbar(
            xs,
            medians,
            yerr=[mins, maxs],
            fmt="^-",
            color=ORANGE,
            lw=1.5,
            ms=4,
            capsize=2,
            label="Known-coefficient linear",
        )
        ax.set(xscale="log", yscale="log", xlabel="Supplied independent copies", ylim=(0.00065, 1.2))
        ax.set_xticks(xs, [str(k) for k in xs])
        ax.set_title(
            "a  Gamma-capture family" if not retained else "b  Unassigned retention control",
            loc="left",
            fontsize=10,
            weight="bold",
        )
        ax.grid(axis="y", alpha=0.15, which="major")
        ax.text(0.03, 0.98, "16 copies: 5¹⁶ source ideals", transform=ax.transAxes, fontsize=9, va="top")
        style(ax)
    axes[0].set_ylabel("Preparation + query stage (seconds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(w_pad=1.3, rect=(0, 0.12, 1, 1))
    plotted = {
        "timings": exported,
        "replicates_per_point": 3,
        "whiskers": "observed minimum and maximum, not confidence intervals",
        "shared_source_construction_and_witness_execution_included": False,
        "linear_coefficient_derivation_included": False,
        "direct_rank_unknowns": data["component_observation"]["counts"]["timed_unknown_answers"],
        "complete_pipeline_advantage_established": False,
    }
    return save(fig, "article_scaling", output, preview), plotted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/mtg/fig")
    parser.add_argument("--data", type=Path, default=ROOT / "paper/mtg/results/article_figure_data.json")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    reports = {n: json.loads((ROOT / "Experiment/MTG/evidence" / (n + ".json")).read_text()) for n in REPORTS}
    assert all(r.get("status", "PASS") == "PASS" for r in reports.values())
    outputs, figures = [], {}
    for render in (construction, transient_network, isotope_controls, scaling):
        output, values = render(reports, args.output, args.preview)
        outputs.append(output)
        figures[render.__name__] = values
    record = {
        "schema": "synkit.mtg-article-figures/1",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "new_chemical_execution": False,
        "source_reports": {
            str((ROOT / "Experiment/MTG/evidence" / (n + ".json")).relative_to(ROOT)): digest(
                ROOT / "Experiment/MTG/evidence" / (n + ".json")
            )
            for n in REPORTS
        },
        "renderer_sha256": digest(Path(__file__)),
        "figures": figures,
        "outputs": outputs,
    }
    args.data.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "figures": len(outputs),
                "source_reports": len(REPORTS),
                "data": str(args.data),
                "new_chemical_execution": False,
            }
        )
    )


if __name__ == "__main__":
    main()
