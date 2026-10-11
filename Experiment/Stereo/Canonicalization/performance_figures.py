"""Export scientific cost/registration figures from completed report artifacts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# Figure style shared with the manuscript's TikZ graphics (fig/thesis_tokens.tex).
NAVY, BLUE, TEAL, GOLD, VERMILION = "#17324D", "#0072B2", "#009E73", "#E69F00", "#D55E00"
TEXT, MUTED, RULE = "#1A1A1A", "#666666", "#D9D9D9"
# 31 picas is the manuscript's physical text measure. Keeping the PDF canvas
# at that width preserves the declared type sizes when it is included at full width.
FIGURE_WIDTH = 31 * 12 / 72.27
STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.labelspacing": 0.35,
    "axes.labelcolor": TEXT, "axes.edgecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "axes.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "grid.color": RULE, "grid.linewidth": 0.55, "lines.linewidth": 1.1, "lines.markersize": 4,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "none",
}


def _pin_font():
    """Pin Matplotlib's bundled TrueType serif for portable, correctly embedded PDFs.

    This is the thesis module's serif fallback. Installed Latin Modern / CMU
    OpenType CFF files can produce PDF font-type mismatch warnings with Type 42.
    """
    from matplotlib import font_manager

    bundle = Path(matplotlib.get_data_path()) / "fonts/ttf"
    entries = font_manager.fontManager.ttflist
    font_manager.fontManager.ttflist = [
        font for font in entries if font.name != "DejaVu Serif" or Path(font.fname).is_relative_to(bundle)]
    font_manager._get_font.cache_clear()
    font_manager.fontManager._findfont_cached.cache_clear()


def _finish(axes, titles=()):
    """Use compact scientific panel keys and restrained axes at printed size."""
    for index, axis in enumerate(axes):
        axis.grid(axis="y", which="major", zorder=0)
        axis.text(0.0, 1.08, chr(ord("A") + index), transform=axis.transAxes, fontsize=9,
                  fontweight="bold", color=TEXT, va="center", ha="left", clip_on=False, zorder=30)
        if titles:
            axis.annotate(titles[index], (0.0, 1.08), xycoords="axes fraction", xytext=(12, 0),
                          textcoords="offset points", fontsize=8.6, fontweight="bold",
                          va="center", annotation_clip=False)


def _completed(records):
    groups = defaultdict(list)
    for value in records:
        if value["status"] == "complete":
            groups[value["id"], value["mode"]].append(value)
    return groups


def _support_growth(axis, groups):
    for family, label, color, marker in (("helical_cyclic", "cyclic path", VERMILION, "s"),
                                         ("planar_chirality", "plane", TEAL, "D"),
                                         ("helical_open", "open path", BLUE, "o")):
        data = []
        for (name, mode), rows in groups.items():
            if mode == "generators" and name.startswith(f"support:{family}:") and name.endswith(":unique"):
                size = int(name.split(":")[2])
                vertices = rows[0]["primary_members"][0]["auxiliary_vertices"]
                data.append((size, vertices))
        if data:
            data.sort()
            axis.plot(*zip(*data), marker=marker, label=label, color=color)
    axis.set(xlabel="Support length", ylabel="Auxiliary vertices")
    if axis.lines:
        axis.legend(loc="upper left")


def costs(report, registered=None):
    """Relate expansion, search effort and matched symmetry-output overhead."""
    groups = _completed(report["records"])
    figure = plt.figure(figsize=(FIGURE_WIDTH, 4.9))
    axes = [figure.add_axes((0.13, 0.62, 0.31, 0.27)),
            figure.add_axes((0.64, 0.62, 0.30, 0.27)),
            figure.add_axes((0.13, 0.13, 0.80, 0.27))]
    _support_growth(axes[0], groups)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks((5, 9, 17, 33), ("5", "9", "17", "33"))
    axes[0].minorticks_off()
    axes[0].set_xlabel("Support length")
    search = []
    for (name, mode), rows in groups.items():
        if mode != "generators":
            continue
        vertices = statistics.median(sum(member["auxiliary_vertices"] for member in row["primary_members"])
                                     for row in rows)
        seconds = statistics.median(sum(member["search_seconds"] for member in row["primary_members"])
                                    for row in rows)
        nodes = statistics.median(sum(member["search_nodes"] for member in row["primary_members"])
                                  for row in rows)
        search.append((vertices, seconds, nodes))
    ordered = LinearSegmentedColormap.from_list("thesis_search_nodes", ("#CDDCEF", "#89B2DC", "#3C7AB8", "#1C3767"))
    observed = axes[1].scatter([v for v, _, _ in search], [t for _, t, _ in search],
                               c=[n for _, _, n in search], cmap=ordered,
                               norm=LogNorm(vmin=1, vmax=max(n for _, _, n in search)),
                               s=17, edgecolor=NAVY, linewidth=0.25, zorder=3)
    axes[1].set(xscale="log", yscale="log", xlabel="Auxiliary vertices", ylabel="Search time (s)")
    coloraxis = figure.add_axes((0.64, 0.49, 0.30, 0.015))
    colorbar = figure.colorbar(observed, cax=coloraxis, orientation="horizontal", ticks=(1, 10, 100))
    colorbar.ax.set_xticklabels(("1", "10", "100"))
    colorbar.ax.tick_params(labelsize=7.2, length=2)
    colorbar.set_label("Search nodes", size=7.2, labelpad=1)
    paired = []
    for (name, mode), rows in groups.items():
        other = groups.get((name, "complete_group"))
        if mode == "generators" and other:
            generator = statistics.median(sum(member["witness_seconds"] for member in row["primary_members"])
                                          for row in rows)
            full = statistics.median(sum(member["witness_seconds"] for member in row["primary_members"])
                                     for row in other)
            maps = statistics.median(sum(member["returned_witnesses"] for member in row["primary_members"])
                                     for row in other)
            if generator <= 0 or full <= 0:
                raise ValueError("Matched output times must be positive.")
            paired.append((name, maps, full / generator))
    axes[2].scatter([n for _, n, _ in paired], [r for _, _, r in paired], s=16,
                    color=BLUE, alpha=0.62, linewidth=0.25, edgecolor=NAVY, zorder=3)
    axes[2].axhline(1, color=MUTED, linestyle="--", linewidth=0.8)
    axes[2].set(xscale="log", yscale="log", xlabel="Full-group witnesses", ylabel="Output time ratio")
    axes[2].set_xlim(0.75, 480)
    axes[2].set_ylim(0.4, 100)
    for name, maps, ratio in paired:
        if name == "components:tetrahedral:2":
            axes[2].annotate("2 tetrahedral components", (maps, ratio), xytext=(-6, 9),
                             textcoords="offset points", ha="right", fontsize=7.2,
                             arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5})
        elif name == "support:helical_cyclic:33:symmetric":
            axes[2].annotate("cyclic path, 33", (maps, ratio), xytext=(5, 5),
                             textcoords="offset points", fontsize=7.2)
    _finish(axes, ("Expansion", "Search", "Symmetry output"))
    return figure


def _registration(axis, report):
    """Median total time of both grouping methods, with the individual repetitions."""
    for method, label, color, marker in (("certificates", "certificates", VERMILION, "s"),
                                         ("pairwise", "pairwise", BLUE, "o")):
        sizes, medians = [], []
        for row in report["summary"]:
            value = row[method]["total_seconds_completed_only"]["median"]
            if value is not None:
                sizes.append(row["size"])
                medians.append(value)
        axis.plot(sizes, medians, marker=marker, label=label, color=color, zorder=3)
        complete = [row for row in report["records"] if row["method"] == method and row["status"] == "complete"]
        axis.scatter([row["size"] for row in complete], [row["total_seconds"] for row in complete],
                     s=8, color=color, marker=marker, alpha=0.35, linewidth=0, zorder=2)
    sizes = [row["size"] for row in report["summary"]]
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, [str(size) for size in sizes])
    axis.minorticks_off()
    axis.set(yscale="log", xlabel="Records in collection", ylabel="Median registration time (s)")
    axis.legend(loc="upper left")


def resources(report):
    """Display native memory, output size and phase costs without censored values."""
    groups = _completed(report["records"])
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.9), constrained_layout=True)
    for (_name, mode), rows in groups.items():
        if mode != "generators":
            continue
        vertices = sum(member["auxiliary_vertices"] for member in rows[0]["primary_members"])
        color, marker = (VERMILION, "s") if rows[0]["stress"] else (BLUE, "o")
        look = dict(s=14, color=color, marker=marker, alpha=0.75, linewidth=0, zorder=3)
        axes[0, 0].scatter(vertices, statistics.median(r["worker_peak_rss_mib"] for r in rows), **look)
        axes[0, 1].scatter(vertices, statistics.median(r["certificate_bytes"] for r in rows), **look)
        cpu = statistics.median(r["worker_operation_cpu_seconds"] for r in rows)
        wall = statistics.median(r["worker_operation_seconds"] for r in rows)
        axes[1, 1].scatter(cpu, wall, **look)
    complete = [r for r in report["records"] if r["mode"] == "generators" and r["status"] == "complete"]
    fields = ("expansion_seconds", "snapshot_seconds", "search_seconds", "witness_seconds",
              "packaging_seconds", "mirror_seconds", "pairwise_seconds", "report_serialization_seconds")
    distributions = []
    for field in fields:
        values = []
        for row in complete:
            value = row.get(field)
            if value is None and field in row["primary_members"][0]:
                value = sum(member[field] for member in row["primary_members"])
            if value is not None and value > 0:
                values.append(value)
        distributions.append(values)
    labels = ("Exp.", "Snap.", "Search", "Wit.", "Pack", "Mirror", "Pair", "Report")
    axes[1, 0].boxplot(distributions, tick_labels=labels, showfliers=False, widths=0.55,
                       boxprops={"color": NAVY, "linewidth": 0.7}, whiskerprops={"color": NAVY, "linewidth": 0.7},
                       capprops={"color": NAVY, "linewidth": 0.7}, medianprops={"color": VERMILION, "linewidth": 1.1})
    axes[1, 0].tick_params(axis="x", rotation=45)
    for position, values in enumerate(distributions, 1):
        offsets = [((index % 11) - 5) * 0.018 for index in range(len(values))]
        axes[1, 0].scatter([position + offset for offset in offsets], values, s=2.0,
                           color=BLUE, alpha=0.18, edgecolor="none", zorder=1)
    axes[1, 0].set(yscale="log", ylabel="Phase time, completed trials (s)")
    axes[0, 0].set(xscale="log", xlabel="Auxiliary vertices", ylabel="Worker peak RSS (MiB)")
    axes[0, 1].set(xscale="log", yscale="log", xlabel="Auxiliary vertices", ylabel="Certificate size (bytes)")
    lower = min(r["worker_operation_cpu_seconds"] for r in complete)
    upper = max(r["worker_operation_seconds"] for r in complete)
    axes[1, 1].plot((lower, upper), (lower, upper), color=MUTED, linestyle="--", linewidth=0.8)
    axes[1, 1].set(xscale="log", yscale="log", xlabel="CPU time, median (s)",
                   ylabel="Wall time, median (s)")
    axes[0, 0].scatter([], [], s=14, color=BLUE, marker="o", label="ordinary")
    axes[0, 0].scatter([], [], s=14, color=VERMILION, marker="s", label="stress")
    axes[0, 0].legend(loc="upper left", handletextpad=0.2)
    _finish(axes.flat, ("Peak memory", "Output size", "Phase timing", "CPU and wall time"))
    return figure


def main():
    # Plot functions can also be loaded from the editable figure archive,
    # without importing the SynKit package or its experiment provenance API.
    from Experiment.Stereo.provenance import capture_source, finish_source, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = capture_source()
    performance = json.loads(args.performance.read_text())
    registered = json.loads(args.registration.read_text())
    if not performance["study_valid"] or not registered["passed"]:
        raise ValueError("Failed correctness/registration reports cannot supply publication plots.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _pin_font()
    plt.rcParams.update(STYLE)
    outputs = []
    figures = (("controlled_stereo_costs", lambda report: costs(report, registered), performance),
               ("controlled_stereo_resources", resources, performance))
    for name, function, report in figures:
        figure = function(report)
        for extension in ("pdf", "svg"):
            path = args.output_dir / (name + "." + extension)
            metadata = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None}
            figure.savefig(path, metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            outputs.append({"path": str(path), "sha256": sha256(path)})
        plt.close(figure)
    manifest = {"schema": "synkit.cost-figure-evidence/1", "provenance": finish_source(source),
                "inputs": [{"path": str(path), "sha256": sha256(path)}
                           for path in (args.performance, args.registration)], "outputs": outputs,
                "scope": "completed-only timing panels with all planned-run outcomes reported separately"}
    (args.output_dir / "cost_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
