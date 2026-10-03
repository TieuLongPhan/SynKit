"""Export scientific cost/registration figures from completed report artifacts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Experiment.Stereo.provenance import capture_source, finish_source, sha256  # noqa: E402


def _completed(records):
    groups = defaultdict(list)
    for value in records:
        if value["status"] == "complete":
            groups[value["id"], value["mode"]].append(value)
    return groups


def _coverage(axis, records):
    statuses = ("complete", "incomplete_input", "unsupported_input", "bounded_search", "timeout", "rss_limit",
                "worker_output_limit", "bounded_mirror", "study_total_cap", "error", "worker_error")
    counts = {mode: Counter(row["status"] for row in records if row["mode"] == mode)
              for mode in ("generators", "complete_group")}
    palette = plt.get_cmap("tab20")
    bottoms = [0, 0]
    for index, status in enumerate(statuses):
        values = [counts[mode][status] for mode in counts]
        if any(values):
            axis.bar((0, 1), values, bottom=bottoms, label=status.replace("_", " "), color=palette(index))
            bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
    axis.set(xticks=(0, 1), xticklabels=("Generators", "Full group"), ylabel="Planned runs")
    axis.legend(frameon=False, fontsize=6, loc="upper right")


def _support_growth(axis, groups):
    for family, color in (("helical_open", "#23689b"), ("helical_cyclic", "#b64b26"),
                          ("planar_chirality", "#27856b")):
        data = []
        for (name, mode), rows in groups.items():
            if mode == "generators" and name.startswith(f"support:{family}:") and name.endswith(":unique"):
                size = int(name.split(":")[2])
                vertices = rows[0]["primary_members"][0]["auxiliary_vertices"]
                data.append((size, vertices))
        if data:
            data.sort()
            axis.plot(*zip(*data), marker="o", markersize=4, label=family.replace("_", " "), color=color)
    axis.set(xlabel="Support length", ylabel="Auxiliary vertices")
    if axis.lines:
        axis.legend(frameon=False, fontsize=7)


def costs(report):
    groups = _completed(report["records"])
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ordinary, stress = [], []
    for (name, mode), rows in groups.items():
        if mode != "generators":
            continue
        vertices = statistics.median(sum(member["auxiliary_vertices"] for member in row["primary_members"])
                                     for row in rows)
        seconds = statistics.median(sum(member["search_seconds"] for member in row["primary_members"])
                                    for row in rows)
        (stress if rows[0]["stress"] else ordinary).append((vertices, seconds))
    for data, label, color in ((ordinary, "Ordinary", "#23689b"), (stress, "Stress", "#b64b26")):
        if data:
            axes[0, 0].scatter(*zip(*data), label=label, s=22, color=color, alpha=0.8)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Auxiliary vertices", ylabel="Search median (s)")
    axes[0, 0].legend(frameon=False)
    _support_growth(axes[0, 1], groups)
    paired = []
    for (name, mode), rows in groups.items():
        other = groups.get((name, "complete_group"))
        if mode == "generators" and other:
            generator = statistics.median(sum(member["witness_seconds"] for member in row["primary_members"])
                                          for row in rows)
            full = statistics.median(sum(member["witness_seconds"] for member in row["primary_members"])
                                     for row in other)
            paired.append((generator, full))
    if paired:
        axes[1, 0].scatter(*zip(*paired), s=18, color="#23689b", alpha=0.75)
        lower = max(1e-7, min(min(pair) for pair in paired))
        upper = max(max(pair) for pair in paired)
        axes[1, 0].plot((lower, upper), (lower, upper), color="gray", linestyle="--", linewidth=0.8)
    axes[1, 0].set(xscale="log", yscale="log", xlabel="Generator output median (s)",
                   ylabel="Full-group output median (s)")
    _coverage(axes[1, 1], report["records"])
    for index, axis in enumerate(axes.flat):
        axis.set_title(chr(ord("A") + index), loc="left", fontweight="bold", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def registration(report):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for method, label, color in (("pairwise", "Pairwise representatives", "#23689b"),
                                 ("certificates", "Certificates + exact indexing", "#b64b26")):
        sizes, medians = [], []
        for row in report["summary"]:
            value = row[method]["total_seconds_completed_only"]["median"]
            if value is not None:
                sizes.append(row["size"])
                medians.append(value)
        axes[0].plot(sizes, medians, marker="o", markersize=4, label=label, color=color)
        complete = [row for row in report["records"] if row["method"] == method and row["status"] == "complete"]
        axes[0].scatter([row["size"] for row in complete], [row["total_seconds"] for row in complete],
                        s=10, color=color, alpha=0.35)
    rows = [row for row in report["summary"] if row["pairwise_over_certificates_median_ratio"] is not None]
    axes[1].plot([row["size"] for row in rows],
                 [row["pairwise_over_certificates_median_ratio"] for row in rows], marker="o", color="#27856b")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set(xscale="log", yscale="log", xlabel="Collection records", ylabel="Total median (s)")
    axes[1].set(xscale="log", xlabel="Collection records", ylabel="Pairwise / certificate median")
    axes[0].legend(frameon=False, fontsize=7)
    for index, axis in enumerate(axes):
        axis.set_title(chr(ord("A") + index), loc="left", fontweight="bold", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def resources(report):
    """Display native memory, output size and phase costs without censored values."""
    groups = _completed(report["records"])
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    for (_name, mode), rows in groups.items():
        if mode != "generators":
            continue
        vertices = sum(member["auxiliary_vertices"] for member in rows[0]["primary_members"])
        color = "#b64b26" if rows[0]["stress"] else "#23689b"
        axes[0, 0].scatter(vertices, statistics.median(r["worker_peak_rss_mib"] for r in rows),
                           s=20, color=color, alpha=0.7)
        axes[0, 1].scatter(vertices, statistics.median(r["certificate_bytes"] for r in rows),
                           s=20, color=color, alpha=0.7)
        cpu = statistics.median(r["worker_operation_cpu_seconds"] for r in rows)
        wall = statistics.median(r["worker_operation_seconds"] for r in rows)
        axes[1, 1].scatter(cpu, wall, s=20, color=color, alpha=0.7)
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
    labels = ("Expand", "Snapshot", "Search", "Witness", "Pack", "Mirror", "Pairwise", "Report")
    axes[1, 0].boxplot(distributions, tick_labels=labels, showfliers=False, widths=0.6)
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].set(yscale="log", ylabel="Completed phase cost (s)")
    axes[0, 0].set(xscale="log", xlabel="Auxiliary vertices", ylabel="Worker peak RSS (MiB)")
    axes[0, 1].set(xscale="log", yscale="log", xlabel="Auxiliary vertices", ylabel="Full certificate bytes")
    lower = min(r["worker_operation_cpu_seconds"] for r in complete)
    upper = max(r["worker_operation_seconds"] for r in complete)
    axes[1, 1].plot((lower, upper), (lower, upper), color="gray", linestyle="--", linewidth=0.8)
    axes[1, 1].set(xscale="log", yscale="log", xlabel="Operation CPU median (s)",
                   ylabel="Operation wall median (s)")
    for index, axis in enumerate(axes.flat):
        axis.set_title(chr(ord("A") + index), loc="left", fontweight="bold", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def main():
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
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    outputs = []
    figures = (("controlled_stereo_costs", costs, performance),
               ("controlled_stereo_resources", resources, performance),
               ("curated_stereo_registration", registration, registered))
    for name, function, report in figures:
        figure = function(report)
        for extension in ("pdf", "svg"):
            path = args.output_dir / (name + "." + extension)
            figure.savefig(path)
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
