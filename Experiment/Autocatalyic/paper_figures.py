"""Draw the manuscript data figures from the stored pilot results.

The script only reads ``results/`` and writes ``paper/autocatalytic/figures``.
Run ``run.py`` first, then from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/paper_figures.py
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/synkit-autocatalytic-mpl")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
FIGURES = HERE.parents[1]/"paper"/"autocatalytic"/"figures"

# Okabe--Ito roles shared with the other SynKit manuscripts.
BLUE, VERMILION, TEAL, GOLD = "#0072B2", "#D55E00", "#009E73", "#E69F00"
NAVY, MUTED, TEXT = "#17324D", "#666666", "#1A1A1A"
WIDTH = 6.3  # text width of the manuscript in inches


def style():
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.edgecolor": TEXT, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "lines.linewidth": 1.4,
        "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def read_csv(name):
    with (RESULTS/name).open() as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        item = {}
        for key, value in row.items():
            try:
                item[key] = float(value)
            except ValueError:
                item[key] = {"True": True, "False": False}.get(value, value)
        out.append(item)
    return out


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=8, x=0.12)
    ax.text(0.0, 1.045, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, fontweight="bold", color="white",
            bbox={"boxstyle": "circle,pad=0.22", "fc": NAVY, "ec": "none"})


def zero_line(ax):
    ax.axhline(0, color=MUTED, lw=0.6, zorder=0)


def log_axis_with_zero(ax, values, label):
    """Place the exact zero limit one decade below the smallest positive value."""
    positive = sorted(v for v in values if v > 0)
    stand_in = positive[0]/100
    ax.set_xscale("log")
    ticks = [stand_in]+positive
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"]+[rf"$10^{{{round(math.log10(v))}}}$" for v in positive])
    ax.minorticks_off()
    ax.set_xlabel(label)
    ax.axvspan(stand_in/3, positive[0]/10, color="0.93", lw=0, zorder=0)
    ax.set_xlim(stand_in/3, positive[-1]*3)
    return lambda v: stand_in if v == 0 else v


def core_markers(ax, x, y, cores, color):
    """Filled markers where a productive core exists, open markers elsewhere."""
    ax.plot(x, y, "-", color=color, lw=1.0, zorder=2)
    for xi, yi, n in zip(x, y, cores):
        ax.plot([xi], [yi], "o", ms=5.5, mec=color, mew=1.2,
                mfc=color if n else "white", zorder=3)


def core_legend(ax, color, loc, some="productive core present"):
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5.5, mec=color, mew=1.2, mfc="white",
                          label="no productive core"),
               plt.Line2D([], [], marker="o", ls="", ms=5.5, mec=color, mew=1.2, mfc=color,
                          label=some)]
    ax.legend(handles=handles, loc=loc, handletextpad=0.3)


def figure_controls():
    rows = read_csv("control_b_trajectory.csv")
    controls = json.loads((RESULTS/"controls.json").read_text())
    reverse = controls["reverse_channel_continuation"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), constrained_layout=True)
    time = [r["time"] for r in rows]
    branches = (("plus", BLUE, "L-rich"), ("minus", VERMILION, "D-rich"))
    for sign, color, label in branches:
        ee = [(r[f"L_{sign}"]-r[f"D_{sign}"])/(r[f"L_{sign}"]+r[f"D_{sign}"]) for r in rows]
        axes[0].plot(time, ee, color=color)
        axes[0].text(292, ee[-1]*0.80, label, color=color, ha="right", va="center", fontsize=7.5)
    axes[0].axhline(0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    axes[0].text(8, 0.035, "racemate", color=MUTED, fontsize=7.5, va="bottom")
    axes[0].set(xlim=(0, 300), ylim=(-0.6, 0.6), xlabel="time",
                ylabel=r"monomer excess $\mathrm{ee}_m$")
    panel(axes[0], "A", "Growth of a small excess")
    ax = axes[1]
    ax.plot([0, 2], [0, 2], color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.text(0.36, 0.30, r"$L=D$", color=MUTED, fontsize=7.5, rotation=45, ha="center",
            va="top", rotation_mode="anchor")
    for sign, color, label in branches:
        ax.plot([r[f"L_{sign}"] for r in rows], [r[f"D_{sign}"] for r in rows], color=color)
        ax.plot([rows[-1][f"L_{sign}"]], [rows[-1][f"D_{sign}"]], "o", color=color, ms=5)
    ax.plot([1], [1], "o", ms=5, mfc="white", mec=TEXT, mew=1.0, zorder=5)
    ax.annotate("unstable racemate", (1.0, 1.04), xytext=(0.98, 1.86), fontsize=7.5,
                color=TEXT, arrowprops={"arrowstyle": "-", "lw": 0.5, "color": MUTED},
                va="center", ha="left")
    ax.text(rows[-1]["L_plus"], rows[-1]["D_plus"]-0.16, "stable", color=BLUE, fontsize=7.5,
            ha="center", va="top")
    ax.text(rows[-1]["L_minus"]-0.10, rows[-1]["D_minus"], "stable", color=VERMILION,
            fontsize=7.5, ha="right", va="center")
    ax.set(xlim=(0, 2), ylim=(0, 2), xlabel=r"$L$", ylabel=r"$D$", aspect="equal",
           xticks=[0, 1, 2], yticks=[0, 1, 2])
    panel(ax, "B", "Two mirror endpoints")
    eps = [r["epsilon"] for r in reverse]
    place = log_axis_with_zero(axes[2], eps, r"reverse coefficient $\epsilon$")
    core_markers(axes[2], [place(e) for e in eps], [r["racemate"]["alpha_odd"] for r in reverse],
                 [len(r["structural"]["cores"]) for r in reverse], BLUE)
    zero_line(axes[2])
    axes[2].set(ylabel=r"odd growth rate $\alpha_-$", ylim=(-0.01, 0.125))
    core_legend(axes[2], BLUE, "center right")
    axes[2].get_legend().set_bbox_to_anchor((1.02, 0.33))
    panel(axes[2], "C", "Adding reverse reactions")
    fig.savefig(FIGURES/"fig_controls.pdf")
    plt.close(fig)


def figure_thresholds():
    summary = json.loads((RESULTS/"summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), constrained_layout=True)
    labels = {"constructive": r"preference ratio $k_1/k_2$",
              "destructive": r"preference ratio $k_2/k_1$"}
    for ax, letter, mechanism in zip(axes, "AB", labels):
        rows = [r for r in read_csv(f"{mechanism}_racemic_scan.csv")
                if r["positive_monomers"] and r["alpha_even"] < -1e-8]
        rows.sort(key=lambda r: r["ratio"])
        ratio = [r["ratio"] for r in rows]
        odd, even = [r["alpha_odd"] for r in rows], [r["alpha_even"] for r in rows]
        crossing = find_threshold(summary, mechanism)
        ax.axvspan(crossing, ratio[-1], color=BLUE, alpha=0.07, lw=0)
        ax.plot(ratio, odd, color=BLUE)
        ax.plot(ratio, even, color=GOLD)
        zero_line(ax)
        ax.axvline(crossing, color=MUTED, ls=(0, (4, 3)), lw=0.7)
        top = max(odd)*1.25
        ax.set(xlabel=labels[mechanism], ylabel="spectral abscissa",
               xlim=(ratio[0] if mechanism == "constructive" else 0, ratio[-1]),
               ylim=(-0.115, top))
        left = mechanism == "destructive"
        ax.text(crossing+(-0.02 if left else 0.02)*ratio[-1], top*0.97,
                f"threshold {crossing:.2f}", fontsize=7.5, color=MUTED, va="top",
                ha="right" if left else "left")
        if mechanism == "constructive":
            ax.text(ratio[-1]*0.98, odd[-1]*0.72, r"odd sector $\alpha_-$", color=BLUE,
                    ha="right", va="top", fontsize=7.5)
        else:
            ax.text(0.3, -0.012, r"odd sector $\alpha_-$", color=BLUE, ha="left",
                    va="center", fontsize=7.5)
        ax.text(ratio[-1]*0.98, -0.093, r"even sector $\alpha_+$", color=GOLD, ha="right",
                va="bottom", fontsize=7.5)
        middle = 0.5*(crossing+ratio[-1])
        ax.text(middle, -0.045, "racemate unstable\nto a chiral seed", ha="center",
                va="center", fontsize=7.5, color=NAVY)
        if mechanism == "destructive":
            ax.text(0.62*crossing, -0.072, "racemate\nstable", ha="center", va="center",
                    fontsize=7.5, color=MUTED)
        panel(ax, letter, mechanism.capitalize())
    fig.savefig(FIGURES/"fig_thresholds.pdf")
    plt.close(fig)


def find_threshold(node, mechanism):
    """Locate the stored critical ratio of one mechanism in the summary."""
    if isinstance(node, dict):
        if node.get("mechanism") == mechanism and "critical_ratio" in node:
            return node["critical_ratio"]
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = find_threshold(child, mechanism)
        if found is not None:
            return found
    return None


def figure_supports():
    data = json.loads((RESULTS/"sign_structure.json").read_text())["supports"]
    names = ["matched", "mismatched", "homoligation", "heteroligation"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.5), constrained_layout=True, sharey=True,
                             gridspec_kw={"width_ratios": [0.62, 1.5, 1.5]})
    rows = data["constructive"]
    y = list(range(len(rows)))[::-1]
    grid = axes[0]
    for yi, row in zip(y, rows):
        if yi % 2 == 0:
            for ax in axes:
                ax.axhspan(yi-0.5, yi+0.5, color="0.955", lw=0, zorder=0)
        for xi, bit in enumerate(row["bits"]):
            grid.plot([xi], [yi], "o", ms=5, color=NAVY if bit == "1" else "0.82", zorder=3)
        on = [xi for xi, bit in enumerate(row["bits"]) if bit == "1"]
        if len(on) > 1:
            grid.plot([on[0], on[-1]], [yi, yi], color=NAVY, lw=1.0, zorder=2)
    grid.set(xlim=(-0.6, 3.6), ylim=(-0.6, len(rows)+0.35), yticks=[])
    grid.set_xticks(range(4))
    grid.set_xticklabels(names, rotation=90, fontsize=7.5)
    grid.xaxis.tick_top()
    grid.tick_params(length=0)
    for spine in grid.spines.values():
        spine.set_visible(False)
    for ax, letter, mechanism in zip(axes[1:], "AB", ("constructive", "destructive")):
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        for yi, row in zip(y, data[mechanism]):
            rac, core = row["racemate"], row["productive_core_count"] > 0
            if rac is None:
                ax.plot([-0.118], [yi], "o", ms=5.5, mec="0.6", mew=1.2,
                        mfc="0.6" if core else "white", zorder=3)
                ax.text(-0.105, yi, "no active even-stable racemate", fontsize=6.5,
                        color="0.5", va="center")
                continue
            color = BLUE if row["passes_phenotype"] else "0.45"
            ax.plot([0, rac["alpha_odd"]], [yi, yi], color=color, lw=1.0, zorder=2)
            ax.plot([rac["alpha_odd"]], [yi], "o", ms=5.5, mec=color, mew=1.2,
                    mfc=color if core else "white", zorder=3)
        ax.set(xlim=(-0.13, 0.19), xlabel=r"odd growth rate $\alpha_-$",
               xticks=[-0.1, 0, 0.1])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.text(0.008, len(rows)-0.3, r"racemate unstable $\rightarrow$", color=BLUE,
                ha="left", va="center", fontsize=7)
        ax.text(-0.008, len(rows)-0.3, r"$\leftarrow$ stable", color="0.45", ha="right",
                va="center", fontsize=7)
        ax.set_title(mechanism.capitalize(), loc="left", pad=8, x=0.09)
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, fontweight="bold", color="white",
                bbox={"boxstyle": "circle,pad=0.22", "fc": NAVY, "ec": "none"})
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5.5, mec="0.3", mew=1.2, mfc="0.3",
                          label="contains a productive core"),
               plt.Line2D([], [], marker="o", ls="", ms=5.5, mec="0.3", mew=1.2, mfc="white",
                          label="no productive core"),
               plt.Line2D([], [], marker="o", ls="-", lw=1.0, ms=5.5, color=BLUE,
                          label="chiral phenotype")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, columnspacing=1.6,
               handletextpad=0.3)
    # Give the membership grid the same vertical extent as the data panels.
    fig.get_layout_engine().set(h_pad=0.09)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    reference, box = axes[1].get_position(), grid.get_position()
    grid.set_position([box.x0, reference.y0, box.width, reference.height])
    fig.savefig(FIGURES/"fig_supports.pdf", bbox_inches=None)
    plt.close(fig)


def figure_archive():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), constrained_layout=True)
    labels = {"constructive": r"preference ratio $k_1/k_2$",
              "destructive": r"preference ratio $k_2/k_1$"}
    for ax, letter, mechanism in zip(axes, "AB", labels):
        rows = read_csv(f"archive_sweep_{mechanism}.csv")
        ratio = [r["ratio"] for r in rows]
        ax.plot(ratio, [r["model_ee"] for r in rows], color=BLUE, lw=1.4, zorder=2,
                label="this study")
        ax.plot(ratio[::4], [r["archived_ee"] for r in rows][::4], "o", ms=3.2, mfc="white",
                mec=VERMILION, mew=0.9, zorder=3, label="archived program")
        ax.set(xlabel=labels[mechanism], ylabel=r"monomer excess $\mathrm{ee}_m$",
               ylim=(-0.05, 1.08), xlim=(0, ratio[-1]))
        panel(ax, letter, mechanism.capitalize())
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.92))
    fig.savefig(FIGURES/"fig_archive.pdf")
    plt.close(fig)


def figure_domain():
    stored = json.loads((RESULTS/"parameter_domain.json").read_text())
    scans = (("records", BLUE, "six published coefficients"),
             ("extended_records", TEAL, "plus background, hydrolysis, racemization"))
    short = ("match", "mismatch", "homo", "hetero")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), constrained_layout=True)
    for ax, letter, mechanism in zip(axes, "AB", ("constructive", "destructive")):
        keys = sorted(stored["records"][mechanism]["unstable_by_support"],
                      key=lambda k: (k.count("1"), k))
        y = np.arange(len(keys))[::-1].astype(float)
        for offset, (name, color, _) in zip((0.19, -0.19), scans):
            record = stored[name][mechanism]
            total = record["samples"]
            unstable = [100*record["unstable_by_support"].get(k, 0)/total for k in keys]
            minimal = [100*record["minimal_by_support"].get(k, 0)/total for k in keys]
            ax.barh(y+offset, unstable, height=0.34, color=color, alpha=0.28, lw=0)
            ax.barh(y+offset, minimal, height=0.34, color=color, lw=0)
            for yi, value in zip(y+offset, unstable):
                ax.text(value+1.2, yi, f"{value:.0f}%", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([" + ".join(n for n, bit in zip(short, k) if bit == "1")
                            for k in keys], fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set(xlim=(0, 78), xlabel="sampled rate sets (%)")
        ax.set_title(mechanism.capitalize(), loc="left", pad=8, x=0.17)
        ax.text(0.0, 1.045, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, fontweight="bold", color="white",
                bbox={"boxstyle": "circle,pad=0.22", "fc": NAVY, "ec": "none"})
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, lw=0, label=label)
               for _, color, label in scans]
    handles += [plt.Rectangle((0, 0), 1, 1, color="0.35", alpha=0.28, lw=0,
                              label="racemate chirally unstable"),
                plt.Rectangle((0, 0), 1, 1, color="0.35", lw=0,
                              label="and no smaller support is")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, handlelength=1.2,
               columnspacing=2.0)
    fig.savefig(FIGURES/"fig_domain.pdf")
    plt.close(fig)


def figure_reverse_limit():
    rows = read_csv("destructive_reverse_limit.csv")
    beta = [r["kmon"] for r in rows]
    cores = [r["core_count"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), constrained_layout=True)
    label = r"reverse conversion factor $\beta$"
    place = log_axis_with_zero(axes[0], beta, label)
    x = [place(b) for b in beta]
    core_markers(axes[0], x, [r["alpha_odd"] for r in rows], cores, BLUE)
    core_markers(axes[0], x, [r["alpha_endpoint"] for r in rows], cores, VERMILION)
    zero_line(axes[0])
    axes[0].set(ylabel="spectral abscissa", ylim=(-0.055, 0.045))
    axes[0].text(x[2], 0.0285, r"racemate, odd sector $\alpha_-$ (unstable)", color=BLUE,
                 fontsize=7.5, ha="center", va="bottom")
    axes[0].text(x[2], -0.046, "chiral endpoint (stable)", color=VERMILION, fontsize=7.5,
                 ha="center", va="top")
    core_legend(axes[0], "0.3", "center left", some="three productive cores")
    axes[0].get_legend().set_bbox_to_anchor((0.12, 0.55))
    panel(axes[0], "A", "Stability")
    log_axis_with_zero(axes[1], beta, label)
    core_markers(axes[1], x, [r["ee_monomer"] for r in rows], cores, BLUE)
    axes[1].set(ylabel=r"endpoint excess $|\mathrm{ee}_m|$", ylim=(0, 0.7))
    panel(axes[1], "B", "Enrichment")
    fig.savefig(FIGURES/"fig_reverse_limit.pdf")
    plt.close(fig)


def figure_racemization():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), constrained_layout=True)
    ends = {}
    for mechanism, color, marker in (("constructive", BLUE, "o"),
                                     ("destructive", VERMILION, "s")):
        rows = read_csv(f"{mechanism}_racemization.csv")
        x = [r["kr_fraction"] for r in rows]
        axes[0].plot(x, [abs(r["ee_monomer"]) for r in rows], marker+"-", color=color)
        axes[1].plot(x, [r["peptide_residue_outflow"] for r in rows], marker+"-", color=color)
        ends[mechanism] = (color, abs(rows[1]["ee_monomer"]), rows[1]["peptide_residue_outflow"])
    axes[0].set(ylabel=r"endpoint excess $|\mathrm{ee}_m|$", ylim=(-0.03, 0.9))
    axes[1].set(ylabel="peptide residue outflow", ylim=(0, 0.085))
    for ax, letter, title in zip(axes, "AB", ("Enrichment", "Output")):
        ax.axvspan(1, 1.2, color="0.93", lw=0, zorder=0)
        ax.axvline(1, color=MUTED, ls=(0, (4, 3)), lw=0.7)
        ax.set(xlim=(-0.04, 1.2), xlabel=r"racemization $k_r/k_r^{\mathrm{crit}}$")
        panel(ax, letter, title)
    axes[0].text(1.1, 0.45, "racemate\nstable", rotation=90, ha="center", va="center",
                 fontsize=7.5, color=MUTED)
    axes[0].text(0.29, ends["constructive"][1]+0.05, "constructive", color=BLUE, fontsize=7.5)
    axes[0].text(0.29, ends["destructive"][1]+0.05, "destructive", color=VERMILION, fontsize=7.5)
    axes[1].text(0.29, ends["constructive"][2]+0.004, "constructive", color=BLUE, fontsize=7.5)
    axes[1].text(0.29, ends["destructive"][2]+0.004, "destructive", color=VERMILION,
                 fontsize=7.5)
    fig.savefig(FIGURES/"fig_racemization.pdf")
    plt.close(fig)


def main():
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_controls()
    figure_thresholds()
    figure_supports()
    figure_archive()
    figure_domain()
    figure_reverse_limit()
    figure_racemization()


if __name__ == "__main__":
    main()
