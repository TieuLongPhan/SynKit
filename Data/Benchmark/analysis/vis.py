import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools
from matplotlib.axes import Axes
from statannotations.Annotator import Annotator
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# -------------------------------------------------------------
# GLOBAL STYLE (LaTeX + serif)
# -------------------------------------------------------------
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    },
)

# -------------------------------------------------------------
# TABLEAU-10 (colour-blind safe)
# -------------------------------------------------------------
TABLEAU10 = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def plot_violin(
    df: pd.DataFrame,
    methods: list[str],
    *,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    palette: list[str] | dict | None = None,
    ylabel: str = r"Log Scale of Time $[\mathrm{s}]$",
    title: str = r"Processing Time by Method",
    fontsize: int = 14,
    to_milliseconds: bool = False,
    swarm: bool = True,
    show_mean: bool = True,
    show_median: bool = True,
    show_std: bool = True,
    show_stats: bool = True,
    log_scale: bool = False,
    show_outlier: bool = False,
    iqr_shade: bool = False,
    show_pvalue: bool = False,
    offset_title: float = 1.0,
):
    """
    Publication-ready violin plot with optional statannotations p-value labels placed under the title.
    """
    # 1) Prepare data
    data = df[methods].melt(var_name="Method", value_name="Time").dropna()
    if to_milliseconds:
        data["Time"] *= 1000
        ylabel = ylabel.replace("[\\mathrm{s}]", "[\\mathrm{ms}]")

    stats = (
        data.groupby("Method")["Time"]
        .agg(mean="mean", std="std", median="median")
        .reindex(methods)
    )

    # 2) Palette
    if palette is None:
        palette = {m: TABLEAU10[i % len(TABLEAU10)] for i, m in enumerate(methods)}
    elif isinstance(palette, list):
        palette = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    # 3) Create/reuse axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 4) Draw violins
    sns.violinplot(
        x="Method",
        y="Time",
        data=data,
        order=methods,
        palette=[palette[m] for m in methods],
        inner="quartile",
        cut=0,
        scale="width",
        linewidth=1.3,
        alpha=0.75,
        ax=ax,
    )

    # 5) Optional swarm
    if swarm:
        sns.swarmplot(
            x="Method",
            y="Time",
            data=data,
            order=methods,
            color="white",
            edgecolor="black",
            linewidth=0.5,
            size=3.2,
            zorder=6,
            ax=ax,
        )

    # 6) Mean/median/std + stats text
    for i, m in enumerate(methods):
        mu, sd, med = stats.loc[m]
        if show_std:
            ax.errorbar(
                i,
                mu,
                yerr=sd,
                fmt="none",
                ecolor="black",
                elinewidth=2.2,
                capsize=7,
                zorder=5,
            )
        if show_mean:
            ax.scatter(
                i,
                mu,
                s=180,
                facecolor="white",
                edgecolor="black",
                linewidth=2.0,
                zorder=6,
            )
        if show_median:
            ax.scatter(i, med, s=120, marker="D", color="black", zorder=6)
        if show_stats:
            ax.text(
                i,
                mu + sd * 0.05,
                rf"{mu:.2f} $\pm$ {sd:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
            )

    # 7) IQR shading
    if iqr_shade:
        for i, m in enumerate(methods):
            q1 = data.loc[data.Method == m, "Time"].quantile(0.25)
            q3 = data.loc[data.Method == m, "Time"].quantile(0.75)
            ax.fill_between(
                [i - 0.4, i + 0.4], q1, q3, color="gray", alpha=0.1, zorder=0
            )

    # 8) Outliers
    if show_outlier:
        for i, m in enumerate(methods):
            mx = data.loc[data.Method == m, "Time"].max()
            ax.scatter(
                i,
                mx,
                marker="*",
                s=230,
                color="red",
                edgecolor="black",
                linewidth=1.1,
                zorder=7,
            )
            ax.annotate(
                f"{mx:.2f}",
                xy=(i, mx),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=fontsize,
                color="red",
            )

    # 9) Apply log scale *before* p-value annotation
    if log_scale:
        ax.set_yscale("log")
        # title += r" (log scale)"
        # title = None

    # 10) p-value brackets under title
    if show_pvalue:
        pairs = list(itertools.combinations(methods, 2))
        annot = Annotator(ax, pairs, data=data, x="Method", y="Time", order=methods)
        # Draw bracket at 100% of the data range above max, then stars a bit higher
        annot.configure(
            test="t-test_ind",
            text_format="star",
            loc="outside",
            line_offset=0.1,  # 100% above the top of the violins
            text_offset=0.95,  # 105% above => directly under title
            verbose=0,
            fontsize=fontsize + 2,
            line_width=2,
            comparisons_correction="holm",
            # comparisons_correction=None
        )
        annot.apply_and_annotate()
        # Finally extend the y-limit to accommodate that bracket
        ymin, ymax = data["Time"].min(), data["Time"].max()
        yrange = ymax - ymin
        ax.set_ylim(ymin, ymax + 1.05 * yrange)

    # 11) Final styling
    ax.set_title(title, fontsize=fontsize + 4, pad=15, y=offset_title, loc="left")
    ax.set_ylabel(ylabel, fontsize=fontsize + 1)
    ax.set_xlabel("")
    ax.tick_params(labelsize=fontsize - 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_significance_heatmap_ttest_annotator(
    ax: Axes,
    df: pd.DataFrame,
    methods: list[str],
    *,
    alpha: float = 0.05,
    adjust_method: str = "holm",
    to_milliseconds: bool = False,
    palette: list[str] | None = None,
    linewidth: float = 0.5,
    fontsize: int = 12,
    title: str | None = None,
    labels: list[str] | None = None,
    show_colorbar: bool = True,
) -> Axes:
    """
    Heatmap of pairwise paired t-tests with Holm correction.

    Categories:
      0 = NS        (light gray)
      1 = p < alpha (light blue)
      2 = p < 0.01  (medium blue)
      3 = p < 0.001 (dark blue)

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the heatmap.
    df : pandas.DataFrame
        DataFrame containing the methods as columns.
    methods : list[str]
        Column names in df to compare.
    alpha : float
        Significance level for Holm correction (default 0.05).
    adjust_method : str
        Multiple-testing correction method (default 'holm').
    to_milliseconds : bool
        If True, multiply data by 1000 before testing.
    palette : list[str] | None
        Four colors for categories 0–3; uses gray→blue palette if None.
    linewidth : float
        Width of lines between cells.
    fontsize : int
        Base font size for labels and title.
    title : str | None
        Title of the heatmap.
    labels : list[str] | None
        Custom tick labels; if None, uses method names.
    show_colorbar : bool
        Whether to display the colorbar (default True).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified Axes.
    """
    # 1) Scale to ms if requested
    data = df[methods].copy()
    if to_milliseconds:
        data = data * 1000

    # 2) Compute raw p-values
    n = len(methods)
    raw_p, pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            xi = data[methods[i]].dropna()
            yj = data[methods[j]].dropna()
            common = xi.index.intersection(yj.index)
            if len(common) < 2:
                p = 1.0
            else:
                _, p = ttest_rel(xi.loc[common], yj.loc[common])
            raw_p.append(p)
            pairs.append((i, j))

    # 3) Adjust with Holm
    _, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method=adjust_method)

    # 4) Create category matrix
    cats = np.zeros((n, n), dtype=int)
    for (i, j), p in zip(pairs, p_adj):
        if p < 1e-3:
            cats[i, j] = cats[j, i] = 3
        elif p < 1e-2:
            cats[i, j] = cats[j, i] = 2
        elif p < alpha:
            cats[i, j] = cats[j, i] = 1
        else:
            cats[i, j] = cats[j, i] = 0

    # 5) Setup colormap
    if palette is None:
        palette = ["#f0f0f0", "#9ecae1", "#6baed6", "#2171b5"]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=len(palette))

    # 6) Plot heatmap
    heat = sns.heatmap(
        cats,
        mask=np.eye(n, dtype=bool),
        cmap=cmap,
        norm=norm,
        cbar=show_colorbar,
        cbar_kws=(
            {
                "ticks": [0, 1, 2, 3],
                "shrink": 0.7,
                "drawedges": False,
            }
            if show_colorbar
            else {}
        ),
        linewidths=linewidth,
        linecolor="white",
        xticklabels=labels if labels is not None else methods,
        yticklabels=labels if labels is not None else methods,
        square=True,
        ax=ax,
    )

    # 7) Style colorbar
    if show_colorbar:
        cbar = heat.collections[0].colorbar
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["NS", r"$p<0.05$", r"$p<0.01$", r"$p<0.001$"])
        cbar.ax.tick_params(length=0, labelsize=fontsize)
        cbar.outline.set_visible(False)
        try:
            cbar.solids.set_edgecolor("face")
            cbar.solids.set_linewidth(0)
        except AttributeError:
            pass

    # 8) Final axis styling
    ax.set_title(title or "", fontsize=fontsize + 2, pad=12, loc="left")
    ax.tick_params(labelsize=fontsize)

    # 9) Rotate tick labels to avoid overlap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="center")

    return ax
