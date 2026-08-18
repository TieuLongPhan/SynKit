#!/usr/bin/env python3
"""Plot partial-AAM expansion accuracy, coverage, and runtime evidence."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.partial_expand.timing_artifacts import SCHEMA  # noqa: E402

RESULTS = HERE / "Data"
PAPER_FIGURE = ROOT / "paper" / "lwg" / "fig" / "partial_aam_comparison.png"
RUNTIME_METHODS = ("rb1", "rb2", "gm")
METHODS = ("synkit", "gm", "rb1", "rb2")
LABELS = {
    "synkit": r"$\mathtt{SynKit}$",
    "gm": r"$\mathtt{GM}$",
    "rb1": r"$\mathtt{RB1}$",
    "rb2": r"$\mathtt{RB2}$",
}
# Validated colourblind-safe categorical palette (Okabe--Ito); identity is also
# carried by the direct method labels on every row.
COLORS = {
    "synkit": "#0072B2",
    "gm": "#009E73",
    "rb1": "#E69F00",
    "rb2": "#CC79A7",
}
RUNTIME_COLORS = {
    "rb1": "#4C78A8",
    "rb2": "#F2A65A",
    "gm": "#E07B7B",
}
# Display order (fastest general expansion first); shared across both panels.
ORDER = ("synkit", "rb1", "rb2", "gm")
INK = "#232323"
MUTED = "#6B6B6B"
GRID = "#E7E9EB"
SPINE = "#B7BCC1"
RADICAL_RECORDS = 5_426

# The external radical counts were retained in sprint/SS_LOG.md before the
# redundant per-case files were removed. SynKit is loaded from its current
# minimal-path five-run aggregate below.
EXTERNAL_RADICAL_VALID_COUNTS = (3_869, 5_326, 5_326)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synkit-results",
        type=Path,
        default=RESULTS / "synkit-normal-expansion-5x" / "aggregate.json",
    )
    parser.add_argument(
        "--external-results",
        type=Path,
        default=RESULTS / "gm-rb-normal-expansion-5x",
    )
    parser.add_argument(
        "--synkit-radical-results",
        type=Path,
        default=RESULTS / "synkit-radical-expansion-5x" / "aggregate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "expansion-comparison-general-radical",
        help="Output stem; PDF and PNG are written.",
    )
    parser.add_argument(
        "--runtime-results",
        type=Path,
        help=(
            "Directory containing general-<method>-run-<NN>-timings.json.gz; "
            "when supplied, render the RB1/RB2/GM runtime violin instead"
        ),
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        default=RESULTS / "partial-aam-runtime",
        help="Runtime-violin output stem; PDF and PNG are written.",
    )
    parser.add_argument("--runtime-repetitions", type=int, default=5)
    parser.add_argument(
        "--runtime-reduction",
        choices=("median", "mean", "pooled"),
        default="median",
        help=(
            "Reduce repeated measurements per reaction before plotting; "
            "median is the default statistical unit"
        ),
    )
    return parser.parse_args()


def load_synkit(path: Path) -> list[float]:
    payload = json.loads(path.read_text())
    aggregate = next(
        item for item in payload["aggregates"] if item["method"] == "synkit"
    )
    values = aggregate["metrics"]["mean_generation_seconds_per_attempt"]["values"]
    return [float(value) * 1000 for value in values]


def load_synkit_radical_coverage(path: Path) -> int:
    payload = json.loads(path.read_text())
    aggregate = next(
        item for item in payload["aggregates"] if item["method"] == "synkit"
    )
    metrics = aggregate["metrics"]
    return round(float(metrics["accepted_coverage"]["mean"]) * RADICAL_RECORDS)


def load_external(directory: Path) -> dict[str, list[float]]:
    timings = {method: [] for method in METHODS[1:]}
    for path in sorted(directory.glob("general-*-generation.json")):
        payload = json.loads(path.read_text())
        method = payload["methods"][0]
        name = str(method["method"])
        if name in timings:
            timings[name].append(float(method["generation_seconds"]["mean"]) * 1000)
    missing = [name for name, values in timings.items() if len(values) != 5]
    if missing:
        raise ValueError(f"Expected five retained runs for: {', '.join(missing)}")
    return timings


def _load_timing_payload(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported timing schema in {path}")
    if payload.get("unit") != "seconds":
        raise ValueError(f"Expected second-based timings in {path}")
    return payload


def _runtime_reducer(name: str) -> Callable[[list[float]], float]:
    if name == "median":
        return statistics.median
    if name == "mean":
        return statistics.mean
    raise ValueError(f"No per-reaction reducer for {name!r}")


def _load_timing_run(
    path: Path,
    expected_method: str,
) -> tuple[int, dict[int, float], str]:
    payload = _load_timing_payload(path)
    if payload.get("method") != expected_method:
        raise ValueError(f"Method mismatch in {path}")
    repetition = int(payload["repetition"])
    samples: dict[int, float] = {}
    for sample in payload["samples"]:
        record_id = int(sample["record_id"])
        if record_id in samples:
            raise ValueError(f"Duplicate record {record_id} in {path}")
        if sample.get("status") != "OUTPUT":
            raise ValueError(f"Cannot plot non-output record {record_id} from {path}")
        seconds = float(sample["generation_seconds"])
        if seconds <= 0:
            raise ValueError(f"Non-positive timing for record {record_id}")
        samples[record_id] = seconds * 1000
    return repetition, samples, str(payload["dataset"]["sha256"])


def _load_method_runs(
    directory: Path,
    method: str,
    repetitions: int,
) -> tuple[dict[int, dict[int, float]], set[int], str]:
    paths = sorted(directory.glob(f"general-{method}-run-*-timings.json.gz"))
    if len(paths) != repetitions:
        raise ValueError(
            f"Expected {repetitions} {method.upper()} timing files, "
            f"found {len(paths)} in {directory}"
        )
    runs: dict[int, dict[int, float]] = {}
    dataset_hashes = set()
    for path in paths:
        repetition, samples, dataset_sha256 = _load_timing_run(path, method)
        if repetition in runs:
            raise ValueError(f"Duplicate {method} repetition {repetition}")
        runs[repetition] = samples
        dataset_hashes.add(dataset_sha256)
    expected_runs = set(range(1, repetitions + 1))
    if set(runs) != expected_runs:
        raise ValueError(
            f"{method.upper()} repetitions are {sorted(runs)}; "
            f"expected {sorted(expected_runs)}"
        )
    record_sets = [set(samples) for samples in runs.values()]
    if any(records != record_sets[0] for records in record_sets[1:]):
        raise ValueError(f"{method.upper()} runs do not contain identical records")
    if len(dataset_hashes) != 1:
        raise ValueError(f"{method.upper()} runs do not use one dataset")
    return runs, record_sets[0], dataset_hashes.pop()


def load_runtime_samples(
    directory: Path,
    *,
    repetitions: int = 5,
    reduction: str = "median",
) -> dict[str, list[float]]:
    """Load milliseconds per reaction from repeated compact timing artifacts."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    collected: dict[str, list[float]] = {}
    shared_records: set[int] | None = None
    shared_dataset_sha256: str | None = None
    for method in RUNTIME_METHODS:
        runs, records, dataset_sha256 = _load_method_runs(
            directory,
            method,
            repetitions,
        )
        if shared_records is None:
            shared_records = records
        elif records != shared_records:
            raise ValueError(f"{method.upper()} does not contain the shared record set")
        if shared_dataset_sha256 is None:
            shared_dataset_sha256 = dataset_sha256
        elif dataset_sha256 != shared_dataset_sha256:
            raise ValueError(f"{method.upper()} does not use the shared dataset")
        if reduction == "pooled":
            collected[method] = [
                value
                for repetition in sorted(runs)
                for value in runs[repetition].values()
            ]
            continue
        reducer = _runtime_reducer(reduction)
        collected[method] = [
            reducer([runs[run][record_id] for run in sorted(runs)])
            for record_id in sorted(records)
        ]
    return collected


def _runtime_violin_panel(axis, samples: dict[str, list[float]]) -> None:
    values = [samples[method] for method in RUNTIME_METHODS]
    positions = list(range(1, len(RUNTIME_METHODS) + 1))
    violins = axis.violinplot(
        values,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=300,
    )
    for body, method in zip(violins["bodies"], RUNTIME_METHODS):
        body.set_facecolor(RUNTIME_COLORS[method])
        body.set_edgecolor(INK)
        body.set_linewidth(0.8)
        body.set_alpha(0.64)

    for position, method in zip(positions, RUNTIME_METHODS):
        method_values = samples[method]
        mean = statistics.mean(method_values)
        sample_std = statistics.stdev(method_values) if len(method_values) > 1 else 0.0
        maximum = max(method_values)
        quartiles = statistics.quantiles(method_values, n=4, method="inclusive")
        for percentile, style in zip(quartiles, (":", "--", ":")):
            axis.hlines(
                percentile,
                position - 0.27,
                position + 0.27,
                color=INK,
                linestyle=style,
                linewidth=0.9,
                alpha=0.8,
                zorder=3,
            )
        lower = max(min(method_values), mean - sample_std)
        axis.errorbar(
            position,
            mean,
            yerr=[[mean - lower], [sample_std]],
            fmt="o",
            markersize=8,
            color="black",
            markeredgecolor="black",
            ecolor="black",
            elinewidth=1.4,
            capsize=5,
            zorder=5,
        )
        axis.text(
            position,
            mean * 1.18,
            f"{mean:.2f} \u00b1 {sample_std:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=INK,
            zorder=6,
        )
        axis.scatter(
            position,
            maximum,
            marker="*",
            s=120,
            facecolor="#E53935",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
        axis.text(
            position - 0.05,
            maximum * 1.10,
            f"{maximum:.2f}",
            ha="right",
            va="bottom",
            fontsize=8.6,
            color="#E53935",
            zorder=6,
        )

    axis.set_xticks(positions, [LABELS[method] for method in RUNTIME_METHODS])
    axis.set_ylabel("Generation time per reaction (ms)")
    axis.set_yscale("log")
    all_values = [value for method_values in values for value in method_values]
    axis.set_ylim(min(all_values) * 0.72, max(all_values) * 1.75)
    axis.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)
    axis.set_title(
        "Partial atom-mapping extension runtime",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        pad=9,
    )
    _style_open_axis(axis)


def plot_runtime(samples: dict[str, list[float]], output: Path) -> None:
    """Render RB1/RB2/GM reaction-level runtime distributions without ILP."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.3,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": 160,
            "axes.linewidth": 0.8,
        }
    )
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    _runtime_violin_panel(axis, samples)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(figure)


def _style_open_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, length=3, width=0.8)
    axis.set_axisbelow(True)


def _panel_key(axis, label: str) -> None:
    axis.text(
        -0.02,
        1.07,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def _method_yaxis(axis, order) -> None:
    axis.set_yticks(list(range(len(order))), [LABELS[m] for m in order])
    axis.set_ylim(-0.6, len(order) - 0.4)
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0, pad=8)
    _style_open_axis(axis)
    axis.spines["left"].set_visible(False)


def _general_dot_panel(axis, samples: dict, label: str) -> None:
    """Cleveland dot plot: mean +/- sd generation time, five runs per method."""
    _method_yaxis(axis, ORDER)
    for y, method in enumerate(ORDER):
        values = samples[method]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        color = COLORS[method]
        axis.scatter(
            values,
            [y] * len(values),
            s=15,
            color=color,
            alpha=0.30,
            edgecolor="none",
            zorder=2,
        )
        axis.errorbar(
            mean,
            y,
            xerr=sd,
            fmt="o",
            markersize=8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.2,
            zorder=4,
        )
        axis.text(
            mean + sd + 0.035,
            y,
            f"{mean:.3f} \u00b1 {sd:.3f}",
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK,
        )
    axis.set_xlim(1.70, 2.74)
    axis.set_xlabel("Time per input (ms)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_title(
        "General-corpus expansion time",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    _panel_key(axis, label)


def _coverage_bar_panel(axis, coverage: dict, label: str) -> None:
    """Radical valid-completion coverage as bars; 100% General ITS reference."""
    _method_yaxis(axis, ORDER)
    base = 60.0
    for y, method in enumerate(ORDER):
        pct = coverage[method] / RADICAL_RECORDS * 100
        axis.barh(
            y,
            pct - base,
            left=base,
            height=0.54,
            color=COLORS[method],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        inside = pct - base > 8
        axis.text(
            pct - 0.9 if inside else pct + 0.9,
            y,
            f"{pct:.1f}%",
            ha="right" if inside else "left",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="white" if inside else INK,
            zorder=4,
        )
    axis.axvline(100, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
    axis.text(
        100,
        -0.52,
        "General ITS 100%",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=MUTED,
    )
    axis.set_xlim(base, 109)
    axis.set_xlabel("Valid radical completion (%)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_title(
        "Radical completion coverage",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    _panel_key(axis, label)


def plot(
    synkit: list[float],
    external: dict[str, list[float]],
    radical_synkit_accepted: int,
    output: Path,
) -> None:
    samples = {"synkit": synkit, **external}
    coverage = {
        "synkit": radical_synkit_accepted,
        "gm": EXTERNAL_RADICAL_VALID_COUNTS[0],
        "rb1": EXTERNAL_RADICAL_VALID_COUNTS[1],
        "rb2": EXTERNAL_RADICAL_VALID_COUNTS[2],
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.3,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": 160,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    _general_dot_panel(axes[0], samples, "A")
    _coverage_bar_panel(axes[1], coverage, "B")
    figure.tight_layout(w_pad=3.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    figure.savefig(PAPER_FIGURE, bbox_inches="tight", dpi=300)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.runtime_results is not None:
        plot_runtime(
            load_runtime_samples(
                args.runtime_results,
                repetitions=args.runtime_repetitions,
                reduction=args.runtime_reduction,
            ),
            args.runtime_output,
        )
        print(args.runtime_output.with_suffix(".pdf"))
        print(args.runtime_output.with_suffix(".png"))
        return 0
    radical_synkit_accepted = load_synkit_radical_coverage(args.synkit_radical_results)
    plot(
        load_synkit(args.synkit_results),
        load_external(args.external_results),
        radical_synkit_accepted,
        args.output,
    )
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
