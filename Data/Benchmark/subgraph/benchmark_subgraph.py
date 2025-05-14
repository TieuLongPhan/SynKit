#!/usr/bin/env python3
"""
benchmark_subgraph.py
====================

Benchmark subgraph matching performance across three methods:
  - SubgraphMatch (NetworkX-based)
  - SING (path-based pruning)
  - TurboISO (optimized isomorphism)

For each ITS representation, match all reaction-center (RC) graphs.

Outputs:
  - 'subgraph_results.csv': raw timings per ITS per method
  - 'subgraph_summary.csv': mean±std timings per method
  - 'subgraph_bar.png': bar plot of mean±std matching times

Usage:
  python benchmark_subgraph.py [--data DATA_PATH]
                              [--out_dir OUT_DIR]
                              [--repeat R]
                              [--limit L]

Defaults:
  DATA_PATH  = ./Data/Benchmark/benchmark.json.gz
  OUT_DIR    = ./Data/Benchmark/subgraph
  repeat     = 1   # repetitions per ITS
  limit      = None
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root on sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# SynKit imports
from synkit.IO.data_io import load_database
from synkit.IO.chem_converter import rsmi_to_its, smart_to_gml
from synkit.Graph.ITS import get_rc
from synkit.Graph.Matcher.subgraph_matcher import SubgraphMatch
from synkit.Graph.Matcher.sing import SING
from synkit.Graph.Matcher.turbo_iso import TurboISO


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_and_prepare(
    data_path: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load database, compute ITS and RC graphs.
    """
    data = load_database(str(data_path))
    if limit is not None:
        data = data[:limit]
    for entry in data:
        entry["ITS"] = rsmi_to_its(entry["smart"])
        entry["rc"] = get_rc(entry["ITS"])
        entry["gml"] = smart_to_gml(entry["smart"], core=False, sanitize=True)
        entry["gml_rc"] = smart_to_gml(entry["smart"], core=True, sanitize=True)
    return data


def _subgraph_nx(parent, children):
    sb = SubgraphMatch()
    return [
        sb.subgraph_isomorphism(child, parent, check_type="mono") for child in children
    ]


def _subgraph_gml(parent, children):
    sb = SubgraphMatch()
    return [sb.rule_subgraph_morphism(child, parent) for child in children]


def _subgraph_sing(parent, children):
    sing = SING(parent, max_path_length=3)
    return [sing.search(child, prune=True) for child in children]


def _subgraph_turbo(parent, children):
    tb = TurboISO(parent, node_label=["element", "charge"], edge_label="order")
    return [tb.search(child, prune=True) for child in children]


def test_subgraph(
    data: List[Dict[str, Any]],
    repeat: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Execute subgraph matching for each ITS against all RC graphs.
    Store timings and generate summary and plot.
    """
    methods = [
        ("NX", _subgraph_nx),
        ("GML", _subgraph_gml),
        ("SING", _subgraph_sing),
        ("TurboISO", _subgraph_turbo),
    ]
    records = []
    rc_list = [entry["rc"] for entry in data]
    rc_list_gml = [entry["gml_rc"] for entry in data]
    logging.info(f"Matching {len(rc_list)} RC graphs per ITS, {repeat} repeats")

    for name, func in methods:
        for entry in data:
            if name == "GML":
                parent = entry["gml"]
            else:
                parent = entry["ITS"]
            for _ in range(repeat):
                t0 = time.perf_counter()
                func(parent, rc_list) if name != "GML" else func(parent, rc_list_gml)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                records.append({"method": name, "time_ms": elapsed_ms})
            logging.info(f"{name}: last run time={elapsed_ms:.1f}ms")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "subgraph_results.csv", index=False)
    summary = df.groupby("method")["time_ms"].agg(["mean", "std"]).reset_index()
    summary.to_csv(out_dir / "subgraph_summary.csv", index=False)

    # Plot violin+box distribution
    methods_order = df["method"].unique().tolist()
    plt.figure(figsize=(8, 4))
    sns.violinplot(x="method", y="time_ms", data=df, inner=None, order=methods_order)
    sns.boxplot(
        x="method",
        y="time_ms",
        data=df,
        width=0.1,
        showcaps=True,
        boxprops={"zorder": 2},
        order=methods_order,
    )
    plt.ylabel("Matching time (ms)")
    plt.title("Subgraph matching time distribution")
    # Annotate mean±std using consistent ordering
    summary = (
        df.groupby("method")["time_ms"]
        .agg(["mean", "std"])
        .reindex(methods_order)
        .reset_index()
    )
    for i, row in summary.iterrows():
        plt.text(
            i,
            row["mean"] * 1.2,
            f"{row['mean']:.1f}±{row['std']:.1f}ms",
            ha="center",
            va="bottom",
            fontsize="small",
        )
    plt.tight_layout()
    plt.savefig(out_dir / "subgraph_violin.png", dpi=300)
    logging.info("Saved subgraph matching violin plot")
    return df


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Benchmark subgraph matching methods")
    parser.add_argument(
        "--data", type=str, default="./Data/Benchmark/benchmark.json.gz"
    )
    parser.add_argument("--out_dir", type=str, default="./Data/Benchmark/subgraph")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_and_prepare(Path(args.data), limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_subgraph(data, args.repeat, out_dir)


if __name__ == "__main__":
    main()
