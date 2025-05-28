#!/usr/bin/env python3
"""
benchmark_clustering.py
=======================

Enhanced benchmark comparing ITS and GML clustering with WL hashing at specified iterations.

Two tests:
 1. Full dataset clustering: violin+box plot on log-scaled ms with mean±std annotations.
 2. Scalability: error-bar plot of clustering time (ms) vs N in log–log scale.

Usage:
  python benchmark_clustering.py [--data DATA_PATH]
                                [--out_dir OUT_DIR]
                                [--wl_iters WL1,WL2,WL3]
                                [--repeat R]
                                [--max_n N1,N2,...]
                                [--limit L]

Defaults:
  DATA_PATH  = ./Data/Benchmark/benchmark.json.gz
  OUT_DIR    = ./Data/Benchmark/cluster
  wl_iters   = 1,2,3
  repeat     = 3
  max_n      = 100,300,1000,3000,10000
  limit      = None
"""
import os
import sys
import time
import tracemalloc
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# SynKit imports
from synkit.IO.data_io import load_database
from synkit.IO.chem_converter import rsmi_to_its, smart_to_gml
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Rule.Modify.rule_utils import strip_context


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_and_precompute(
    data_path: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load data and precompute ITS and GML. Applies limit if provided.
    """
    data = load_database(str(data_path))
    if limit is not None:
        data = data[:limit]
    for entry in data:
        entry["ITS"] = rsmi_to_its(entry["smart"], core=False)
        entry["gml"] = strip_context(
            smart_to_gml(entry["smart"], core=False, sanitize=True)
        )

    return data


def cluster_data(
    data_slice: List[Dict[str, Any]],
    key: str,
    wl_iters: int,
) -> Tuple[int, float, int]:
    """
    Run GraphCluster.fit on data_slice using WL hashing at wl_iters.
    For ITS use backend 'nx', for GML use 'mod'.
    Returns number of clusters, elapsed time (s), and peak memory (bytes).
    """
    # Determine clustering engine backend
    cluster_backend = "nx" if key.lower() == "its" else "mod"

    # Compute WL signature
    wl = WLHash(iterations=wl_iters)
    sig_key = f"SIG_{key.upper()}_WL{wl_iters}"
    for entry in data_slice:
        entry[sig_key] = wl.weisfeiler_lehman_graph_hash(entry["ITS"])

    # Instantiate GraphCluster
    cls = GraphCluster(backend=cluster_backend)

    # Benchmark clustering
    tracemalloc.start()
    t0 = time.perf_counter()
    clusters = cls.fit(
        data_slice,
        key,
        attribute_key=sig_key,
    )
    # if key.lower() == "gml":
    #     logging.info(f"{clusters}")
    elapsed = time.perf_counter() - t0
    mem_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return len(clusters), elapsed, mem_peak


def test_full(
    data: List[Dict[str, Any]],
    wl_iters_list: List[int],
    repeat: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Full dataset clustering: compare ITS and GML with WL hashing.
    """
    methods = [("ITS", wl) for wl in wl_iters_list] + [
        ("gml", wl) for wl in wl_iters_list
    ]

    records = []
    n_items = len(data)
    logging.info(f"Test 1: full clustering on {n_items} items, {repeat} repeats")
    for key, wl in methods:
        name = f"{key.upper()}+WL{wl}"
        for r in range(repeat):
            _, elapsed_s, _ = cluster_data(data, key, wl)
            elapsed_ms = elapsed_s * 1000
            records.append({"method": name, "time_ms": elapsed_ms})
            logging.info(f"Full: {name} rep={r} {elapsed_ms:.1f}ms")
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "full_results.csv", index=False)
    summary = df.groupby("method")["time_ms"].agg(["mean", "std"]).reset_index()
    summary.to_csv(out_dir / "full_summary.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.violinplot(x="method", y="time_ms", data=df, inner=None)
    sns.boxplot(
        x="method",
        y="time_ms",
        data=df,
        width=0.1,
        showcaps=True,
        boxprops={"zorder": 2},
    )
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.ylabel("Clustering time (ms, log scale)")
    plt.title("Full-data clustering time distribution")
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
    plt.savefig(out_dir / "full_violin_log.png", dpi=300)
    logging.info("Saved full-data violin plot (log)")
    return df


def test_scalability(
    data: List[Dict[str, Any]],
    wl_iters_list: List[int],
    Ns: List[int],
    repeat: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Scalability: compare ITS and GML with WL hashing across sizes.
    """
    methods = [("ITS", wl) for wl in wl_iters_list] + [
        ("gml", wl) for wl in wl_iters_list
    ]

    records = []
    logging.info(f"Test 2: scalability Ns={Ns}, repeats={repeat}")
    for N in Ns:
        subset = data[:N]
        for key, wl in methods:
            name = f"{key.upper()}+WL{wl}"
            times = []
            for _ in range(repeat):
                times.append(cluster_data(subset, key, wl)[1] * 1000)
            records.append(
                {
                    "method": name,
                    "N": N,
                    "mean_ms": np.mean(times),
                    "std_ms": np.std(times),
                }
            )
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "scalability_results.csv", index=False)

    plt.figure(figsize=(6, 4))
    for name, sub in df.groupby("method"):
        plt.errorbar(
            sub["N"],
            sub["mean_ms"],
            yerr=sub["std_ms"],
            fmt="o-",
            capsize=5,
            label=name,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N (number of graphs)")
    plt.ylabel("Mean clustering time (ms)")
    plt.title("Scalability: time vs N (log–log)")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "scalability_loglog.png", dpi=300)
    logging.info("Saved scalability error-bar plot (log–log)")
    return df


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Benchmark GraphCluster WL methods")
    p.add_argument("--data", type=str, default="./Data/Benchmark/benchmark.json.gz")
    p.add_argument("--out_dir", type=str, default="./Data/Benchmark/cluster")
    p.add_argument(
        "--wl_iters",
        type=str,
        default="0,1,2,3",
        help="Comma-separated WL iteration counts",
    )
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument(
        "--max_n", type=str, default="100,300,1000,3000,10000,20000,30000,40000"
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    wl_iters_list = [int(x) for x in args.wl_iters.split(",")]
    Ns = [int(x) for x in args.max_n.split(",")]
    logging.info("Loading data and precomputing...")
    data = load_and_precompute(Path(args.data), limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Test Full...")
    test_full(data, wl_iters_list, args.repeat, out_dir)
    logging.info("Test Scalability...")
    test_scalability(data, wl_iters_list, Ns[1:], args.repeat, out_dir)


if __name__ == "__main__":
    main()
