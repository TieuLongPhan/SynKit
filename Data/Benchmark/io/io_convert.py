#!/usr/bin/env python3
"""
benchmark_io.py
===============

Benchmark GML/SMART/ITS conversions in SynKit with project-root path setup.

Usage:
  python benchmark_io.py [--data Data/Benchmark/benchmark.json.gz] [--limit N] [--out_dir Data/Benchmark/io]

Outputs in the specified out_dir:
  - conversion_times.json.gz  : raw per-step timings (list-of-lists)
  - conversion_times.csv.gz   : flat table (index, step, time_s)
  - conversion_stats.csv.gz   : stats (mean, std, count per step)
"""
import time
import sys
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark IO conversions in SynKit")
    parser.add_argument(
        "--data",
        type=str,
        default="Data/Benchmark/benchmark.json.gz",
        help="Path to input JSON(.gz) database",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of entries to process"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Data/Benchmark/io",
        help="Output directory for results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from synkit.IO.gml_to_nx import GMLToNX
    from synkit.IO.nx_to_gml import NXToGML
    from synkit.IO.data_io import load_database, save_dict_to_json
    from synkit.IO.chem_converter import (
        smart_to_gml,
        gml_to_smart,
        rsmi_to_graph,
        graph_to_rsmi,
    )
    from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
    from synkit.Graph.ITS.its_construction import ITSConstruction

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    STEP_NAMES = [
        "smart_to_gml",
        "gml_to_smart",
        "smart_to_its",
        "its_to_smart",
        "its_to_gml",
        "gml_to_its",
        "its_to_rc",
    ]

    def benchmark_io(
        data_path: Path,
        limit: Optional[int] = None,
        out_dir: Path = Path("Data/Benchmark/io"),
    ) -> None:
        """
        Run IO conversion benchmarks on dataset.

        Parameters
        ----------
        data_path : Path
            Path to JSON(.gz) database with entries containing 'smart' keys.
        limit : Optional[int]
            Maximum number of entries to process.
        out_dir : Path
            Directory to save output files.
        """
        out_dir = out_dir.expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        data: List[Dict[str, Any]] = load_database(str(data_path))
        if limit is not None:
            data = data[:limit]
        total = len(data)
        logging.info(f"Benchmarking {total} entries from {data_path} (limit={limit})")

        raw_times: Dict[str, List[float]] = {step: [] for step in STEP_NAMES}
        records: List[Dict[str, Any]] = []

        for idx, entry in enumerate(data):
            record: Dict[str, Any] = {"index": idx}

            # smart_to_gml
            t0 = time.perf_counter()
            gml = smart_to_gml(
                entry["smart"],
                core=False,
                sanitize=True,
                reindex=True,
                explicit_hydrogen=True,
            )
            dt = time.perf_counter() - t0
            raw_times["smart_to_gml"].append(dt)
            record["smart_to_gml"] = dt

            # gml_to_smart
            t0 = time.perf_counter()
            _ = gml_to_smart(gml, sanitize=True, explicit_hydrogen=False)
            dt = time.perf_counter() - t0
            raw_times["gml_to_smart"].append(dt)
            record["gml_to_smart"] = dt

            # smart_to_its
            t0 = time.perf_counter()
            r, p = rsmi_to_graph(entry["smart"])
            its = ITSConstruction().ITSGraph(r, p)
            dt = time.perf_counter() - t0
            raw_times["smart_to_its"].append(dt)
            record["smart_to_its"] = dt

            # its_to_smart
            t0 = time.perf_counter()
            r2, p2 = its_decompose(its)
            _ = graph_to_rsmi(r2, p2, its, explicit_hydrogen=False)
            dt = time.perf_counter() - t0
            raw_times["its_to_smart"].append(dt)
            record["its_to_smart"] = dt

            # its_to_gml
            t0 = time.perf_counter()
            _ = NXToGML().transform((r2, p2, its))
            dt = time.perf_counter() - t0
            raw_times["its_to_gml"].append(dt)
            record["its_to_gml"] = dt

            # gml_to_its
            t0 = time.perf_counter()
            _ = GMLToNX(gml).transform()
            dt = time.perf_counter() - t0
            raw_times["gml_to_its"].append(dt)
            record["gml_to_its"] = dt

            # its_to_rc
            t0 = time.perf_counter()
            _ = get_rc(its)
            dt = time.perf_counter() - t0
            raw_times["its_to_rc"].append(dt)
            record["its_to_rc"] = dt

            records.append(record)
            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                logging.info(f"Processed {idx+1}/{total} entries")

        save_dict_to_json(raw_times, out_dir / "conversion_times.json.gz")
        logging.info("Saved raw conversion times JSON")

        df = pd.DataFrame(records)
        df_long = df.melt(
            id_vars=["index"],
            value_vars=STEP_NAMES,
            var_name="step",
            value_name="time_s",
        )
        df_long.to_csv(out_dir / "conversion_times.csv", index=False)
        logging.info("Saved flat CSV of timings")

        stats = (
            df_long.groupby("step")["time_s"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats.to_csv(out_dir / "conversion_stats.csv", index=False)
        logging.info("Saved aggregate statistics CSV")

    args = parse_args()
    data_path = Path(args.data)
    limit = args.limit
    out_dir = Path(args.out_dir)
    benchmark_io(data_path, limit=limit, out_dir=out_dir)
