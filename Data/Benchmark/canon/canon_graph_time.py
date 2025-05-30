# #!/usr/bin/env python3
# import sys
# import os
# import time
# import json
# from pathlib import Path
# from typing import Any, Dict, List

# import pandas as pd

# # Ensure project root is on sys.path
# project_root = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
# )
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from synkit.Graph.canon_graph import GraphCanonicaliser
# from synkit.IO import load_database, save_dict_to_json, rsmi_to_its


# def benchmark_graph_canonicalisation(
#     records: List[Dict[str, Any]], its_key: str = "ITS"
# ) -> Dict[str, List[float]]:
#     """
#     Benchmark per-record graph canonicalisation times for various backends:
#       - generic
#       - wl with 1–3 iterations
#       - morgan with radius 1–3
#     """
#     # Build config/label lists
#     configs: List[Dict[str, Any]] = []
#     labels: List[str] = []

#     # generic
#     configs.append({"backend": "generic"})
#     labels.append("generic")

#     # WL₁–₃
#     for i in range(1, 4):
#         configs.append({"backend": "wl", "wl_iterations": i})
#         labels.append(f"wl_{i}")

#     # Morgan₁–₃
#     for r in range(1, 4):
#         configs.append({"backend": "morgan", "morgan_radius": r})
#         labels.append(f"morgan_{r}")

#     # Prepare timing container
#     times: Dict[str, List[float]] = {lbl: [] for lbl in labels}

#     # Run the benchmarks
#     for cfg, lbl in zip(configs, labels):
#         canon = GraphCanonicaliser(**cfg)
#         for rec in records:
#             G = rec[its_key]
#             t0 = time.perf_counter()
#             _ = canon.canonicalise_graph(G).canonical_graph
#             times[lbl].append(time.perf_counter() - t0)

#     return times


# if __name__ == "__main__":
#     # load first 100 records
#     df = load_database(f"{project_root}/Data/Benchmark/benchmark.json.gz")[:]

#     # build ITS graphs
#     for rec in df:
#         rec["ITS"] = rsmi_to_its(rec["smart"])

#     # run benchmark
#     results = benchmark_graph_canonicalisation(df)

#     # save out
#     out_path = (
#         Path(project_root) / "Data" / "Benchmark" / "canon" / "canon_graph_times.json"
#     )
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     save_dict_to_json(results, str(out_path))


#!/usr/bin/env python3
import sys
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# Ensure project root is on sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# SynKit IO and Graph imports
from synkit.IO import setup_logging, load_database, save_dict_to_json, rsmi_to_its
from synkit.Graph.canon_graph import GraphCanonicaliser
def benchmark_graph_canonicalisation(
    records: List[Dict[str, Any]], its_key: str = "ITS"
) -> Dict[str, List[float]]:
    """
    Benchmark per-record graph canonicalisation times for various backends:
      - generic
      - wl with 1–3 iterations
      - morgan with radius 1–3
      - nauty
    """
    # Build config/label lists
    configs: List[Dict[str, Any]] = []
    labels: List[str] = []

    # generic
    configs.append({"backend": "generic"})
    labels.append("generic")

    # WL₁–₃
    for i in range(1, 4):
        configs.append({"backend": "wl", "wl_iterations": i})
        labels.append(f"wl_{i}")

    # Morgan₁–₃
    for r in range(1, 4):
        configs.append({"backend": "morgan", "morgan_radius": r})
        labels.append(f"morgan_{r}")

    # Nauty
    configs.append({"backend": "nauty"})
    labels.append("nauty")

    # Prepare timing container
    times: Dict[str, List[float]] = {lbl: [] for lbl in labels}

    # Run the benchmarks
    for cfg, lbl in zip(configs, labels):
        for rec in records:
            G = rec[its_key]
            canon = GraphCanonicaliser(**cfg)
            t0 = time.perf_counter()
            _ = canon.canonicalise_graph(G).canonical_graph
            times[lbl].append(time.perf_counter() - t0)

    return times


if __name__ == "__main__":
    # Initialize logging
    log_path = Path(project_root) / "Data" / "Benchmark" / "canon" / "benchmark_graph.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level="INFO", log_filename=str(log_path))

    from logging import getLogger
    logger = getLogger(__name__)

    # load first 100 records
    db_path = Path(project_root) / "Data" / "Benchmark" / "benchmark.json.gz"
    df = load_database(str(db_path))[:]
    logger.info("Loaded %d records for graph canonicalisation", len(df))

    # build ITS graphs
    for rec in df:
        rec["ITS"] = rsmi_to_its(rec.get("smart", rec.get("SMART")))
    logger.info("Converted SMART strings to ITS graphs")

    # run benchmark
    results = benchmark_graph_canonicalisation(df)
    logger.info("Completed benchmark for backends: %s", ", ".join(results.keys()))

    # save out
    out_path = Path(project_root) / "Data" / "Benchmark" / "canon" / "canon_graph_times.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict_to_json(results, str(out_path))
    logger.info("Timing results saved to %s", out_path)
