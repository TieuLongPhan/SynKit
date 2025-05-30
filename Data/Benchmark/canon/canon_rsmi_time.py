import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import os
import pandas as pd

# Ensure project root is on sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def benchmark_smart_canonicalisation(
    records: List[Dict[str, Any]],
    smart_key: str = "smart",
    node_attrs: List[str] = ["element", "aromatic", "charge", "hcount", "neighbors"],
) -> Dict[str, List[float]]:
    """
    Benchmark per-record canonicalisation times for various backends.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Each dict must contain the SMART(SMILES/SMARTS) under `smart_key`.
    smart_key : str
        Key in each dict for the string to canonicalise.
    node_attrs : List[str]
        Node-attribute keys for CanonRSMI.

    Returns
    -------
    Dict[str, List[float]]
        Mapping from backend labels ('generic', 'wl_1'…'wl_3', 'morgan_1'…'morgan_3')
        to a list of processing times (seconds) for each record.
    """
    df = pd.DataFrame(records)

    # Prepare configurations and labels
    configs: List[Dict[str, Any]] = []
    labels: List[str] = []

    # generic
    configs.append({"backend": "generic"})
    labels.append("generic")

    # wl and morgan with iterations 1–3
    for backend in ("wl", "morgan"):
        for i in range(1, 4):
            cfg: Dict[str, Any] = {"backend": backend}
            if backend == "wl":
                cfg["wl_iterations"] = i
                lbl = f"wl_{i}"
            else:
                cfg["morgan_radius"] = i
                lbl = f"morgan_{i}"
            configs.append(cfg)
            labels.append(lbl)

    # Initialize result container
    times: Dict[str, List[float]] = {lbl: [] for lbl in labels}

    # Run benchmark
    for cfg, lbl in zip(configs, labels):
        canon = CanonRSMI(**cfg, node_attrs=node_attrs)
        for smi in df[smart_key]:
            t0 = time.perf_counter()
            _ = canon.canonicalise(smi).canonical_rsmi
            times[lbl].append(time.perf_counter() - t0)

    return times


if __name__ == "__main__":
    from synkit.Chem import CanonRSMI

    from synkit.IO import load_database, save_database,save_dict_to_json

    print(project_root)
    df = load_database(f"{project_root}/Data/Benchmark/benchmark.json.gz")[:]

    times = benchmark_smart_canonicalisation(df)

    save_dict_to_json(
        times, f"{project_root}/Data/Benchmark/canon/canon_rsmi_times.json"
    )
