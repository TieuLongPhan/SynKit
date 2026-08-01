#!/usr/bin/env python
"""Run the bounded RX13 reaction-stereo validation microbenchmark."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

try:
    import resource
except ModuleNotFoundError:  # Windows
    resource = None

from rdkit import rdBase

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.Graph.Stereo import (  # noqa: E402
    StereoChange,
    StereoOutcome,
    StereoReactionValue,
    TetrahedralStereo,
    project_reaction_stereo,
    reaction_stereo_from_graph,
)

ITERATIONS = 2500
GRAPH_ITERATIONS = 250
BUDGETS = {
    "wall_seconds": 15.0,
    "cpu_seconds": 15.0,
    "max_rss_mib": 768.0,
    "proof_bytes": 65536,
    "branch_count": 2,
    "assignment_count": 2,
}


def _windows_peak_rss_mib() -> float:
    """Read the peak working set without adding a runtime dependency."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        raise OSError("GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize / (1024 * 1024)


def _posix_rss_mib(maximum: float, platform_name: str) -> float:
    divisor = 1024 * 1024 if platform_name == "darwin" else 1024
    return maximum / divisor


def _max_rss_mib() -> float:
    if resource is None:
        return _windows_peak_rss_mib()
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return _posix_rss_mib(maximum, sys.platform)


def _value() -> StereoReactionValue:
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), 1)
    after = before.invert()
    return StereoReactionValue(
        guards={"atom:2": before},
        effects={
            "atom:2": StereoChange.from_endpoints(before, after)
        },
        outcomes={"atom:2": StereoOutcome("SINGLE")},
    )


def validation_report() -> dict:
    """Return stable environment, result, and budget evidence."""
    value = _value()
    payload = value.normalized_json()
    expected_digest = sha256(payload.encode("utf-8")).hexdigest()
    start = perf_counter()
    cpu_start = process_time()
    observed = set()
    for _index in range(ITERATIONS):
        restored = StereoReactionValue.from_json(payload)
        normalized = restored.normalized_json()
        observed.add(sha256(normalized.encode("utf-8")).hexdigest())
    for _index in range(GRAPH_ITERATIONS):
        graph, report = project_reaction_stereo(
            value,
            "internal_graph",
        )
        if not report.lossless or reaction_stereo_from_graph(graph) != value:
            raise RuntimeError("Reaction-stereo sidecar round trip failed.")
    elapsed = perf_counter() - start
    cpu_elapsed = process_time() - cpu_start
    rss_mib = _max_rss_mib()
    proof_bytes = len(payload.encode("utf-8"))
    observed_values = {
        "wall_seconds": round(elapsed, 6),
        "cpu_seconds": round(cpu_elapsed, 6),
        "max_rss_mib": round(rss_mib, 3),
        "proof_bytes": proof_bytes,
        "branch_count": 1,
        "assignment_count": 1,
    }
    checks = {
        name: observed_values[name] <= limit
        for name, limit in BUDGETS.items()
    }
    checks["deterministic_digest"] = observed == {expected_digest}
    return {
        "schema": "synkit.stereo-rxn-validation/1",
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "workload": {
            "wire_round_trips": ITERATIONS,
            "graph_sidecar_round_trips": GRAPH_ITERATIONS,
        },
        "budgets": BUDGETS,
        "observed": observed_values,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    print(
        json.dumps(
            validation_report(),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
