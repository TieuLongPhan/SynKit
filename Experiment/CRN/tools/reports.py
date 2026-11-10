"""Readable scientific results without saved fingerprints or build paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_BUILD_FIELDS = {
    "git_commit",
    "synkit_git_commit",
    "git_dirty",
    "boost_include",
    "compiler_path",
    "library_path",
    "rdkit_sdk",
    "executable",
    "kernel_library",
    "native_dpo_expander",
    "native_dpo_manifest",
    "native_backend",
    "configuration",
}


def result_payload(report, *, include_environment=None):
    """Keep scientific results and checks; record the environment once per run."""
    if include_environment is None:
        include_environment = report.get("schema") == "synkit.crn-experiments/2"
    snapshots = {
        (row["backend"], row["depth"]): row.get("snapshot_sha256")
        for row in report.get("runs", [])
        if isinstance(row, dict) and "backend" in row and "depth" in row
    }

    def clean(value, expected=None, *, root=False):
        if isinstance(value, list):
            return [clean(item, expected) for item in value]
        if not isinstance(value, dict):
            return value
        if "backend" in value and "depth" in value:
            expected = snapshots.get((value["backend"], value["depth"]), value.get("snapshot_sha256"))
        result = {}
        for key, item in value.items():
            if "sha256" in key.lower() or key in _BUILD_FIELDS:
                continue
            if key == "log" and value.get("status") == "complete":
                continue
            if key == "environment" and not (root and include_environment):
                continue
            result[key] = clean(item, expected)
        if expected is not None and "snapshot_sha256" in value:
            result["snapshot_matches"] = value["snapshot_sha256"] == expected
        snapshot = value.get("snapshot")
        if isinstance(snapshot, dict) and "sha256" in snapshot:
            contents = {key: item for key, item in snapshot.items() if key != "sha256"}
            result["snapshot_verified"] = (
                hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest() == snapshot["sha256"]
            )
        return result

    data = clean(report, root=True)
    first = ("schema", "status", "models", "networks", "modules", "growth", "measurements", "runs")
    return {
        **{key: data[key] for key in first if key in data},
        **{key: value for key, value in data.items() if key not in first and key != "checks"},
        **({"checks": data["checks"]} if "checks" in data else {}),
    }


def save_result(path, report):
    """Write a portable result without changing the in-memory analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_payload(report), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
