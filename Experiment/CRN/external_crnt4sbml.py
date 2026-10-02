#!/usr/bin/env python
"""Cross-check shared structural quantities with CRNT4SBML 0.0.15.

The comparison uses SBML, the public interchange boundary between the tools.
CRNT4SBML requires a legacy Python/dependency stack, so the caller supplies its
isolated Python executable while this process remains in the SynKit environment.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/external_crnt4sbml.py \
      --crnt4sbml-python /path/to/crnt4sbml/python \
      --output results/crn-external-crnt4sbml.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import environment, timed, write_report  # noqa: E402
from synkit.CRN.Benchmark import BENCHMARK_NETWORKS  # noqa: E402
from synkit.CRN.IO import write_sbml  # noqa: E402
from synkit.CRN.Props import crnt_summary  # noqa: E402

SCHEMA = "synkit.crn-external-crnt4sbml/1"
SHARED_FIELDS = (
    "n_species",
    "n_reactions",
    "n_complexes",
    "n_linkage_classes",
    "rank",
    "deficiency",
    "weakly_reversible",
    "linkage_class_deficiencies",
)


def compare_payload(
    synkit_rows: Dict[str, Dict[str, Any]], external: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare a worker payload with SynKit values by network name.

    :param synkit_rows: SynKit summaries keyed by benchmark name.
    :param external: Payload emitted by :mod:`crnt4sbml_worker`.
    :return: Comparison rows and aggregate checks.
    """
    external_rows = {row["name"]: row for row in external.get("networks", [])}
    rows = []
    for name, ours in synkit_rows.items():
        theirs = external_rows.get(name)
        if theirs is None:
            rows.append({"name": name, "matches": False, "error": "missing row"})
            continue
        if "error" in theirs:
            rows.append(
                {"name": name, "matches": False, "error": theirs["error"]}
            )
            continue

        differences = {
            field: {"synkit": ours[field], "crnt4sbml": theirs[field]}
            for field in SHARED_FIELDS
            if ours[field] != theirs[field]
        }
        rows.append(
            {
                "name": name,
                "matches": not differences,
                "input_sha256": ours.get("input_sha256"),
                "differences": differences,
                "synkit": {field: ours[field] for field in SHARED_FIELDS},
                "crnt4sbml": {field: theirs[field] for field in SHARED_FIELDS},
            }
        )

    names_match = set(synkit_rows) == set(external_rows)
    checks = {
        "same_network_set": names_match,
        "all_shared_structural_quantities_agree": (
            names_match and all(row["matches"] for row in rows)
        ),
    }
    return {"checks": checks, "networks": rows}


def external_report(python: Path) -> Dict[str, Any]:
    """Export benchmarks, invoke CRNT4SBML, and compare its results.

    :param python: Python executable in an environment containing CRNT4SBML.
    :return: Versioned evidence payload.
    """
    worker = Path(__file__).with_name("crnt4sbml_worker.py")
    synkit_rows: Dict[str, Dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="synkit-crnt4sbml-") as directory:
        root = Path(directory)
        manifest = {"networks": []}
        for entry in BENCHMARK_NETWORKS:
            crn = entry.build()
            summary = crnt_summary(crn).to_dict()
            summary["weakly_reversible"] = summary.pop("is_weakly_reversible")
            synkit_rows[entry.name] = summary
            path = root / f"{entry.name}.xml"
            write_sbml(crn, path, model_id=entry.name, model_name=entry.description)
            synkit_rows[entry.name]["input_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            manifest["networks"].append({"name": entry.name, "path": str(path)})

        manifest_path = root / "manifest.json"
        worker_output = root / "crnt4sbml.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        seconds, completed, error = timed(
            lambda: subprocess.run(
                [str(python), str(worker), str(manifest_path), str(worker_output)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
        )
        if completed is None:
            return {
                "schema": SCHEMA,
                "status": "FAIL",
                "error": error,
                "environment": environment(),
            }
        try:
            external = json.loads(worker_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "schema": SCHEMA,
                "status": "FAIL",
                "error": f"invalid worker output: {type(exc).__name__}: {exc}",
                "worker_stderr": completed.stderr,
                "environment": environment(),
            }

    comparison = compare_payload(synkit_rows, external)
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(comparison["checks"].values()) else "FAIL",
        "claim_boundary": (
            "This is an interoperability and agreement check for quantities "
            "both tools expose. It is not an independent validation of SynKit-only "
            "siphon, trap, canonicalization, pathway, or rule-expansion operations."
        ),
        "interface": "SBML Level 3 Version 2 core exported by SynKit",
        "environment": environment(),
        "external_environment": external.get("environment", {}),
        "external_tool": {
            "name": "CRNT4SBML",
            "version": external.get("environment", {}).get("crnt4sbml"),
            "pypi": "https://pypi.org/project/crnt4sbml/",
            "repository": (
                "https://github.com/PNNL-Comp-Mass-Spec/CRNT4SBML"
            ),
        },
        "seconds": seconds,
        **comparison,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and write the external-comparison report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--crnt4sbml-python",
        type=Path,
        required=True,
        help="Python executable in an isolated CRNT4SBML 0.0.15 environment",
    )
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args(argv)
    return write_report(external_report(arguments.crnt4sbml_python), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
