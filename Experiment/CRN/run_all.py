#!/usr/bin/env python
"""Run SynKit CRN validation, scaling and case studies from cached inputs.

Includes representation, conservation and donor controls. BioModels and the
isolated CRNT4SBML comparison are
optional. Each study writes its own evidence file. Failures are recorded without
discarding the remaining studies. Only the optional BioModels probe downloads
missing inputs, unless the caller requires its local cache.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/run_all.py --output-dir results/
    python Experiment/CRN/run_all.py --with-biomodels --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.studies import (  # noqa: E402
    biomodels,
    external_crnt4sbml,
    formose_case_study,
    glycolysis_donors,
    representation,
    supplementary,
)
from Experiment.CRN.studies import kegg_case_study  # noqa: E402
from Experiment.CRN.studies import scaling, validation  # noqa: E402
from Experiment.CRN.tools.common import (
    environment,
    source_hashes,
    write_report,
)  # noqa: E402

#: Small scaling sizes for an execution check, not the default workload.
QUICK_SIZES = (10, 20, 40)


def _run(
    name: str,
    builder: Callable[[], Dict[str, Any]],
    output_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run one study, write its evidence file, and summarize the outcome.

    :param name: Study name, used for the evidence filename.
    :type name: str
    :param builder: Callable returning the study's report payload.
    :type builder: Callable[[], Dict[str, Any]]
    :param output_dir: Directory for evidence files, or ``None`` to skip writing.
    :type output_dir: Optional[pathlib.Path]
    :return: Summary with ``name``, ``status``, ``checks`` and ``path``.
    :rtype: Dict[str, Any]
    """
    print(f"==> {name}", flush=True)
    from synkit.CRN.backend import resolve_backend
    from synkit.CRN import native_backend

    try:
        _, library = resolve_backend()
        if library is None:
            report = builder()
        else:
            with native_backend(library):
                report = builder()
        if (
            report.get("status") != "PASS"
            or report.get("complete") is False
            or not all(value is True for value in report.get("checks", {}).values())
        ):
            report["status"] = "FAIL"
    except Exception as exc:
        report = {
            "schema": "synkit.crn-study-error/1",
            "status": "FAIL",
            "checks": {"study_completed": False},
            "error": f"{type(exc).__name__}: {exc}",
        }

    path = None
    if output_dir is not None:
        path = output_dir / f"crn-{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    status = report.get("status", "UNKNOWN")
    failed = [k for k, v in report.get("checks", {}).items() if not v]
    print(f"    {status}" + (f" -- failed: {', '.join(failed)}" if failed else ""))

    return {
        "name": name,
        "status": status,
        "schema": report.get("schema"),
        "checks": report.get("checks", {}),
        "error": report.get("error"),
        "path": None if path is None else str(path.resolve()),
        "sha256": (
            None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, run every study, and report a combined status.

    :param argv: Command-line arguments, defaulting to ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write one evidence file per study into this directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use small sizes for the scaling sweep",
    )
    parser.add_argument(
        "--with-biomodels",
        action="store_true",
        help="also run the BioModels probe (downloads on first run)",
    )
    parser.add_argument(
        "--crnt4sbml-python",
        type=Path,
        help=(
            "also cross-check the validation set with the supplied isolated "
            "CRNT4SBML Python executable"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="write the combined summary here instead of standard output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep the rule-expansion INFO logging from the formose study",
    )
    parser.add_argument(
        "--offline", action="store_true", help="require cached BioModels inputs"
    )
    arguments = parser.parse_args(argv)

    if not arguments.verbose:
        logging.disable(logging.INFO)

    sizes = QUICK_SIZES if arguments.quick else scaling.DEFAULT_SIZES

    studies: List[Dict[str, Any]] = [
        _run("validation", validation.validation_report, arguments.output_dir),
        _run(
            "scaling",
            lambda: scaling.scaling_report(
                sizes=sizes,
                families=scaling.DEFAULT_FAMILIES,
                tasks=scaling.DEFAULT_TASKS,
                repeats=1,
                seed=0,
                time_budget=300.0,
            ),
            arguments.output_dir,
        ),
        _run(
            "kegg",
            lambda: kegg_case_study.kegg_report(
                kegg_case_study.CASE_STUDY_MODULES, drop_currency=True
            ),
            arguments.output_dir,
        ),
        _run(
            "formose",
            lambda: formose_case_study.formose_report([1, 2, 3, 4], analyse_at=4),
            arguments.output_dir,
        ),
        _run(
            "representation",
            lambda: representation.representation_report(
                sizes=(4, 6) if arguments.quick else (10, 14, 18, 20),
                repeats=3 if arguments.quick else 5,
            ),
            arguments.output_dir,
        ),
        _run("supplementary", supplementary.supplementary_report, arguments.output_dir),
        _run("glycolysis-donors", glycolysis_donors.donor_report, arguments.output_dir),
    ]

    omitted = {}

    if arguments.with_biomodels:
        studies.append(
            _run(
                "biomodels",
                lambda: biomodels.biomodels_report(
                    biomodels.DEFAULT_MODELS, offline=arguments.offline, budget=120.0
                ),
                arguments.output_dir,
            )
        )
    else:
        omitted["biomodels"] = (
            "enable with --with-biomodels; requires cached XML or network"
        )

    if arguments.crnt4sbml_python is not None:
        studies.append(
            _run(
                "external-crnt4sbml",
                lambda: external_crnt4sbml.external_report(arguments.crnt4sbml_python),
                arguments.output_dir,
            )
        )
    else:
        omitted["external-crnt4sbml"] = (
            "requires --crnt4sbml-python in an isolated environment"
        )

    if arguments.summary is not None:
        for study in studies:
            if study["path"] is not None:
                study["path"] = os.path.relpath(
                    study["path"], arguments.summary.resolve().parent
                )

    summary = {
        "schema": "synkit.crn-experiments/2",
        "status": "PASS" if all(s["status"] == "PASS" for s in studies) else "FAIL",
        "complete": all(s["status"] == "PASS" for s in studies),
        "profile": "quick" if arguments.quick else "default",
        "all_studies_included": not omitted,
        "omitted_studies": omitted,
        "environment": environment(),
        "source_sha256": source_hashes(),
        "result_semantics": {
            "structural_arithmetic": "exact integer and rational arithmetic",
            "positive_discrete_decisions": "verified firing certificate",
            "negative_discrete_decisions": "exhaustive finite search",
            "timings_and_svd_reference": "numerical measurements",
        },
        "studies": studies,
    }
    return write_report(summary, arguments.summary)


from Experiment.CRN.tools.common import native_experiment

main = native_experiment(main)

if __name__ == "__main__":
    raise SystemExit(main())
