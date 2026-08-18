#!/usr/bin/env python
"""Run every SynKit CRN experiment behind the manuscript, in order.

Executes the validation set, the scaling sweep, the KEGG case study and the
formose case study, and optionally the BioModels interoperability probe. Each
study writes its own evidence file into the output directory; this driver
reports a combined status and exits non-zero if any study fails.

The BioModels probe is opt-in because its first run downloads from the EBI.
Everything else is offline and deterministic.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/run_all.py --output-dir results/
    python Experiment/CRN/run_all.py --with-biomodels --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN import biomodels, formose_case_study, kegg_case_study  # noqa: E402
from Experiment.CRN import scaling, validation  # noqa: E402
from Experiment.CRN.common import write_report  # noqa: E402

#: Sizes used by ``--quick``, which keeps the whole sweep under a minute.
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
    report = builder()

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
        "checks": report.get("checks", {}),
        "path": None if path is None else str(path),
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
        "--summary",
        type=Path,
        help="write the combined summary here instead of standard output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep the rule-expansion INFO logging from the formose study",
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
    ]

    if arguments.with_biomodels:
        studies.append(
            _run(
                "biomodels",
                lambda: biomodels.biomodels_report(
                    biomodels.DEFAULT_MODELS, offline=False, budget=120.0
                ),
                arguments.output_dir,
            )
        )

    summary = {
        "schema": "synkit.crn-experiments/1",
        "status": "PASS"
        if all(s["status"] == "PASS" for s in studies)
        else "FAIL",
        "studies": studies,
    }
    return write_report(summary, arguments.summary)


if __name__ == "__main__":
    raise SystemExit(main())
