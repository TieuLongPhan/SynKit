#!/usr/bin/env python
"""Verify saved CRN study reports against their run summary."""

import argparse
import json
from pathlib import Path

CORE_STUDIES = {
    "validation",
    "scaling",
    "kegg",
    "formose",
    "representation",
    "supplementary",
    "glycolysis-donors",
}
ALL_STUDIES = CORE_STUDIES | {"biomodels", "external-crnt4sbml", "hydrogen"}


def _check_study(row, directory):
    errors = []
    name = row["name"]
    if row.get("status") != "PASS":
        errors.append(f"{name}: study failed")
    if not row.get("path"):
        errors.append(f"{name}: report was not saved")
        return errors
    path = directory / row["path"]
    try:
        payload = path.read_bytes()
        report = json.loads(payload)
    except (OSError, ValueError) as exc:
        errors.append(f"{name}: {exc}")
        return errors
    if not row.get("schema") or report.get("schema") != row["schema"]:
        errors.append(f"{name}: report schema mismatch")
    if report.get("checks") != row.get("checks"):
        errors.append(f"{name}: checks differ from the run summary")
    if not report.get("checks"):
        errors.append(f"{name}: scientific outcome checks are absent")
    if (
        report.get("status") != "PASS"
        or report.get("complete") is False
        or not all(value is True for value in report.get("checks", {}).values())
    ):
        errors.append(f"{name}: report failed or is incomplete")
    return errors


def check_reports(summary_path, *, require_all=False):
    """Return errors for absent, inconsistent, failed or incomplete saved results.

    A quick profile verifies a reduced workload. ``require_all`` additionally
    requires the default profile and all ten studies, including external inputs.
    """
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = []
    if summary.get("schema") != "synkit.crn-experiments/2":
        return ["unsupported experiment summary schema"]
    if summary.get("status") != "PASS" or summary.get("complete") is not True:
        errors.append("experiment run failed or is incomplete")
    if summary.get("sources_unchanged_during_run") is False:
        errors.append("source files changed during the experiment run")
    studies = summary.get("studies", [])
    names = [row["name"] for row in studies]
    if len(names) != len(set(names)):
        errors.append("duplicate study names")
    missing = (ALL_STUDIES if require_all else CORE_STUDIES) - set(names)
    if missing:
        errors.append("missing studies: " + ", ".join(sorted(missing)))
    if require_all and summary.get("profile") != "default":
        errors.append("all-results verification requires the default workload")
    if require_all and (summary.get("all_studies_included") is not True or summary.get("omitted_studies")):
        errors.append("the summary records omitted studies")
    for row in studies:
        errors.extend(_check_study(row, summary_path.parent))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.json in the report directory")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="require the default workload and all ten studies",
    )
    args = parser.parse_args(argv)
    try:
        errors = check_reports(args.summary, require_all=args.require_all)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [str(exc)]
    for error in errors:
        print(error)
    if not errors:
        print("PASS: saved study reports and scientific checks verified")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
