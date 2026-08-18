#!/usr/bin/env python
"""Analyse cached KEGG metabolic modules end to end.

Pushes four KEGG modules of central carbon metabolism through the whole chain:
retrieval and parsing, canonical representation, stoichiometry and CRNT,
conserved moieties, minimal siphons, structural persistence, and exact flux
realizability with a firing certificate.

The module data ship inside the package, so this study needs no network access
and its numbers do not shift when KEGG is updated. Refresh the cache
deliberately with ``refresh_kegg_cache.py`` and re-run the test suite, since
several tests assert specific metabolites and conservation laws.

Currency metabolites (ATP/ADP/AMP, NAD(P)(H), water, phosphate, protons) are
removed by default: leaving them in makes every siphon and most semiflows
describe cofactor recycling rather than the pathway's carbon skeleton.

Emits schema ``synkit.crn-kegg/1``.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/kegg_case_study.py --output crn-kegg.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import environment, timed, write_report  # noqa: E402
from synkit.CRN.Benchmark import (  # noqa: E402
    CASE_STUDY_MODULES,
    GLYCOLYSIS_FLUX,
    analyze_kegg_module,
    check_glycolysis_flux,
    load_kegg_cache,
)

SCHEMA = "synkit.crn-kegg/1"

#: Findings the study asserts, so that a silent change in KEGG or in the
#: analysis stack fails the run rather than quietly altering the manuscript.
EXPECTED_GLYCOLYSIS_MOIETY = {
    "alpha-D-Glucose",
    "alpha-D-Glucose 6-phosphate",
    "D-Fructose 6-phosphate",
    "D-Fructose 1,6-bisphosphate",
    "Glycerone phosphate",
    "D-Glyceraldehyde 3-phosphate",
    "3-Phospho-D-glyceroyl phosphate",
    "3-Phospho-D-glycerate",
    "2-Phospho-D-glycerate",
    "Phosphoenolpyruvate",
    "Pyruvate",
}


def kegg_report(
    modules: Sequence[str],
    *,
    drop_currency: bool,
) -> Dict[str, Any]:
    """Analyse each module and assemble the evidence report.

    :param modules: KEGG module identifiers to analyse.
    :type modules: Sequence[str]
    :param drop_currency: Whether currency metabolites are removed.
    :type drop_currency: bool
    :return: Report payload.
    :rtype: Dict[str, Any]
    """
    cache = load_kegg_cache()

    analyses = []
    for module_id in modules:
        seconds, result, error = timed(
            lambda mid=module_id: analyze_kegg_module(
                mid, drop_currency=drop_currency
            )
        )
        if result is None:
            analyses.append({"module_id": module_id, "error": error})
            continue
        payload = result.to_dict()
        payload["seconds"] = seconds
        analyses.append(payload)

    flux_seconds, flux, flux_error = timed(check_glycolysis_flux)
    flux_payload: Dict[str, Any] = (
        {"error": flux_error} if flux is None else dict(flux)
    )
    flux_payload["seconds"] = flux_seconds

    glycolysis = next(
        (a for a in analyses if a.get("module_id") == "M00001"), {}
    )
    moieties = [set(m) for m in glycolysis.get("moieties", [])]
    siphons = [set(s) for s in glycolysis.get("siphons", [])]

    fired: Dict[str, int] = {}
    for reaction in flux_payload.get("certificate") or []:
        fired[reaction] = fired.get(reaction, 0) + 1

    checks = {
        "every_module_analysed": all("error" not in a for a in analyses),
        "deficiency_identity_holds": all(
            a["deficiency"]
            == a["n_complexes"] - a["n_linkage_classes"] - a["rank"]
            for a in analyses
            if "error" not in a
        ),
        "glycolysis_carbon_backbone_recovered": EXPECTED_GLYCOLYSIS_MOIETY
        in moieties,
        "glucose_is_a_minimal_siphon": {"alpha-D-Glucose"} in siphons,
        "ferredoxin_couple_recovered": {
            "Oxidized ferredoxin",
            "Reduced ferredoxin",
        }
        in moieties,
        "glycolytic_flux_is_realizable": bool(flux_payload.get("realizable")),
        "certificate_matches_requested_flux": fired == dict(GLYCOLYSIS_FLUX),
    }

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "Conserved moieties, siphons and persistence are structural "
            "properties of the stoichiometry as curated by KEGG. They describe "
            "the module as written, not the in vivo pathway, and say nothing "
            "about flux magnitude, regulation or kinetics."
        ),
        "environment": environment(),
        "source": {
            "database": cache.get("source"),
            "retrieved": cache.get("retrieved"),
            "drop_currency": drop_currency,
        },
        "checks": checks,
        "modules": analyses,
        "glycolysis_flux": flux_payload,
    }


def main() -> int:
    """Parse arguments, run the study, and write the evidence file.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modules", nargs="+", default=list(CASE_STUDY_MODULES))
    parser.add_argument(
        "--keep-currency",
        action="store_true",
        help="retain ATP/NAD(H)/water and the other currency metabolites",
    )
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args()

    report = kegg_report(
        arguments.modules,
        drop_currency=not arguments.keep_currency,
    )
    return write_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
