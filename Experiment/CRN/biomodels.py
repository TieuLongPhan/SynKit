#!/usr/bin/env python
"""Import curated BioModels SBML files and analyse them structurally.

Answers two questions the manuscript needs: does the SBML adapter cope with
files the package did not write, and where does the analysis stack's practical
ceiling sit on *real* models rather than synthetic families.

Models are fetched from the EBI BioModels REST API and cached under
``Experiment/CRN/data/biomodels/``. The cached files are not redistributed with
SynKit --- BioModels content carries its own terms --- so the first run needs
network access and subsequent runs do not. Pass ``--offline`` to fail rather
than fetch.

For each model the study records the parsed size, the CRNT quantities, conserved
moieties, minimal siphons, structural persistence and an SBML round trip, each
under a wall-clock budget so one intractable model cannot stall the sweep.

Emits schema ``synkit.crn-biomodels/1``.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/biomodels.py --output crn-biomodels.json
    python Experiment/CRN/biomodels.py --models BIOMD0000000001 --offline
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import DATA_DIR, environment, timed, write_report  # noqa: E402
from synkit.CRN import (  # noqa: E402
    conserved_moieties,
    crn_from_sbml,
    crn_to_sbml,
    crnt_summary,
    find_siphons,
    siphon_persistence_details,
)

SCHEMA = "synkit.crn-biomodels/1"

CACHE_DIR = DATA_DIR / "biomodels"
_DOWNLOAD_URL = (
    "https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml"
)

#: A size-varied sample of curated models, from a dozen species to several
#: hundred. ``BIOMD0000000562`` is included deliberately: it is an SBML-``qual``
#: logical model with no reactions, and the adapter must say so rather than
#: silently return an empty network.
DEFAULT_MODELS: Sequence[str] = (
    "BIOMD0000000001",
    "BIOMD0000000010",
    "BIOMD0000000051",
    "BIOMD0000000064",
    "BIOMD0000000108",
    "BIOMD0000000404",
    "BIOMD0000000637",
    "BIOMD0000000019",
    "BIOMD0000000175",
    "BIOMD0000000255",
    "BIOMD0000000562",
)


def fetch(model_id: str, *, offline: bool, timeout: float = 60.0) -> Path:
    """Return the local path to a model, downloading it when absent.

    :param model_id: BioModels identifier such as ``"BIOMD0000000001"``.
    :type model_id: str
    :param offline: Whether downloading is forbidden.
    :type offline: bool
    :param timeout: Network timeout in seconds.
    :type timeout: float
    :return: Path to the cached SBML file.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the model is absent and ``offline`` is set.
    """
    path = CACHE_DIR / f"{model_id}.xml"
    if path.exists():
        return path
    if offline:
        raise FileNotFoundError(
            f"{model_id} is not cached at {path} and --offline was requested"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        _DOWNLOAD_URL.format(mid=model_id), timeout=timeout
    ) as response:
        path.write_bytes(response.read())
    return path


def analyse(path: Path, *, budget: float) -> Dict[str, Any]:
    """Import one SBML model and run the structural stack over it.

    :param path: Cached SBML file.
    :type path: pathlib.Path
    :param budget: Per-analysis wall-clock budget in seconds.
    :type budget: float
    :return: Per-model record.
    :rtype: Dict[str, Any]
    """
    record: Dict[str, Any] = {"model_id": path.stem, "bytes": path.stat().st_size}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_seconds, crn, parse_error = timed(lambda: crn_from_sbml(path))
        record["warnings"] = [str(w.message) for w in caught]

    record["parse_seconds"] = parse_seconds
    if crn is None:
        record["error"] = parse_error
        return record

    record["n_species"] = crn.n_species
    record["n_reactions"] = crn.n_reactions

    if crn.n_reactions == 0:
        # A model with no reactions is not a reaction network; record why and
        # skip the analyses rather than reporting vacuous zeros as results.
        record["skipped"] = "no reactions declared"
        return record

    seconds, summary, error = timed(lambda: crnt_summary(crn), budget=budget)
    record["crnt_seconds"] = seconds
    if summary is None:
        record["crnt_error"] = error
    else:
        record.update(
            {
                "n_complexes": summary.n_complexes,
                "n_linkage_classes": summary.n_linkage_classes,
                "rank": summary.rank,
                "deficiency": summary.deficiency,
                "weakly_reversible": summary.is_weakly_reversible,
            }
        )

    seconds, moieties, error = timed(
        lambda: conserved_moieties(crn), budget=budget
    )
    record["moiety_seconds"] = seconds
    record["n_conserved_moieties"] = None if moieties is None else len(moieties)
    if moieties is None:
        record["moiety_error"] = error

    seconds, siphons, error = timed(lambda: find_siphons(crn), budget=budget)
    record["siphon_seconds"] = seconds
    record["n_minimal_siphons"] = None if siphons is None else len(siphons)
    if siphons is None:
        record["siphon_error"] = error

    seconds, persistent, error = timed(
        lambda: siphon_persistence_details(crn).persistence_ok, budget=budget
    )
    record["persistence_seconds"] = seconds
    record["persistent"] = None if persistent is None else bool(persistent)

    seconds, roundtrip, error = timed(
        lambda: crn_from_sbml(crn_to_sbml(crn)), budget=budget
    )
    record["roundtrip_seconds"] = seconds
    record["roundtrip_ok"] = roundtrip is not None and (
        (roundtrip.n_species, roundtrip.n_reactions)
        == (crn.n_species, crn.n_reactions)
    )
    if roundtrip is None:
        record["roundtrip_error"] = error

    return record


def biomodels_report(
    models: Sequence[str],
    *,
    offline: bool,
    budget: float,
) -> Dict[str, Any]:
    """Fetch, analyse, and assemble the evidence report.

    :param models: BioModels identifiers.
    :type models: Sequence[str]
    :param offline: Whether downloading is forbidden.
    :type offline: bool
    :param budget: Per-analysis wall-clock budget in seconds.
    :type budget: float
    :return: Report payload.
    :rtype: Dict[str, Any]
    """
    records: List[Dict[str, Any]] = []
    for model_id in models:
        try:
            path = fetch(model_id, offline=offline)
        except Exception as exc:
            records.append(
                {"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        records.append(analyse(path, budget=budget))

    reaction_models = [
        r for r in records if r.get("n_reactions") and "error" not in r
    ]
    empty_models = [r for r in records if r.get("skipped")]

    checks = {
        "every_model_parsed": all("error" not in r for r in records),
        "every_reaction_model_round_trips": all(
            r.get("roundtrip_ok") for r in reaction_models
        ),
        "deficiency_identity_holds": all(
            r["deficiency"]
            == r["n_complexes"] - r["n_linkage_classes"] - r["rank"]
            for r in reaction_models
            if "deficiency" in r
        ),
        "reaction_free_models_are_flagged": all(
            r.get("warnings") for r in empty_models
        ),
    }

    def _largest(key: str) -> Optional[Dict[str, Any]]:
        candidates = [r for r in reaction_models if r.get(key) is not None]
        return (
            max(candidates, key=lambda r: r["n_species"]) if candidates else None
        )

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "This is an interoperability and scale probe over a small, "
            "hand-picked sample of curated models. It is not a systematic "
            "survey of BioModels and supports no claim about coverage of the "
            "repository."
        ),
        "environment": environment(),
        "source": {
            "repository": "EBI BioModels",
            "url": "https://www.ebi.ac.uk/biomodels/",
            "cache": str(CACHE_DIR.relative_to(REPOSITORY_ROOT)),
            "note": (
                "Cached locally and not redistributed with SynKit; BioModels "
                "content carries its own terms of use."
            ),
        },
        "parameters": {"budget_seconds": budget, "offline": offline},
        "checks": checks,
        "largest_model_completing_siphons": _largest("n_minimal_siphons"),
        "models": records,
    }


def main() -> int:
    """Parse arguments, run the sweep, and write the evidence file.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail instead of downloading models that are not cached",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=120.0,
        help="per-analysis wall-clock budget in seconds",
    )
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args()

    report = biomodels_report(
        arguments.models,
        offline=arguments.offline,
        budget=arguments.budget,
    )
    return write_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
