#!/usr/bin/env python
"""Audit BioModels participant conventions against independent XML CRNT values.

The three structural readings retain reaction-side participants, cancel shared
stoichiometry, or add every modifier to both sides with coefficient one. Raw
reversible flags are passed directly through the public SynKit importer; this
study deliberately does not pre-expand reversible reactions. Modifier addition
is a controlled structural experiment, not an interpretation of regulatory
kinetics. Inputs are checked against the packaged BioModels manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.studies.sbml_reference import (  # noqa: E402
    child,
    crnt_from_sbml,
    document_root,
    local_name,
    model_element,
    reaction_side,
    stoichiometric_columns,
)
from Experiment.CRN.studies.biomodels import DEFAULT_MODELS, fetch  # noqa: E402
from Experiment.CRN.tools.common import (  # noqa: E402
    DATA_DIR,
    environment,
    native_experiment,
    source_hashes,
    timed,
    write_report,
)
from synkit.CRN import crn_from_sbml, crnt_summary  # noqa: E402

SCHEMA = "synkit.crn-participant-conventions/1"
VARIANTS = ("kept", "cancelled", "modifiers")
SHARED_FIELDS = (
    "n_species",
    "n_reactions",
    "n_complexes",
    "n_linkage_classes",
    "rank",
    "deficiency",
    "n_conservation_laws",
    "weakly_reversible",
)


def _replace_side(reaction, name, coefficients):
    listing = child(reaction, name)
    namespace = reaction.tag.partition("}")[0] + "}" if "}" in reaction.tag else ""
    if listing is None:
        listing = ET.SubElement(reaction, namespace + name)
    for item in list(listing):
        if local_name(item) == "speciesReference":
            listing.remove(item)
    for species, value in sorted(coefficients.items()):
        if not value:
            continue
        if value.denominator != 1:
            raise ValueError("Participant variants require static integer coefficients")
        ET.SubElement(
            listing,
            namespace + "speciesReference",
            {
                "species": species,
                "stoichiometry": str(value.numerator),
            },
        )


def participant_variant(source, mode):
    """Build a controlled XML variant while retaining reversible flags."""
    if mode not in VARIANTS:
        raise ValueError(f"Unknown participant convention {mode!r}")
    root = copy.deepcopy(document_root(source))
    model = model_element(root)
    listing = child(model, "listOfReactions")
    changed = 0
    for reaction in [] if listing is None else listing:
        if local_name(reaction) != "reaction" or mode == "kept":
            continue
        left = reaction_side(reaction, "listOfReactants")
        right = reaction_side(reaction, "listOfProducts")
        if mode == "cancelled":
            for species in set(left) & set(right):
                shared = min(left[species], right[species])
                left[species] -= shared
                right[species] -= shared
                changed += 1
        else:
            modifiers = child(reaction, "listOfModifiers")
            for reference in [] if modifiers is None else modifiers:
                if local_name(reference) == "modifierSpeciesReference":
                    species = reference.get("species")
                    if not species:
                        raise ValueError("Modifier reference has no species identifier")
                    left[species] = left.get(species, 0) + 1
                    right[species] = right.get(species, 0) + 1
                    changed += 1
        # Convert Python integers introduced by modifier addition to Fractions.
        from fractions import Fraction

        _replace_side(reaction, "listOfReactants", {key: Fraction(value) for key, value in left.items()})
        _replace_side(reaction, "listOfProducts", {key: Fraction(value) for key, value in right.items()})
    return root, changed


def analyse_participant_conventions(path, *, budget=120.0):
    """Compare all three variants with the independent rational XML audit."""
    path = Path(path)
    payload = path.read_bytes()
    root = document_root(payload)
    original_columns = stoichiometric_columns(root)
    record = {
        "model_id": path.stem,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "n_modifier_refs": sum(local_name(item) == "modifierSpeciesReference" for item in root.iter()),
        "variants": {},
    }
    for mode in VARIANTS:
        variant, changed = participant_variant(root, mode)
        started = perf_counter()
        reference = crnt_from_sbml(variant)
        reference_seconds = perf_counter() - started
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = crn_from_sbml(variant)
        if reference["n_reactions"] == 0:
            record.update(
                {
                    "reaction_free": True,
                    "n_species": reference["n_species"],
                    "n_reactions": 0,
                    "warnings": [str(item.message) for item in caught],
                    "independent_reference": reference,
                    "agrees": model.n_reactions == 0 and model.n_species == reference["n_species"],
                    "net_stoichiometry_unchanged": True,
                }
            )
            return record
        seconds, summary, error = timed(lambda: crnt_summary(model), budget=budget)
        if error is not None:
            raise RuntimeError(f"{path.stem}/{mode}: {error}")
        actual = {
            field: getattr(summary, field)
            for field in SHARED_FIELDS
            if field not in {"n_conservation_laws", "weakly_reversible"}
        }
        actual["n_conservation_laws"] = summary.n_species - summary.rank
        actual["weakly_reversible"] = summary.is_weakly_reversible
        differences = {
            field: {"synkit": actual[field], "xml_reference": reference[field]}
            for field in SHARED_FIELDS
            if actual[field] != reference[field]
        }
        unchanged = original_columns == stoichiometric_columns(variant)
        record["variants"][mode] = {
            "independent_reference": reference,
            "synkit": actual,
            "differences": differences,
            "agrees": not differences,
            "net_stoichiometry_unchanged": unchanged,
            "reference_seconds": reference_seconds,
            "synkit_seconds": seconds,
            "warnings": [str(item.message) for item in caught],
        }
        record[f"deficiency_{mode}"] = actual["deficiency"]
        if mode == "kept":
            record.update({"n_species": actual["n_species"], "n_reactions": actual["n_reactions"]})
        elif mode == "cancelled":
            record["n_cancelled_participants"] = changed
        else:
            record["n_added_modifier_participants"] = changed
    record["agrees"] = all(row["agrees"] for row in record["variants"].values())
    record["net_stoichiometry_unchanged"] = all(
        row["net_stoichiometry_unchanged"] for row in record["variants"].values()
    )
    return record


def participant_conventions_report(models=DEFAULT_MODELS, *, offline=False, budget=120.0):
    """Assemble a complete checksum-verified report, preserving failures."""
    manifest_path = DATA_DIR / "biomodels_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = source_hashes()
    records = []
    for model_id in models:
        try:
            expected = manifest["models"][model_id]
            path = fetch(model_id, offline=offline)
            payload = path.read_bytes()
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("Input differs from the declared BioModels snapshot")
            record = analyse_participant_conventions(path, budget=budget)
            record["input_verified"] = True
            records.append(record)
        except Exception as error:
            records.append({"model_id": model_id, "error": f"{type(error).__name__}: {error}"})
    checks = {
        "requested_models_present": bool(models) and len(records) == len(models) and len(set(models)) == len(models),
        "all_inputs_match_manifest": all(row.get("input_verified") for row in records),
        "all_models_complete": all("error" not in row for row in records),
        "all_shared_structural_quantities_agree": all(row.get("agrees") for row in records),
        "net_stoichiometry_unchanged": all(row.get("net_stoichiometry_unchanged") for row in records),
        "reaction_free_models_flagged": all(row.get("warnings") for row in records if row.get("reaction_free")),
        "source_unchanged": sources == source_hashes(),
    }
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "date": date.today().isoformat(),
        "source": "Experiment/CRN/studies/participant_conventions.py",
        "source_sha256": sources,
        "environment": environment(),
        "inputs": {
            "manifest": str(manifest_path.relative_to(REPOSITORY_ROOT)),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "offline": offline,
            "models": list(models),
        },
        "reversible_policy": (
            "Level 2 omitted flags default to true; Level 3 requires an explicit flag. "
            "Raw XML is imported without pre-expanding reversible reactions."
        ),
        "claim_boundary": (
            "Exact static reaction structure only. Modifier addition is a controlled "
            "participant-convention experiment; modifiers can encode inhibition and "
            "activation and are not automatically interpreted as catalysts."
        ),
        "checks": checks,
        "models": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    return write_report(
        participant_conventions_report(
            arguments.models,
            offline=arguments.offline,
            budget=arguments.budget,
        ),
        arguments.output,
    )


main = native_experiment(main)

if __name__ == "__main__":
    raise SystemExit(main())
