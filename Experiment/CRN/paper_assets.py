#!/usr/bin/env python
"""Render manuscript tables from SynKit CRN evidence files.

Converts the JSON evidence produced by the studies in this directory into the
LaTeX table bodies used by ``paper/synkit_crn/main.tex``. Evidence with an
unexpected schema or a failing status is rejected, so a table can never be
generated from a run that did not pass.

Only the tabular rows are emitted, not the surrounding ``table`` environment,
so captions and labels stay under the manuscript's control.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/run_all.py --output-dir results/
    python Experiment/CRN/paper_assets.py results/ --output-dir paper/synkit_crn/tables/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import load_report  # noqa: E402

#: Display names for the validation networks, in manuscript order.
VALIDATION_LABELS = {
    "reversible_isomerization": "Reversible isomerization",
    "irreversible_isomerization": "Irreversible isomerization",
    "isomerization_cycle": "Isomerization cycle",
    "disjoint_reversible_blocks": "Three disjoint reversible pairs",
    "disjoint_blocks_with_stoichiometry": "Disjoint pairs, non-unit stoich.",
    "michaelis_menten": "Michaelis--Menten",
    "reversible_michaelis_menten": "Reversible Michaelis--Menten",
    "edelstein": "Edelstein",
    "futile_cycle_1_site": "Futile cycle, one site",
    "futile_cycle_2_site": "Futile cycle, two sites",
    "shinar_feinberg_acr": "Shinar--Feinberg ACR motif",
    "open_inflow_outflow": "Open reactor $0\\to A\\to B\\to 0$",
}


def _yes_no(value: Optional[bool]) -> str:
    """Render an optional boolean as a table cell.

    :param value: Value to render.
    :type value: Optional[bool]
    :return: ``"yes"``, ``"no"`` or ``"--"``.
    :rtype: str
    """
    if value is None:
        return "--"
    return "yes" if value else "no"


def _seconds(value: Optional[float]) -> str:
    """Render a timing as a table cell.

    :param value: Seconds, or ``None`` when the measurement was abandoned.
    :type value: Optional[float]
    :return: Formatted cell.
    :rtype: str
    """
    if value is None:
        return "$>$budget"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _count(value: Optional[int]) -> str:
    """Render an optional count, distinguishing zero from "not measured".

    :param value: Count, or ``None`` when the measurement was abandoned.
    :type value: Optional[int]
    :return: Formatted cell.
    :rtype: str
    """
    return "--" if value is None else str(value)


def _pathway_label(name: str) -> str:
    """Shorten a KEGG module name to its pathway phrase.

    Module names carry a trailing description after a comma
    (``"Glycolysis (Embden-Meyerhof pathway), glucose => pyruvate"``), which is
    dropped --- but only when the comma is outside parentheses, so that
    ``"Citrate cycle (TCA cycle, Krebs cycle)"`` survives intact.

    :param name: Full KEGG module name.
    :type name: str
    :return: Pathway phrase.
    :rtype: str
    """
    depth = 0
    for index, character in enumerate(name):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            return name[:index]
    return name


def validation_table(report: Dict[str, Any]) -> str:
    """Render the validation-set table body.

    :param report: Report with schema ``synkit.crn-validation/1``.
    :type report: Dict[str, Any]
    :return: LaTeX rows.
    :rtype: str
    """
    rows: List[str] = []
    for network in report["networks"]:
        computed = network["computed"]
        rows.append(
            " & ".join(
                [
                    VALIDATION_LABELS.get(network["name"], network["name"]),
                    str(computed["n_species"]),
                    str(computed["n_reactions"]),
                    str(computed["n_complexes"]),
                    str(computed["n_linkage_classes"]),
                    str(computed["rank"]),
                    str(computed["deficiency"]),
                    _yes_no(computed["weakly_reversible"]),
                    str(computed["n_conservation_laws"]),
                    _yes_no(computed["persistent"]),
                ]
            )
            + " \\\\"
        )
    return "\n".join(rows) + "\n"


def scaling_table(report: Dict[str, Any], *, family: str = "reversible_chain") -> str:
    """Render the scaling table body for one network family.

    :param report: Report with schema ``synkit.crn-scaling/1``.
    :type report: Dict[str, Any]
    :param family: Network family to tabulate.
    :type family: str
    :return: LaTeX rows.
    :rtype: str
    """
    tasks = ["rank", "crnt_summary", "conserved_moieties", "minimal_siphons",
             "canonical_form"]
    by_size: Dict[int, Dict[str, Any]] = {}
    for record in report["measurements"]:
        if record["family"] != family:
            continue
        entry = by_size.setdefault(
            record["n_species"], {"n_species": record["n_species"]}
        )
        entry[record["task"]] = record["seconds"]

    rows: List[str] = []
    for n_species in sorted(by_size):
        entry = by_size[n_species]
        rows.append(
            " & ".join(
                [str(n_species)] + [_seconds(entry.get(task)) for task in tasks]
            )
            + " \\\\"
        )
    return "\n".join(rows) + "\n"


def kegg_table(report: Dict[str, Any]) -> str:
    """Render the KEGG case-study table body.

    :param report: Report with schema ``synkit.crn-kegg/1``.
    :type report: Dict[str, Any]
    :return: LaTeX rows.
    :rtype: str
    """
    rows: List[str] = []
    for module in report["modules"]:
        if "error" in module:
            continue
        pathway = _pathway_label(module["module_name"])
        rows.append(
            " & ".join(
                [
                    module["module_id"],
                    pathway,
                    str(module["n_species"]),
                    str(module["n_reactions"]),
                    str(module["n_complexes"]),
                    str(module["n_linkage_classes"]),
                    str(module["rank"]),
                    str(module["deficiency"]),
                    _yes_no(module["weakly_reversible"]),
                    str(len(module["moieties"])),
                    str(len(module["siphons"])),
                ]
            )
            + " \\\\"
        )
    return "\n".join(rows) + "\n"


def biomodels_table(report: Dict[str, Any]) -> str:
    """Render the BioModels interoperability table body.

    :param report: Report with schema ``synkit.crn-biomodels/1``.
    :type report: Dict[str, Any]
    :return: LaTeX rows.
    :rtype: str
    """
    rows: List[str] = []
    for model in report["models"]:
        if model.get("skipped") or "error" in model:
            continue
        rows.append(
            " & ".join(
                [
                    model["model_id"],
                    str(model["n_species"]),
                    str(model["n_reactions"]),
                    _count(model.get("deficiency")),
                    _count(model.get("n_conserved_moieties")),
                    _count(model.get("n_minimal_siphons")),
                    _seconds(model.get("siphon_seconds")),
                    _yes_no(model.get("roundtrip_ok")),
                ]
            )
            + " \\\\"
        )
    return "\n".join(rows) + "\n"


RENDERERS = {
    "validation": ("synkit.crn-validation/1", validation_table),
    "scaling": ("synkit.crn-scaling/1", scaling_table),
    "kegg": ("synkit.crn-kegg/1", kegg_table),
    "biomodels": ("synkit.crn-biomodels/1", biomodels_table),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render every available evidence file into a LaTeX table body.

    :param argv: Command-line arguments, defaulting to ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "evidence_dir",
        type=Path,
        help="directory holding crn-*.json evidence files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write .tex table bodies here instead of standard output",
    )
    arguments = parser.parse_args(argv)

    rendered = 0
    for name, (schema, renderer) in RENDERERS.items():
        path = arguments.evidence_dir / f"crn-{name}.json"
        if not path.exists():
            continue

        body = renderer(load_report(path, expected_schema=schema))
        rendered += 1

        if arguments.output_dir is None:
            print(f"% --- {name} ---")
            print(body)
        else:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            destination = arguments.output_dir / f"crn-{name}.tex"
            destination.write_text(body, encoding="utf-8")
            print(f"wrote {destination}")

    if rendered == 0:
        parser.error(f"no crn-*.json evidence files found in {arguments.evidence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
