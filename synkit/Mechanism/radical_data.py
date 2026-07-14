"""Audited adapter for SynKit's partial-AAM radical mechanism dataset."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdkit import Chem

from synkit.Graph.Mech.conversion import (
    build_its_from_rsmi,
    typed_convert_arrow_code,
)

from .model import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanisticStep,
    VerificationIssue,
)

RADICAL_CLASS_TO_MACRO = {
    "homolyze": "HOMOLYSIS",
    "recombine": "RECOMBINATION",
    "addition": "RADICAL_ADDITION",
    "retroaddition": "BETA_SCISSION",
    "abstraction": "H_ABSTRACTION",
    "resonance": "RADICAL_RESONANCE",
}


@dataclass(frozen=True)
class RadicalNormalizationReport:
    row_number: int
    status: str
    source_class: str
    macro: str | None = None
    aam_expanded: bool = False
    aliases_normalized: tuple[str, ...] = ()
    issues: tuple[VerificationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "status": self.status,
            "source_class": self.source_class,
            "macro": self.macro,
            "aam_expanded": self.aam_expanded,
            "aliases_normalized": list(self.aliases_normalized),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class RadicalDatasetRecord:
    source_row: tuple[str, ...]
    report: RadicalNormalizationReport
    mechanism: MechanismRecord | None = None

    @property
    def accepted(self) -> bool:
        return self.mechanism is not None and self.report.status == "NORMALIZED"


def _split_dataset_text(text: str) -> tuple[str, str]:
    reaction, flow = text.strip().rsplit(None, 1)
    if reaction.count(">>") != 1:
        raise ValueError("Radical record must contain one reaction '>>'.")
    if "=" in flow:
        raise ValueError("NONSTANDARD_FLOW_SEPARATOR: expected '-' fishhook syntax.")
    return reaction, flow


def _mapped_atoms_complete(reaction: str) -> bool:
    for side in reaction.split(">>"):
        mol = Chem.MolFromSmiles(side, sanitize=False)
        if mol is None:
            return False
        if any(
            atom.GetAtomicNum() > 0 and atom.GetAtomMapNum() == 0
            for atom in mol.GetAtoms()
        ):
            return False
    return True


def _fishhook_move(
    typed_row: Sequence[Any], *, group_id: str, coupling_id: str, event_id: str
) -> ElectronMove:
    action, source_maps, target_maps = typed_row[:3]
    source_name, target_name = str(action).split("-/", 1)
    target_name = target_name.removesuffix("+")

    # In typed polar EPD a one-atom endpoint is called LP. In this reviewed
    # radical dataset the same shape denotes the SOMO/radical endpoint; bond
    # endpoints retain the σ/π type inferred from the ITS bond-order change.
    source_kind = "∙" if len(source_maps) == 1 else source_name
    target_kind = "∙" if len(target_maps) == 1 else target_name
    return ElectronMove(
        source=ElectronLocus(source_kind, tuple(source_maps)),
        target=ElectronLocus(target_kind, tuple(target_maps)),
        electron_count=1,
        arrow_type="fishhook",
        group_id=group_id,
        coupling_id=coupling_id,
        event_id=event_id,
        metadata={"inferred_from_typed_epd": str(action)},
    )


def normalize_radical_row(
    row: Sequence[str], *, row_number: int = 0
) -> RadicalDatasetRecord:
    """Normalize one four-column radical CSV row or return a quarantine report."""
    source_row = tuple(str(value) for value in row)
    source_class = source_row[3].strip().lower() if len(source_row) >= 4 else ""
    macro = RADICAL_CLASS_TO_MACRO.get(source_class)
    issues: list[VerificationIssue] = []

    if len(source_row) != 4:
        issues.append(
            VerificationIssue("INVALID_DATASET_ROW", "Expected four CSV columns.")
        )
    if macro is None:
        issues.append(
            VerificationIssue(
                "UNREVIEWED_RADICAL_CLASS",
                f"Radical class {source_class!r} has no approved macro mapping.",
            )
        )
    if issues:
        return RadicalDatasetRecord(
            source_row,
            RadicalNormalizationReport(
                row_number, "QUARANTINED", source_class, macro, issues=tuple(issues)
            ),
        )

    try:
        reaction, flow = _split_dataset_text(source_row[0])
        was_complete = _mapped_atoms_complete(reaction)
        arrow_code = flow.replace("-", "=")
        its, expanded, _cleaned, _diagnostics = build_its_from_rsmi(
            reaction,
            arrow_code,
            expand_aam=True,
            remove_non_arrow_maps=True,
        )
        typed = typed_convert_arrow_code(arrow_code, its)
        group_id = f"g{row_number or 1}"
        coupling_id = f"{source_class}-{row_number or 1}"
        moves = tuple(
            _fishhook_move(
                typed_row,
                group_id=group_id,
                coupling_id=coupling_id,
                event_id=f"e{index}",
            )
            for index, typed_row in enumerate(typed, start=1)
        )
        group = ElectronMoveGroup(
            group_id,
            moves,
            macro=macro,
            metadata={"source_class": source_class},
        )
        grammar_issues = group.issues()
        if grammar_issues:
            return RadicalDatasetRecord(
                source_row,
                RadicalNormalizationReport(
                    row_number,
                    "QUARANTINED",
                    source_class,
                    macro,
                    aam_expanded=not was_complete,
                    aliases_normalized=("LP→∙", "Sigma→σ", "Pi→π"),
                    issues=grammar_issues,
                ),
            )
        report = RadicalNormalizationReport(
            row_number,
            "NORMALIZED",
            source_class,
            macro,
            aam_expanded=not was_complete,
            aliases_normalized=("LP→∙", "Sigma→σ", "Pi→π"),
        )
        mechanism = MechanismRecord(
            expanded,
            (MechanisticStep("s1", (group,)),),
            provenance={
                "format": "synkit_radical_csv",
                "source_row": row_number,
                "temperature": source_row[1],
                "stage": source_row[2],
                "normalization": report.to_dict(),
            },
        )
        return RadicalDatasetRecord(source_row, report, mechanism)
    except Exception as exc:
        issue = VerificationIssue(
            "RADICAL_DATA_NORMALIZATION_FAILED",
            str(exc),
            observed=source_row[0] if source_row else None,
        )
        return RadicalDatasetRecord(
            source_row,
            RadicalNormalizationReport(
                row_number, "QUARANTINED", source_class, macro, issues=(issue,)
            ),
        )


def iter_radical_csv(path: str | Path) -> Iterable[RadicalDatasetRecord]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if row:
                yield normalize_radical_row(row, row_number=row_number)


def radical_dataset_summary(records: Iterable[RadicalDatasetRecord]) -> dict[str, Any]:
    records = list(records)
    return {
        "total": len(records),
        "accepted": sum(record.accepted for record in records),
        "quarantined": sum(not record.accepted for record in records),
        "classes": dict(Counter(record.report.source_class for record in records)),
        "expanded_aam": sum(record.report.aam_expanded for record in records),
        "issue_codes": dict(
            Counter(issue.code for record in records for issue in record.report.issues)
        ),
    }
