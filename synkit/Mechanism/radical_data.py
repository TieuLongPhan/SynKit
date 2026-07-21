"""Audited adapter for SynKit's partial-AAM radical mechanism dataset."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdkit import Chem

from synkit.Graph.ITS.its_expand import ITSExpand
from synkit.Graph.Mech.conversion import (
    build_its_from_rsmi,
    typed_convert_arrow_code,
    validate_arrow_maps,
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
    "ha resonance": "LONE_PAIR_RADICAL_RELOCATION",
}


@dataclass(frozen=True)
class RadicalAAMCompletionResult:
    """Fail-closed evidence for arrow-independent radical AAM completion."""

    source_reaction: str
    mapped_reaction: str | None
    status: str
    method: str | None
    source_was_complete: bool
    all_atoms_mapped: bool = False
    unique_atom_maps: bool = False
    balanced_atom_maps: bool = False
    source_anchors_preserved: bool = False
    constitution_preserved: bool = False
    expansion_side: str | None = None
    fallback_used: bool = False
    unmapped_explicit_hydrogens_folded: bool = False
    folded_unmapped_explicit_hydrogen_count: int = 0
    explicit_hydrogen_serialization: bool = False
    stereochemistry_ignored_for_expansion: bool = False
    radical_state_preserved: bool = False
    failure_reason: str | None = None

    @property
    def usable(self) -> bool:
        """Whether the result is a validated complete atom mapping."""
        return self.status in {"ALREADY_COMPLETE", "COMPLETED"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_reaction": self.source_reaction,
            "mapped_reaction": self.mapped_reaction,
            "status": self.status,
            "method": self.method,
            "source_was_complete": self.source_was_complete,
            "usable": self.usable,
            "all_atoms_mapped": self.all_atoms_mapped,
            "unique_atom_maps": self.unique_atom_maps,
            "balanced_atom_maps": self.balanced_atom_maps,
            "source_anchors_preserved": self.source_anchors_preserved,
            "constitution_preserved": self.constitution_preserved,
            "expansion_side": self.expansion_side,
            "fallback_used": self.fallback_used,
            "unmapped_explicit_hydrogens_folded": (
                self.unmapped_explicit_hydrogens_folded
            ),
            "folded_unmapped_explicit_hydrogen_count": (
                self.folded_unmapped_explicit_hydrogen_count
            ),
            "explicit_hydrogen_serialization": (self.explicit_hydrogen_serialization),
            "stereochemistry_ignored_for_expansion": (
                self.stereochemistry_ignored_for_expansion
            ),
            "radical_state_preserved": self.radical_state_preserved,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class RadicalNormalizationReport:
    row_number: int
    status: str
    source_class: str
    macro: str | None = None
    aam_expanded: bool = False
    aam_expansion_side: str | None = None
    aam_fallback_used: bool = False
    constitution_checked: bool = False
    unmapped_explicit_hydrogens_folded: bool = False
    folded_unmapped_explicit_hydrogen_count: int = 0
    stereochemistry_ignored_for_expansion: bool = False
    radical_state_preserved: bool = False
    equivalent_map_swap: tuple[int, int] | None = None
    arrow_repair_assessment: str | None = None
    spin_assessment: str = "UNASSESSED_NO_SOURCE_EVIDENCE"
    aliases_normalized: tuple[str, ...] = ()
    issues: tuple[VerificationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "status": self.status,
            "source_class": self.source_class,
            "macro": self.macro,
            "aam_expanded": self.aam_expanded,
            "aam_expansion_side": self.aam_expansion_side,
            "aam_fallback_used": self.aam_fallback_used,
            "constitution_checked": self.constitution_checked,
            "unmapped_explicit_hydrogens_folded": (
                self.unmapped_explicit_hydrogens_folded
            ),
            "folded_unmapped_explicit_hydrogen_count": (
                self.folded_unmapped_explicit_hydrogen_count
            ),
            "stereochemistry_ignored_for_expansion": (
                self.stereochemistry_ignored_for_expansion
            ),
            "radical_state_preserved": self.radical_state_preserved,
            "equivalent_map_swap": (
                list(self.equivalent_map_swap) if self.equivalent_map_swap else None
            ),
            "arrow_repair_assessment": self.arrow_repair_assessment,
            "spin_assessment": self.spin_assessment,
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
    try:
        reaction, flow = text.strip().split(None, 1)
    except ValueError as exc:
        raise ValueError(
            "Radical record must contain reaction SMILES followed by electron flow."
        ) from exc
    if reaction.count(">>") != 1:
        raise ValueError("Radical record must contain one reaction '>>'.")
    return reaction, flow


def _normalize_dataset_flow(flow: str) -> tuple[str, tuple[str, ...]]:
    """Normalize unambiguous legacy flow separators to internal equals form."""
    steps: list[str] = []
    aliases: list[str] = []
    atom_list = r"\d+(?:\s*,\s*\d+)*"
    if any(raw_step != raw_step.strip() for raw_step in flow.split(";")):
        aliases.append("FlowWhitespace→canonical")

    for raw_step in flow.split(";"):
        step = raw_step.strip()
        if not step:
            continue

        if step.count("=") == 1 and "-" not in step:
            lhs, rhs = step.split("=", 1)
            aliases.append("FlowSeparator=→-")
        elif step.count("-") == 1 and "=" not in step:
            lhs, rhs = step.split("-", 1)
        elif step.count("-") == 2 and "=" not in step:
            match = re.fullmatch(rf"\s*({atom_list})\s*-\s*(\d+)\s*-\s*(\d+)\s*", step)
            if match is None:
                raise ValueError(
                    f"MALFORMED_FLOW_SEPARATOR: cannot normalize {step!r}."
                )
            lhs, rhs_a, rhs_b = match.groups()
            rhs = f"{rhs_a},{rhs_b}"
            aliases.append("FlowPairSeparator-→,")
        else:
            raise ValueError(f"MALFORMED_FLOW_SEPARATOR: cannot normalize {step!r}.")

        lhs = lhs.strip()
        rhs = rhs.strip()
        if not re.fullmatch(atom_list, lhs) or not re.fullmatch(atom_list, rhs):
            raise ValueError(f"MALFORMED_FLOW_LOCUS: cannot parse {step!r}.")
        steps.append(f"{lhs}={rhs}")

    if not steps:
        raise ValueError("MALFORMED_FLOW_SEPARATOR: electron flow is empty.")
    return ";".join(steps), tuple(dict.fromkeys(aliases))


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


def _side_atom_map_identities(
    smiles: str, *, require_complete: bool
) -> dict[int, tuple[int, int]]:
    """Return positive map identities while rejecting unsafe map topology."""
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.MolFromSmiles(smiles, parser)
    if mol is None:
        raise ValueError(f"Cannot parse reaction side {smiles!r}.")

    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0]
    atom_maps = [atom.GetAtomMapNum() for atom in atoms]
    if require_complete and any(atom_map <= 0 for atom_map in atom_maps):
        raise ValueError("Completed AAM contains an unmapped atom.")
    positive_maps = [atom_map for atom_map in atom_maps if atom_map > 0]
    if len(positive_maps) != len(set(positive_maps)):
        raise ValueError("A reaction side contains duplicate positive atom maps.")
    return {
        atom.GetAtomMapNum(): (atom.GetAtomicNum(), atom.GetIsotope())
        for atom in atoms
        if atom.GetAtomMapNum() > 0
    }


def _validate_completed_radical_aam(source: str, candidate: str) -> None:
    """Validate completeness, map balance, identity, anchors, and constitution."""
    if source.count(">>") != 1 or candidate.count(">>") != 1:
        raise ValueError("AAM validation requires one reaction '>>'.")
    if not ITSExpand.endpoint_constitutions_match(source, candidate):
        raise ValueError("Completed AAM changed an endpoint constitution.")

    source_sides = [
        _side_atom_map_identities(side, require_complete=False)
        for side in source.split(">>")
    ]
    candidate_sides = [
        _side_atom_map_identities(side, require_complete=True)
        for side in candidate.split(">>")
    ]
    if set(candidate_sides[0]) != set(candidate_sides[1]):
        raise ValueError("Completed AAM has different reactant and product map sets.")
    for atom_map, identity in candidate_sides[0].items():
        if candidate_sides[1][atom_map] != identity:
            raise ValueError(
                f"Completed AAM map {atom_map} changes atom identity across endpoints."
            )
    for atom_map in set(source_sides[0]) | set(source_sides[1]):
        for source_side, candidate_side in zip(source_sides, candidate_sides):
            if (
                atom_map in source_side
                and candidate_side.get(atom_map) != source_side[atom_map]
            ):
                raise ValueError(
                    f"Completed AAM moved or removed source anchor map {atom_map}."
                )


def complete_radical_aam(reaction: str) -> RadicalAAMCompletionResult:
    """Complete and validate a radical AAM without consulting electron arrows.

    Existing positive map numbers are hard anchors. Partial mappings are expanded
    from the ITS with endpoint-constitution, radical-state, and explicit-H guards.
    The returned result is usable only after all validation gates pass.
    """
    source_was_complete = _mapped_atoms_complete(reaction)
    try:
        if source_was_complete:
            candidate = reaction
            expansion = None
            status = "ALREADY_COMPLETE"
            method = "source"
        else:
            expansion = ITSExpand.expand_aam_with_its_report(
                reaction,
                preserve_older_map=True,
                fallback_to_other_side=True,
                require_constitution_preservation=True,
                fold_unmapped_explicit_hydrogens=True,
                ignore_stereochemistry=True,
                explicit_hydrogen=True,
                preserve_radical_state=True,
            )
            candidate = expansion.rsmi
            status = "COMPLETED"
            method = "its"

        _validate_completed_radical_aam(reaction, candidate)
        return RadicalAAMCompletionResult(
            reaction,
            candidate,
            status,
            method,
            source_was_complete,
            all_atoms_mapped=True,
            unique_atom_maps=True,
            balanced_atom_maps=True,
            source_anchors_preserved=True,
            constitution_preserved=True,
            expansion_side=expansion.selected_side if expansion else None,
            fallback_used=expansion.fallback_used if expansion else False,
            unmapped_explicit_hydrogens_folded=(
                expansion.unmapped_explicit_hydrogens_folded if expansion else False
            ),
            folded_unmapped_explicit_hydrogen_count=(
                expansion.folded_unmapped_explicit_hydrogen_count if expansion else 0
            ),
            explicit_hydrogen_serialization=(
                expansion.explicit_hydrogen_serialization if expansion else False
            ),
            stereochemistry_ignored_for_expansion=(
                expansion.stereochemistry_ignored_for_expansion if expansion else False
            ),
            radical_state_preserved=(
                expansion.radical_state_preserved if expansion else True
            ),
        )
    except Exception as exc:
        return RadicalAAMCompletionResult(
            reaction,
            None,
            "FAILED",
            None,
            source_was_complete,
            failure_reason=str(exc),
        )


def _positive_source_atom_maps(reaction: str) -> set[int]:
    """Collect supplied positive maps without treating completed maps as evidence."""
    atom_maps: set[int] = set()
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    for side in reaction.split(">>"):
        mol = Chem.MolFromSmiles(side, parser)
        if mol is None:
            continue
        atom_maps.update(
            atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0
        )
    return atom_maps


def _reactant_symmetry_map_pairs(
    mapped_reaction: str, *, allowed_maps: set[int]
) -> tuple[tuple[int, int], ...]:
    """Find supplied map pairs in the same RDKit reactant symmetry class."""
    reactants, _products = mapped_reaction.split(">>")
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.MolFromSmiles(reactants, parser)
    if mol is None:
        return ()

    source_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    map_free = Chem.Mol(mol)
    for atom in map_free.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = Chem.CanonicalRankAtoms(
        map_free,
        breakTies=False,
        includeChirality=True,
        includeIsotopes=True,
    )
    classes: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for atom, atom_map, rank in zip(mol.GetAtoms(), source_maps, ranks):
        if atom_map in allowed_maps:
            classes[(rank, atom.GetAtomicNum(), atom.GetIsotope())].append(atom_map)
    return tuple(
        pair
        for symmetry_class in classes.values()
        for pair in combinations(symmetry_class, 2)
    )


def _swap_reaction_atom_maps(mapped_reaction: str, atom_maps: tuple[int, int]) -> str:
    """Swap two map labels globally without changing endpoint constitution."""
    left_map, right_map = atom_maps
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    swapped_sides: list[str] = []
    for side in mapped_reaction.split(">>"):
        mol = Chem.MolFromSmiles(side, parser)
        if mol is None:
            raise ValueError(f"Cannot parse reaction side {side!r} for map swap.")
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == left_map:
                atom.SetAtomMapNum(right_map)
            elif atom.GetAtomMapNum() == right_map:
                atom.SetAtomMapNum(left_map)
        swapped_sides.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ">>".join(swapped_sides)


def _normalization_failure_code(message: str) -> str:
    """Classify fail-closed adapter errors without claiming chemical repair."""
    if "MALFORMED_FLOW_" in message:
        return "MALFORMED_ELECTRON_FLOW"
    if "missing_arrow_maps" in message:
        return "FLOW_ATOM_MAP_MISSING"
    if "ITS graph has no edge" in message:
        return "FLOW_BOND_ABSENT_FROM_ITS"
    if "Cannot align a reconstructed endpoint" in message:
        return "AAM_RADICAL_STATE_UNALIGNABLE"
    if "changed an endpoint constitution" in message:
        return "AAM_ENDPOINT_CONSTITUTION_CHANGED"
    if "Unsupported electron locus" in message:
        return "FLOW_LOCUS_UNTYPABLE"
    if "ATOM_TRANSFER_STATE_MISMATCH" in message:
        return "FLOW_ATOM_TRANSFER_STATE_MISMATCH"
    return "RADICAL_DATA_NORMALIZATION_FAILED"


def _fishhook_move(
    typed_row: Sequence[Any],
    *,
    group_id: str,
    coupling_id: str | None,
    event_id: str,
) -> ElectronMove:
    action, source_maps, target_maps = typed_row[:3]
    source_name, target_name = str(action).split("-/", 1)
    target_name = target_name.removesuffix("+")

    # In typed polar EPD a one-atom endpoint is called LP. In this reviewed
    # radical dataset the same shape denotes the SOMO/radical endpoint; bond
    # endpoints retain the σ/π type inferred from the ITS bond-order change.
    if len(source_maps) == 1 and len(target_maps) == 1:
        # State-aware atom-to-atom typing is authoritative when both compact
        # endpoints resolve to explicit nonbonding resource transitions.
        source_kind = source_name
        target_kind = target_name
    else:
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


def normalize_radical_row(  # noqa: C901
    row: Sequence[str],
    *,
    row_number: int = 0,
    enforce_source_macro: bool = True,
) -> RadicalDatasetRecord:
    """Normalize one four-column radical CSV row or return a quarantine report.

    ``enforce_source_macro=False`` validates and reconstructs the typed
    fishhook group independently of the dataset's reaction-class label. The
    state-resolved single-fishhook transition retains its operational marker
    because replay needs its paired lone-pair/radical commit semantics.
    """
    source_row = tuple(str(value) for value in row)
    source_class = source_row[3].strip().lower() if len(source_row) >= 4 else ""
    macro = RADICAL_CLASS_TO_MACRO.get(source_class)
    issues: list[VerificationIssue] = []

    if len(source_row) != 4:
        issues.append(
            VerificationIssue("INVALID_DATASET_ROW", "Expected four CSV columns.")
        )
    if enforce_source_macro and macro is None:
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

    flow_aliases: tuple[str, ...] = ()
    try:
        reaction, flow = _split_dataset_text(source_row[0])
        arrow_code, flow_aliases = _normalize_dataset_flow(flow)
        was_complete = _mapped_atoms_complete(reaction)
        # Validate flow references against supplied maps before completion.
        # Otherwise a newly assigned map could accidentally legitimize a
        # source arrow that originally named a nonexistent atom.
        validate_arrow_maps(
            reaction,
            arrow_code,
            raise_on_arrow_duplicates=True,
            raise_on_missing_arrow_maps=True,
        )
        aam_completion = complete_radical_aam(reaction)
        if not aam_completion.usable or aam_completion.mapped_reaction is None:
            raise ValueError(
                "Arrow-independent AAM completion failed before flow typing: "
                f"{aam_completion.failure_reason}"
            )
        equivalent_map_swap: tuple[int, int] | None = None
        arrow_repair_assessment: str | None = None
        try:
            its, expanded, _cleaned, _diagnostics = build_its_from_rsmi(
                aam_completion.mapped_reaction,
                arrow_code,
                expand_aam=False,
                remove_non_arrow_maps=False,
            )
            typed = typed_convert_arrow_code(arrow_code, its, electron_count=1)
        except Exception as arrow_exc:
            if "ITS graph has no edge" not in str(arrow_exc):
                raise
            repair_candidates = []
            allowed_maps = _positive_source_atom_maps(reaction)
            for atom_map_pair in _reactant_symmetry_map_pairs(
                aam_completion.mapped_reaction,
                allowed_maps=allowed_maps,
            ):
                swapped = _swap_reaction_atom_maps(
                    aam_completion.mapped_reaction, atom_map_pair
                )
                if not ITSExpand.endpoint_constitutions_match(
                    aam_completion.mapped_reaction, swapped
                ):
                    continue
                try:
                    candidate_its, candidate_rsmi, candidate_cleaned, candidate_diag = (
                        build_its_from_rsmi(
                            swapped,
                            arrow_code,
                            expand_aam=False,
                            remove_non_arrow_maps=False,
                        )
                    )
                    candidate_typed = typed_convert_arrow_code(
                        arrow_code, candidate_its, electron_count=1
                    )
                except Exception:
                    continue
                repair_candidates.append(
                    (
                        atom_map_pair,
                        candidate_its,
                        candidate_rsmi,
                        candidate_cleaned,
                        candidate_diag,
                        candidate_typed,
                    )
                )
            if len(repair_candidates) != 1:
                raise arrow_exc
            (
                equivalent_map_swap,
                its,
                expanded,
                _cleaned,
                _diagnostics,
                typed,
            ) = repair_candidates[0]
            arrow_repair_assessment = "REACTANT_SYMMETRY_UNIQUE"
        expansion_side = aam_completion.expansion_side
        fallback_used = aam_completion.fallback_used
        constitution_checked = aam_completion.constitution_preserved
        hydrogens_folded = aam_completion.unmapped_explicit_hydrogens_folded
        folded_hydrogen_count = aam_completion.folded_unmapped_explicit_hydrogen_count
        stereochemistry_ignored = aam_completion.stereochemistry_ignored_for_expansion
        radical_state_preserved = aam_completion.radical_state_preserved
        is_state_relocation = (
            len(typed) == 1
            and len(typed[0][1]) == 1
            and len(typed[0][2]) == 1
            and str(typed[0][0]).lower().startswith("lp-/lp")
        )
        operational_macro = (
            "LONE_PAIR_RADICAL_RELOCATION"
            if is_state_relocation
            else macro if enforce_source_macro else None
        )
        aliases = (
            "LP→∙",
            "Sigma→σ",
            "Pi→π",
            *flow_aliases,
            *(
                ("CompactAtomArrow→state-resolved:lp→lp",)
                if is_state_relocation
                else ()
            ),
            *(("Stereo→constitution-only",) if stereochemistry_ignored else ()),
            *(
                (
                    "EquivalentReactantMapSwap:"
                    f"{equivalent_map_swap[0]}↔{equivalent_map_swap[1]}",
                )
                if equivalent_map_swap
                else ()
            ),
        )
        group_id = f"g{row_number or 1}"
        coupling_id = (
            None
            if operational_macro == "LONE_PAIR_RADICAL_RELOCATION"
            else f"{source_class}-{row_number or 1}"
        )
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
            macro=operational_macro,
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
                    aam_expansion_side=expansion_side,
                    aam_fallback_used=fallback_used,
                    constitution_checked=constitution_checked,
                    unmapped_explicit_hydrogens_folded=hydrogens_folded,
                    folded_unmapped_explicit_hydrogen_count=folded_hydrogen_count,
                    stereochemistry_ignored_for_expansion=stereochemistry_ignored,
                    radical_state_preserved=radical_state_preserved,
                    equivalent_map_swap=equivalent_map_swap,
                    arrow_repair_assessment=arrow_repair_assessment,
                    aliases_normalized=aliases,
                    issues=grammar_issues,
                ),
            )
        report = RadicalNormalizationReport(
            row_number,
            "NORMALIZED",
            source_class,
            macro,
            aam_expanded=not was_complete,
            aam_expansion_side=expansion_side,
            aam_fallback_used=fallback_used,
            constitution_checked=constitution_checked,
            unmapped_explicit_hydrogens_folded=hydrogens_folded,
            folded_unmapped_explicit_hydrogen_count=folded_hydrogen_count,
            stereochemistry_ignored_for_expansion=stereochemistry_ignored,
            radical_state_preserved=radical_state_preserved,
            equivalent_map_swap=equivalent_map_swap,
            arrow_repair_assessment=arrow_repair_assessment,
            aliases_normalized=aliases,
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
        message = str(exc)
        issue = VerificationIssue(
            _normalization_failure_code(message),
            message,
            observed=source_row[0] if source_row else None,
        )
        return RadicalDatasetRecord(
            source_row,
            RadicalNormalizationReport(
                row_number,
                "QUARANTINED",
                source_class,
                macro,
                aliases_normalized=flow_aliases,
                issues=(issue,),
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
