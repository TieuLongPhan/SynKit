"""Reproducible benchmark assembly and controlled annotation corruptions."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rdkit import Chem

from .adapters import mechanism_from_legacy_epd
from .model import ElectronMoveGroup, MechanismModelError, MechanismRecord
from .radical_data import iter_radical_csv


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    partition: str
    record: MechanismRecord
    provenance: Mapping[str, Any]
    chemistry_reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "partition": self.partition,
            "record": self.record.to_dict(),
            "provenance": dict(self.provenance),
            "chemistry_reviewed": self.chemistry_reviewed,
        }


@dataclass(frozen=True)
class CorruptedAnnotation:
    corruption: str
    payload: Mapping[str, Any]
    expected_issue_code: str
    provenance: Mapping[str, Any]

    def observed_issue_codes(self) -> tuple[str, ...]:
        """Execute this corruption through model parsing and strict replay."""
        try:
            record = MechanismRecord.from_dict(self.payload)
        except MechanismModelError as error:
            prefix = str(error).split(":", 1)[0]
            return (prefix if prefix.isupper() else "MODEL_ERROR",)

        from .replay import MechanismReplayer

        result = MechanismReplayer(verify_stereo="stepwise").replay(record)
        return tuple(issue.code for issue in result.certificate.issues)


def radical_candidates(
    path: str | Path,
    *,
    count: int = 80,
    review_path: str | Path | None = None,
) -> list[BenchmarkCase]:
    """Select deterministic, macro-balanced radical benchmark cases.

    When the canonical reviewed manifest is present, the default call returns
    its reviewed records after verifying that it describes exactly the
    deterministic candidate IDs.  An explicit ``review_path`` can still point
    to a standalone review artifact for development workflows.
    """
    buckets: dict[str, list[Any]] = {}
    for normalized in iter_radical_csv(path):
        if normalized.accepted:
            buckets.setdefault(normalized.report.macro or "UNKNOWN", []).append(
                normalized
            )
    selected = []
    names = sorted(buckets)
    offsets = Counter()
    while len(selected) < count and names:
        for name in tuple(names):
            index = offsets[name]
            if index >= len(buckets[name]):
                names.remove(name)
                continue
            normalized = buckets[name][index]
            offsets[name] += 1
            selected.append(
                BenchmarkCase(
                    case_id=f"radical-{normalized.report.row_number:05d}",
                    partition="radical",
                    record=normalized.mechanism,
                    provenance={
                        "dataset": "legacy radical CSV source pool (not vendored)",
                        "source_row": normalized.report.row_number,
                        "normalization": normalized.report.to_dict(),
                    },
                )
            )
            if len(selected) == count:
                break
    source_path = Path(path)
    if review_path is None:
        reviewed_file = source_path.parent / "radical.json"
        return (
            _load_reviewed_manifest(selected, reviewed_file)
            if reviewed_file.exists()
            else selected
        )
    review_file = Path(review_path)
    return (
        _apply_radical_review(selected, review_file)
        if review_file.exists()
        else selected
    )


def _load_reviewed_manifest(
    candidates: list[BenchmarkCase], manifest_path: Path
) -> list[BenchmarkCase]:
    """Load canonical reviewed cases only when candidate identities agree."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "MechanismBench-radical-reviewed-v1":
        raise ValueError("Unexpected radical reviewed-manifest schema.")
    serialized = payload.get("cases", [])
    expected_ids = [case.case_id for case in candidates]
    observed_ids = [case.get("case_id") for case in serialized]
    if observed_ids != expected_ids:
        raise ValueError(
            "Radical reviewed manifest does not match deterministic candidates."
        )
    return [
        BenchmarkCase(
            case_id=case["case_id"],
            partition=case["partition"],
            record=MechanismRecord.from_dict(case["record"]),
            provenance=case.get("provenance", {}),
            chemistry_reviewed=bool(case.get("chemistry_reviewed", False)),
        )
        for case in serialized
    ]


def _apply_radical_review(
    cases: list[BenchmarkCase], review_path: Path
) -> list[BenchmarkCase]:
    """Apply a human review artifact only to its exact frozen candidate set."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    identifiers = "\n".join(sorted(case.case_id for case in cases))
    digest = hashlib.sha256(identifiers.encode()).hexdigest()
    if digest != review["reviewed_case_ids_sha256"]:
        raise ValueError(
            "Radical chemistry review does not match the selected candidate set."
        )

    corrections = review.get("corrections", {})
    advisories = review.get("advisory_notes", {})
    reviewed: list[BenchmarkCase] = []
    for case in cases:
        record = case.record
        correction = corrections.get(case.case_id)
        original_reaction = record.mapped_reaction
        if correction:
            payload = record.to_dict()
            payload["mapped_reaction"] = correction["corrected_mapped_reaction"]
            payload["provenance"] = {
                **payload.get("provenance", {}),
                "original_mapped_reaction": original_reaction,
                "chemistry_correction": correction,
            }
            record = MechanismRecord.from_dict(payload)

        from .replay import MechanismReplayer

        replay = MechanismReplayer().replay(record)
        replay_valid = replay.certificate.status == "VALID"
        rule_reapplication = _audit_rule_reapplication(record)
        review_provenance = {
            "artifact": str(review_path),
            "review_date": review["review_date"],
            "reviewer": review["reviewer"],
            "decision": review["decision"],
            "corrected": correction is not None,
            "correction_reason": correction.get("reason") if correction else None,
            "advisory": advisories.get(case.case_id),
            "automated_replay": {
                "status": replay.certificate.status,
                "final_matches": replay.certificate.final_match.get("matches"),
                "issue_codes": [issue.code for issue in replay.certificate.issues],
            },
            "rule_reapplication": rule_reapplication,
        }
        reviewed.append(
            BenchmarkCase(
                case.case_id,
                case.partition,
                record,
                {**case.provenance, "chemistry_review": review_provenance},
                chemistry_reviewed=replay_valid,
            )
        )
    return reviewed


def _canonical_unmapped_smiles(text: str) -> str:
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"Could not parse benchmark reaction side: {text!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _audit_rule_reapplication(record: MechanismRecord) -> dict[str, Any]:
    """Extract an RC rule and apply it to the same reactants without atom maps."""
    from synkit.IO.chem_converter import rsmi_to_its
    from synkit.Synthesis.Reactor.syn_reactor import SynReactor

    reactants, products = record.mapped_reaction.split(">>", 1)
    host = _canonical_unmapped_smiles(reactants)
    expected = _canonical_unmapped_smiles(products)
    try:
        rule = rsmi_to_its(
            record.mapped_reaction,
            core=True,
            format="tuple",
            drop_non_aam=False,
            # ITS construction pairs reactant/product nodes by graph node ID.
            # For mapped reactions those IDs must therefore be the AAM values,
            # not each side's independent RDKit atom ordering.
            use_index_as_atom_map=True,
        )
        reactor = SynReactor(
            host,
            rule,
            explicit_h=False,
            radical_policy="strict",
        )
        generated = sorted(
            {
                _canonical_unmapped_smiles(candidate.split(">>", 1)[1])
                for candidate in reactor.smarts
            }
        )
        return {
            "status": "PASS" if expected in generated else "FAIL",
            "unmapped_reactants": host,
            "expected_product": expected,
            "mapping_count": reactor.mapping_count,
            "generated_products": generated,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "unmapped_reactants": host,
            "expected_product": expected,
            "mapping_count": 0,
            "generated_products": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def materialize_polar_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    cases_per_stratum: int = 10,
) -> None:
    """Write the reviewed, replayable polar MechanismBench selection.

    The source dataset is deliberately much larger than the public benchmark.
    Select the first source-ID-ordered record from every top-level POLAR class
    that passes strict replay and rule reapplication, continuing until each
    class has ``cases_per_stratum`` records.  The materialized manifest keeps
    the complete typed record and both audit results, so it remains executable
    even when the source pool is not checked out beside SynKit.
    """
    source = Path(source_path)
    raw_text = source.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if payload.get("schema") != "synepd.clean.polar.v1":
        raise ValueError("Unexpected polar source schema.")
    source_records = payload.get("records", [])
    if payload.get("count") != len(source_records):
        raise ValueError("Polar source count does not match its records.")

    from .replay import MechanismReplayer

    reviewed: list[BenchmarkCase] = []
    selected_by_stratum: dict[str, list[int]] = {}
    replay = MechanismReplayer()
    for stratum in sorted(
        {".".join(item["tax_code"].split(".")[:2]) for item in source_records}
    ):
        selected: list[int] = []
        for item in sorted(source_records, key=lambda item: item["id"]):
            if not item["tax_code"].startswith(f"{stratum}."):
                continue
            record = mechanism_from_legacy_epd(
                item["rsmi"],
                item["epd"],
                provenance={
                    "format": "synepd.clean.polar.v1",
                    "source_id": item["id"],
                    "tax_code": item["tax_code"],
                },
            )
            replay_result = replay.replay(record)
            if replay_result.certificate.status != "VALID":
                continue
            reapplication = _audit_rule_reapplication(record)
            if reapplication["status"] != "PASS":
                continue
            selected.append(item["id"])
            reviewed.append(
                BenchmarkCase(
                    case_id=f"polar-{item['id']:05d}",
                    partition="polar",
                    record=record,
                    provenance={
                        "source_record": {
                            key: item[key]
                            for key in (
                                "id",
                                "tax_code",
                                "entry_code",
                                "reaction_name",
                                "source_reaction_id",
                                "mechanism_id",
                                "mechanism_variant",
                            )
                        },
                        "selection_stratum": stratum,
                        "chemistry_review": {
                            "decision": "reviewed_replayable_selection",
                            "automated_replay": {
                                "status": replay_result.certificate.status,
                                "final_matches": replay_result.certificate.final_match.get(
                                    "matches"
                                ),
                                "issue_codes": [
                                    issue.code
                                    for issue in replay_result.certificate.issues
                                ],
                            },
                            "rule_reapplication": reapplication,
                        },
                    },
                    chemistry_reviewed=True,
                )
            )
            if len(selected) == cases_per_stratum:
                break
        if len(selected) != cases_per_stratum:
            raise ValueError(
                f"Could not select {cases_per_stratum} reviewed records for {stratum}."
            )
        selected_by_stratum[stratum] = selected

    source_checksum = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema": "MechanismBench-polar-reviewed-v1",
        "description": (
            "Deterministic reviewed subset of the SynEPD polar source pool; "
            "the complete 1,915-record pool is intentionally not vendored."
        ),
        "source": {
            "path": "../SynEPD/data/polar.json",
            "schema": payload["schema"],
            "record_count": payload["count"],
            "sha256": source_checksum,
            "license": "CC BY 4.0",
        },
        "selection": {
            "algorithm": (
                "For each top-level POLAR class in lexical order, take the first "
                "source-ID-ordered records that pass strict replay and exact rule "
                "reapplication."
            ),
            "cases_per_stratum": cases_per_stratum,
            "strata": selected_by_stratum,
        },
        "cases": [case.to_dict() for case in reviewed],
    }
    Path(output_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def benchmark_release_issues(cases: Iterable[BenchmarkCase]) -> tuple[str, ...]:
    """Return release-blocking benchmark findings without guessing review status."""
    cases = list(cases)
    issues = []
    counts = Counter(case.partition for case in cases)
    for partition in ("polar", "radical", "stereo"):
        if counts[partition] != 80:
            issues.append(f"PARTITION_COUNT:{partition}:{counts[partition]}/80")
    unreviewed = sum(not case.chemistry_reviewed for case in cases)
    if unreviewed:
        issues.append(f"CHEMISTRY_REVIEW_REQUIRED:{unreviewed}")
    if len({case.case_id for case in cases}) != len(cases):
        issues.append("DUPLICATE_CASE_ID")
    failed_reapplications = sum(
        case.partition == "radical"
        and case.provenance.get("chemistry_review", {})
        .get("rule_reapplication", {})
        .get("status")
        != "PASS"
        for case in cases
    )
    if failed_reapplications:
        issues.append(f"RADICAL_RULE_REAPPLICATION_FAILED:{failed_reapplications}")
    return tuple(issues)


def corrupt_record(  # noqa: C901
    record: MechanismRecord,
) -> tuple[CorruptedAnnotation, ...]:
    """Generate ten traceable negative JSON variants without bypassing model validation."""
    original = record.to_dict()
    variants = []

    def add(name: str, payload: dict[str, Any], issue: str) -> None:
        payload.setdefault("metadata", {})["corruption"] = name
        variants.append(
            CorruptedAnnotation(name, payload, issue, {"generator": "synkit-v2"})
        )

    def first_move() -> tuple[dict[str, Any], dict[str, Any]]:
        payload = deepcopy(original)
        steps = payload.setdefault("steps", [])
        if not steps:
            steps.append({"step_id": "corrupt", "groups": []})
        groups = steps[0].setdefault("groups", [])
        if not groups:
            groups.append({"group_id": "corrupt", "moves": []})
        moves = groups[0].setdefault("moves", [])
        if not moves:
            moves.append(
                {
                    "source": {"locus": "∙", "atom_maps": [1]},
                    "target": {"locus": "∙", "atom_maps": [2]},
                    "electron_count": 1,
                    "arrow_type": "fishhook",
                    "group_id": groups[0]["group_id"],
                    "coupling_id": "missing",
                    "metadata": {},
                    "event_id": "corrupt",
                }
            )
        return payload, moves[0]

    def is_fishhook_group(payload: dict[str, Any]) -> bool:
        return any(
            move["electron_count"] == 1
            for move in payload["steps"][0]["groups"][0]["moves"]
        )

    def remap_complete_group(payload: dict[str, Any]) -> None:
        """Keep a fishhook macro balanced while moving every locus off-graph."""
        moves = payload["steps"][0]["groups"][0]["moves"]
        atom_maps = sorted(
            {
                atom_map
                for move in moves
                for side in ("source", "target")
                for atom_map in move[side]["atom_maps"]
            }
        )
        remapping = {
            atom_map: 999000 + index
            for index, atom_map in enumerate(atom_maps, start=1)
        }
        for move in moves:
            for side in ("source", "target"):
                move[side]["atom_maps"] = [
                    remapping[atom_map] for atom_map in move[side]["atom_maps"]
                ]

    def reverse_complete_group(payload: dict[str, Any]) -> None:
        """Put a complete fishhook event in its product-side direction."""
        group = payload["steps"][0]["groups"][0]
        for move in group["moves"]:
            move["source"], move["target"] = move["target"], move["source"]
        macro = group.get("macro")
        if macro:
            group["macro"] = ElectronMoveGroup.REVERSE_MACRO[macro]

    payload, move = first_move()
    move["electron_count"] = 3
    add("wrong_electron_count", payload, "INVALID_ELECTRON_COUNT")
    payload, move = first_move()
    move["electron_count"] = 1
    move["arrow_type"] = "fishhook"
    move["coupling_id"] = "orphan"
    add("missing_fishhook_partner", payload, "MISSING_COUPLED_FISHHOOK")
    payload, move = first_move()
    if is_fishhook_group(payload):
        remap_complete_group(payload)
    else:
        move["source"] = {"locus": "lp", "atom_maps": [999999]}
    add("wrong_source_or_target_locus", payload, "SOURCE_LOCUS_ABSENT")
    payload, move = first_move()
    if is_fishhook_group(payload):
        reverse_complete_group(payload)
    else:
        atom_maps = sorted(
            {
                atom_map
                for locus in (move["source"], move["target"])
                for atom_map in locus["atom_maps"]
            }
        )
        source = {"locus": "σ", "atom_maps": atom_maps[:2]}
        group_id = move["group_id"]
        payload["steps"][0]["groups"][0]["moves"] = [
            {
                **deepcopy(move),
                "event_id": f"overconsumed-{index + 1}",
                "source": source,
                "target": {"locus": "lp", "atom_maps": [atom_map]},
                "electron_count": 2,
                "arrow_type": "curved",
                "group_id": group_id,
                "coupling_id": None,
            }
            for index, atom_map in enumerate(atom_maps[:2])
        ]
    add("overconsumed_locus", payload, "LOCUS_OVERCONSUMED")
    payload, move = first_move()
    if is_fishhook_group(payload):
        reverse_complete_group(payload)
    else:
        # One-step curved-arrow records have no earlier committed state to
        # reorder.  Requesting a product-side resource from that pre-state is
        # therefore represented by an absent source locus.
        move["source"] = {"locus": "lp", "atom_maps": [999999]}
    add("wrong_step_order", payload, "SOURCE_LOCUS_ABSENT")
    payload = deepcopy(original)
    reactants, products = payload["mapped_reaction"].split(">>", 1)
    payload["mapped_reaction"] = f"{reactants}>>{products}.[CH3:999999]"
    add("incorrect_radical_count", payload, "FINAL_PRODUCT_MISMATCH")
    payload = deepcopy(original)
    payload["steps"][0].setdefault("stereo_effects", []).append(
        {
            "descriptor_target": ["atom", 999999],
            "effect": "FORM",
            "before": None,
            "after": {
                "descriptor_class": "tetrahedral",
                "atoms": [999999, 1, 999997, 999998, "@H:999999"],
                "parity": -1,
                "state": "specified",
                "provenance": "corrupted",
            },
            "provenance": "corrupted",
        }
    )
    add("wrong_tetrahedral_parity", payload, "INVALID_STEREO_FORMATION")
    payload = deepcopy(original)
    payload["steps"][0].setdefault("stereo_effects", []).append(
        {
            "descriptor_target": ["bond", [999999, 999998]],
            "effect": "FORM",
            "before": None,
            "after": {
                "descriptor_class": "planar_bond",
                "atoms": [999997, "@H:999999", 999999, 999998, "@H:999998", 999996],
                "parity": 0,
                "state": "specified",
                "provenance": "corrupted",
            },
            "provenance": "corrupted",
        }
    )
    add("wrong_alkene_references", payload, "INVALID_STEREO_FORMATION")
    payload = deepcopy(original)
    steps = payload.setdefault("steps", [])
    if not steps:
        steps.append({"step_id": "corrupt", "groups": [], "metadata": {}})
    steps[0].setdefault("stereo_effects", []).append(
        {
            "descriptor_target": ["atom", 999999],
            "effect": "PRESERVE",
            "before": None,
            "after": None,
            "provenance": "corrupted",
        }
    )
    add("illegal_stereo_preservation", payload, "STEREO_TRANSITION_FROM_ABSENT")
    payload = deepcopy(original)
    reactants, products = original["mapped_reaction"].split(">>", 1)
    payload["mapped_reaction"] = f"{reactants}>>{products}.[CH4:999999]"
    add("mismatched_final_product", payload, "FINAL_PRODUCT_MISMATCH")
    return tuple(variants)


def write_candidate_manifest(cases: Iterable[BenchmarkCase], path: str | Path) -> None:
    cases = list(cases)
    fully_reviewed_radical = (
        len(cases) == 80
        and all(case.partition == "radical" for case in cases)
        and all(case.chemistry_reviewed for case in cases)
    )
    payload = {
        "schema": (
            "MechanismBench-radical-reviewed-v1"
            if fully_reviewed_radical
            else "MechanismBench-240-candidates-v1"
        ),
        "release_issues": list(benchmark_release_issues(cases)),
        "cases": [case.to_dict() for case in cases],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
