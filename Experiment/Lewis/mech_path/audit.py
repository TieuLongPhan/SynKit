#!/usr/bin/env python3
"""Reconstruct the PMechDB and RMechDB mechanism corpora safely."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import LEWIS_ROOT, sha256  # noqa: E402
from synkit.Graph.Mech.conversion import (  # noqa: E402
    duplicate_atom_maps_in_side,
    convert_reaction_arrow,
    remove_duplicate_atom_maps,
    split_ef_smirks,
)
from synkit.Mechanism.adapters import mechanism_from_legacy_epd  # noqa: E402
from synkit.Mechanism.model import (  # noqa: E402
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanisticStep,
)
from synkit.Mechanism.radical_data import (  # noqa: E402
    _split_dataset_text,
    complete_radical_aam,
    normalize_radical_row,
)
from rdkit import Chem  # noqa: E402

POLAR_DATASET = LEWIS_ROOT / "Data" / "combinatorial_all.csv"
RADICAL_DATASET = LEWIS_ROOT / "Data" / "all.csv"
DEFAULT_OUTPUT = HERE / "Data" / "reconstruction_audit"
EXPECTED = {
    "polar": {"total": 95_888, "accepted": 95_888, "failed": 0},
    "radical": {"total": 5_426, "accepted": 5_425, "failed": 1},
}
RADICAL_REVIEW_IDS = frozenset(
    (279, 404, 540, 1_310, 1_317, 2_046, 2_207, 2_300, 2_516, 3_970, 4_852)
)
RADICAL_UNRESOLVED_IDS = frozenset((2_207,))
RADICAL_FLOW_CORRECTIONS = {
    279: "2-2,14;14,15-2,14;14,15-15",
    404: "5-5,22;21,22-5,22;21,22-21",
    2_046: "10-21",
    2_516: "4-4,8;8-4,8",
    3_970: "20,21-20;20,21-21",
    4_852: "10-10,20;20,21-10,20;20,21-21",
}
RADICAL_PRODUCT_MAP_PERMUTATIONS = {
    1_310: {9: 13, 10: 14, 11: 9, 12: 10, 13: 11, 14: 12},
    1_317: {15: 19, 16: 20, 17: 15, 18: 16, 19: 17, 20: 18},
}
RADICAL_REVIEW_ROWS = (
    (279, "duplicate fishhook", "deduplicate the coupled group", "corrected"),
    (404, "5 to bond (5,21)", "5 to bond (5,22)", "corrected"),
    (540, "one-electron 1 to bond (1,2)", "two-electron 1 to bond (1,2)", "corrected"),
    (1_310, "literal endpoint maps", "retain flow after endpoint-orbit alignment", "corrected"),
    (1_317, "literal endpoint maps", "retain flow after endpoint-orbit alignment", "corrected"),
    (2_046, "21 to 10", "10 to 21", "corrected"),
    (2_207, "H transfer to map 13", "no local endpoint-consistent group", "unresolved"),
    (2_300, "four fishhooks", "two fishhooks plus two paired moves", "corrected"),
    (2_516, "9 to bond (4,8)", "8 to bond (4,8)", "corrected"),
    (3_970, "bond (20,21) to 2", "bond (20,21) to 21", "corrected"),
    (4_852, "bond (20,21) to 2", "bond (20,21) to 21", "corrected"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polar", type=Path, default=POLAR_DATASET)
    parser.add_argument("--radical", type=Path, default=RADICAL_DATASET)
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=("polar", "radical"),
        default=("polar", "radical"),
    )
    parser.add_argument("--limit", type=int, help="Pilot limit applied per corpus")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _result(
    *,
    name: str,
    dataset: Path,
    total: int,
    failure_ids: list[int],
    failure_kinds: Counter[str],
    elapsed: float,
    limit: int | None,
    repairs: Counter[str] | None = None,
) -> dict[str, Any]:
    result = {
        "corpus": name,
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": sha256(dataset),
        },
        "selection": {"limit": limit, "rows": total},
        "counts": {
            "total": total,
            "accepted": total - len(failure_ids),
            "failed": len(failure_ids),
        },
        "failure_kinds": dict(sorted(failure_kinds.items())),
        "seconds": elapsed,
    }
    if repairs is not None:
        repaired_records = repairs.get(
            "duplicate_map_records",
            sum(repairs.values()),
        )
        result["repairs"] = {
            "records": repaired_records,
            "kinds": dict(sorted(repairs.items())),
        }
    return result


def audit_polar(
    dataset: Path,
    *,
    limit: int | None = None,
    progress_every: int = 500,
) -> tuple[dict[str, Any], list[int]]:
    """Reconstruct polar records, retaining only failed logical row IDs."""
    failure_ids: list[int] = []
    failure_kinds: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    total = 0
    started = time.perf_counter()
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        handle.readline()  # Dataset provenance line precedes the CSV header.
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "SMIRKS" not in reader.fieldnames:
            raise ValueError(f"Missing PMechDB SMIRKS header in {dataset}")
        for source_row, row in enumerate(reader, start=1):
            if limit is not None and total >= limit:
                break
            total += 1
            try:
                reaction, arrow_code = split_ef_smirks(str(row["SMIRKS"]))
                reactants, products = reaction.split(">>", 1)
                if duplicate_atom_maps_in_side(
                    reactants
                ) or duplicate_atom_maps_in_side(products):
                    reaction, removed = remove_duplicate_atom_maps(reaction)
                    repairs["duplicate_map_records"] += 1
                    repairs["duplicate_map_labels_removed"] += sum(
                        sum(side.values()) for side in removed.values()
                    )
                converted = convert_reaction_arrow(
                    reaction,
                    arrow_code,
                    orbital_class=row.get("orbital pair classification"),
                )
                mechanism_from_legacy_epd(
                    converted["expanded_rsmi"],
                    converted["typed_converted"],
                    provenance={"source_row": source_row, "corpus": "PMechDB"},
                )
            except Exception as exc:
                failure_ids.append(source_row)
                failure_kinds[type(exc).__name__] += 1
            if progress_every and total % progress_every == 0:
                print(f"polar: {total}", file=sys.stderr, flush=True)
    return (
        _result(
            name="polar",
            dataset=dataset,
            total=total,
            failure_ids=failure_ids,
            failure_kinds=failure_kinds,
            elapsed=time.perf_counter() - started,
            limit=limit,
            repairs=repairs,
        ),
        failure_ids,
    )


def _replace_radical_flow(row: list[str], flow: str) -> list[str]:
    reaction, _recorded_flow = _split_dataset_text(row[0])
    return [f"{reaction} {flow}", *row[1:]]


def _permute_product_maps(row: list[str], permutation: dict[int, int]) -> list[str]:
    reaction, flow = _split_dataset_text(row[0])
    reactants, products = reaction.split(">>", 1)
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(products, parser)
    if molecule is None:
        raise ValueError("Cannot parse the radical product endpoint.")
    for atom in molecule.GetAtoms():
        atom_map = int(atom.GetAtomMapNum())
        atom.SetAtomMapNum(permutation.get(atom_map, atom_map))
    permuted = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return [f"{reactants}>>{permuted} {flow}", *row[1:]]


def _paired_radical_record_540(row: list[str], source_row: int) -> MechanismRecord:
    reaction, _flow = _split_dataset_text(row[0])
    completion = complete_radical_aam(reaction)
    if not completion.usable or completion.mapped_reaction is None:
        raise ValueError("Radical AAM completion failed for reviewed row 540.")
    converted = convert_reaction_arrow(
        completion.mapped_reaction,
        "1=2",
        expand_aam=False,
        remove_non_arrow_maps=False,
    )
    return mechanism_from_legacy_epd(
        converted["expanded_rsmi"],
        converted["typed_converted"],
        provenance={"source_row": source_row, "corpus": "RMechDB"},
    )


def _mixed_radical_record_2300(row: list[str], source_row: int) -> MechanismRecord:
    reaction, _flow = _split_dataset_text(row[0])
    completion = complete_radical_aam(reaction)
    if not completion.usable or completion.mapped_reaction is None:
        raise ValueError("Radical AAM completion failed for reviewed row 2300.")
    group_id = f"g{source_row}"
    coupling_id = f"review-{source_row}"
    moves = (
        ElectronMove(
            ElectronLocus("∙", (1,)),
            ElectronLocus("π", (1, 5)),
            1,
            "fishhook",
            group_id,
            coupling_id=coupling_id,
        ),
        ElectronMove(
            ElectronLocus("∙", (5,)),
            ElectronLocus("π", (1, 5)),
            1,
            "fishhook",
            group_id,
            coupling_id=coupling_id,
        ),
        ElectronMove(
            ElectronLocus("σ", (1, 2)),
            ElectronLocus("σ", (2, 4)),
            2,
            "curved",
            group_id,
        ),
        ElectronMove(
            ElectronLocus("σ", (4, 5)),
            ElectronLocus("σ", (1, 5)),
            2,
            "curved",
            group_id,
        ),
    )
    group = ElectronMoveGroup(group_id, moves)
    return MechanismRecord(
        completion.mapped_reaction,
        (MechanisticStep("s1", (group,)),),
        provenance={"source_row": source_row, "corpus": "RMechDB"},
    )


def _reviewed_radical_record(
    row: list[str], source_row: int
) -> tuple[MechanismRecord | None, str]:
    if source_row in RADICAL_UNRESOLVED_IDS:
        return None, "endpoint_annotation_conflict"
    if source_row == 540:
        return _paired_radical_record_540(row, source_row), "arrow_multiplicity"
    if source_row == 2_300:
        return _mixed_radical_record_2300(row, source_row), "mixed_arrow_group"
    if source_row in RADICAL_FLOW_CORRECTIONS:
        row = _replace_radical_flow(row, RADICAL_FLOW_CORRECTIONS[source_row])
        repair_kind = "flow_correction"
    elif source_row in RADICAL_PRODUCT_MAP_PERMUTATIONS:
        row = _permute_product_maps(
            row, RADICAL_PRODUCT_MAP_PERMUTATIONS[source_row]
        )
        repair_kind = "endpoint_orbit_alignment"
    else:
        repair_kind = "none"
    normalized = normalize_radical_row(
        row,
        row_number=source_row,
        enforce_source_macro=False,
    )
    return normalized.mechanism, repair_kind


def audit_radical(
    dataset: Path,
    *,
    limit: int | None = None,
    progress_every: int = 500,
) -> tuple[dict[str, Any], list[int]]:
    """Normalize radical records, retaining only failed logical row IDs."""
    failure_ids: list[int] = []
    failure_kinds: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    total = 0
    started = time.perf_counter()
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        for source_row, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if limit is not None and total >= limit:
                break
            total += 1
            try:
                mechanism, repair_kind = _reviewed_radical_record(row, source_row)
                if mechanism is None:
                    raise ValueError("No local endpoint-consistent electron-flow group.")
                if repair_kind != "none":
                    repairs[repair_kind] += 1
                    certificate = mechanism.verify()
                    if certificate.status != "VALID":
                        codes = [issue.code for issue in certificate.issues]
                        raise ValueError(
                            "Reviewed radical correction failed replay: "
                            + ",".join(codes)
                        )
            except Exception as exc:
                failure_ids.append(source_row)
                failure_kinds[type(exc).__name__] += 1
            if progress_every and total % progress_every == 0:
                print(f"radical: {total}", file=sys.stderr, flush=True)
    return (
        _result(
            name="radical",
            dataset=dataset,
            total=total,
            failure_ids=failure_ids,
            failure_kinds=failure_kinds,
            elapsed=time.perf_counter() - started,
            limit=limit,
            repairs=repairs,
        ),
        failure_ids,
    )


def _write_failure_ids(path: Path, failure_ids: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source_row",))
        writer.writerows((source_row,) for source_row in failure_ids)


def _write_radical_review(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source_row", "recorded_arrow", "reviewed_arrow", "outcome"))
        writer.writerows(RADICAL_REVIEW_ROWS)


def _validate_full_result(result: dict[str, Any]) -> None:
    name = str(result["corpus"])
    observed = result["counts"]
    expected = EXPECTED[name]
    if observed != expected:
        raise RuntimeError(
            f"{name} audit drift: expected {expected}, observed {observed}"
        )


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")

    output_dir = args.output_dir.resolve()
    targets = [output_dir / "summary.json"]
    targets.extend(output_dir / f"{name}-failures.csv" for name in args.corpora)
    if "radical" in args.corpora:
        targets.append(output_dir / "radical-arrow-review.csv")
    if not args.force and any(path.exists() for path in targets):
        existing = next(path for path in targets if path.exists())
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")

    runners = {
        "polar": lambda: audit_polar(
            args.polar.resolve(),
            limit=args.limit,
            progress_every=args.progress_every,
        ),
        "radical": lambda: audit_radical(
            args.radical.resolve(),
            limit=args.limit,
            progress_every=args.progress_every,
        ),
    }
    reports: list[dict[str, Any]] = []
    failures: dict[str, list[int]] = {}
    wall_started = time.perf_counter()
    for name in args.corpora:
        report, failure_ids = runners[name]()
        if args.limit is None:
            _validate_full_result(report)
        reports.append(report)
        failures[name] = failure_ids
        counts = report["counts"]
        print(
            f"{name}: {counts['accepted']:,}/{counts['total']:,} accepted; "
            f"{counts['failed']:,} failed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, failure_ids in failures.items():
        _write_failure_ids(output_dir / f"{name}-failures.csv", failure_ids)
    if "radical" in args.corpora:
        _write_radical_review(output_dir / "radical-arrow-review.csv")
    summary = {
        "schema": "synkit.mechanism-corpus-reconstruction-audit/1",
        "failure_output_policy": "one-based source-row IDs only",
        "full_run_expected_counts": EXPECTED,
        "corpora": reports,
        "wall_seconds": time.perf_counter() - wall_started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
