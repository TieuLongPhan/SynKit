#!/usr/bin/env python3
"""Compare full-corpus partial-AAM completion methods.

For the general PartialAAMs corpus, every candidate is evaluated against the
independent fully mapped ``smart`` reference with the same current
``AAMValidator``.  The radical corpus has no independent full-AAM reference,
so it is compared on disclosed completion postconditions: complete and unique
maps, balanced map sets, atom identity, endpoint constitution, and radical
state.  Source-anchor preservation is recorded separately and is not an
accuracy/acceptance condition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import signal
import sys
import time
import tomllib
from typing import Any, Callable

from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    POLAR_DATASET,
    RADICAL_DATASET,
    open_text,
    read_json,
    sha256,
    timing_summary,
    write_json,
)
from synkit.Chem.Reaction import AAMValidator  # noqa: E402
from synkit.Graph.ITS.its_expand import ITSExpand  # noqa: E402
from synkit.Mechanism.radical_data import complete_radical_aam  # noqa: E402

METHODS = ("synkit", "synkit_rsmi", "synkit_noanchors", "gm", "rb1", "rb2")
METHOD_LABELS = {
    "synkit": "SynKit guarded ITS completion",
    "synkit_rsmi": "SynKit minimal RSMI ITS completion",
    "synkit_noanchors": "SynKit ITS completion without anchor retention",
    "gm": "GM",
    "rb1": "RB1 (PartialAAMs extend)",
    "rb2": "RB2 (PartialAAMs extend_g)",
}
DEFAULT_DEPENDENCIES = Path("/tmp/synkit-sprint25-partialaam-deps")
DEFAULT_GMAPACHE = Path("/tmp/synkit-sprint25-gmapache")
DEFAULT_PARTIALAAMS = Path("/tmp/PartialAAMs")


class CaseTimeout(TimeoutError):
    """Raised when one completion exceeds the disclosed wall ceiling."""


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _synkit_version() -> str | None:
    installed = _package_version("synkit")
    if installed is not None:
        return installed
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        return tomllib.loads(pyproject.read_text())["project"]["version"]
    return None


def _raise_timeout(_signum, _frame) -> None:
    raise CaseTimeout("Partial-AAM completion exceeded the per-case time ceiling")


def _has_radical(reaction: str) -> bool:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side, parser)
        if molecule is None:
            raise ValueError("RDKit rejected a reaction endpoint")
        if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
            return True
    return False


def _canonical_radical_endpoint(side: str) -> str:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(side, parser)
    if molecule is None:
        raise ValueError(f"RDKit rejected endpoint {side!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _side_maps(
    side: str,
    *,
    require_complete: bool,
) -> tuple[dict[int, tuple[int, int]], bool, bool]:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(side, parser)
    if molecule is None:
        raise ValueError(f"RDKit rejected endpoint {side!r}")
    atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0]
    maps = [atom.GetAtomMapNum() for atom in atoms]
    complete = not any(atom_map <= 0 for atom_map in maps)
    positive = [atom_map for atom_map in maps if atom_map > 0]
    unique = len(positive) == len(set(positive))
    identities = {
        atom.GetAtomMapNum(): (atom.GetAtomicNum(), atom.GetIsotope())
        for atom in atoms
        if atom.GetAtomMapNum() > 0
    }
    return identities, unique, complete or not require_complete


def _anchors_preserved(source_side: str, candidate_side: str) -> bool:
    """Test anchors through an endpoint isomorphism, allowing atom reordering."""
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    source = Chem.MolFromSmiles(source_side, parser)
    candidate = Chem.MolFromSmiles(candidate_side, parser)
    if source is None or candidate is None:
        raise ValueError("RDKit rejected a reaction endpoint")
    if source.GetNumAtoms() != candidate.GetNumAtoms():
        return False
    anchors = [
        (atom.GetIdx(), atom.GetAtomMapNum())
        for atom in source.GetAtoms()
        if atom.GetAtomMapNum() > 0
    ]
    if not anchors:
        return True
    source_query = Chem.Mol(source)
    candidate_target = Chem.Mol(candidate)
    for atom in source_query.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom in candidate_target.GetAtoms():
        atom.SetAtomMapNum(0)
    matches = candidate_target.GetSubstructMatches(
        source_query,
        uniquify=False,
        useChirality=False,
        maxMatches=100000,
    )
    return any(
        all(
            candidate.GetAtomWithIdx(match[source_index]).GetAtomMapNum() == atom_map
            for source_index, atom_map in anchors
        )
        for match in matches
    )


def completion_gates(source: str, candidate: str) -> dict[str, bool]:
    """Evaluate method-independent structural completion postconditions."""
    if source.count(">>") != 1 or candidate.count(">>") != 1:
        raise ValueError("Expected one reaction separator")
    source_sides = source.split(">>")
    candidate_sides = candidate.split(">>")
    candidate_results = [
        _side_maps(side, require_complete=True) for side in candidate_sides
    ]
    candidate_maps = [item[0] for item in candidate_results]
    complete = all(item[2] for item in candidate_results)
    unique = complete and all(item[1] for item in candidate_results)
    balanced = unique and set(candidate_maps[0]) == set(candidate_maps[1])
    identity = balanced and all(
        candidate_maps[0][atom_map] == candidate_maps[1][atom_map]
        for atom_map in candidate_maps[0]
    )
    anchors = unique and all(
        _anchors_preserved(source_side, candidate_side)
        for source_side, candidate_side in zip(source_sides, candidate_sides)
    )
    constitution = ITSExpand.endpoint_constitutions_match(source, candidate)
    source_radicals = [_canonical_radical_endpoint(side) for side in source_sides]
    candidate_radicals = [_canonical_radical_endpoint(side) for side in candidate_sides]
    radical_state = source_radicals == candidate_radicals
    return {
        "all_atoms_mapped": complete,
        "unique_atom_maps": unique,
        "balanced_atom_maps": balanced,
        "atom_identity_preserved": identity,
        "source_anchors_preserved": anchors,
        "endpoint_constitution_preserved": constitution,
        "radical_state_preserved": radical_state,
    }


def _general_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "record_id": int(row["R-id"]),
            "source": str(row["partial"]),
            "reference": str(row["smart"]),
        }
        for row in read_json(path)
    ]


def _radical_rows(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if row:
                records.append(
                    {
                        "record_id": row_number,
                        "source": row[0].strip().split(None, 1)[0],
                        "reference": None,
                    }
                )
    return records


def _synkit_general(source: str) -> str:
    result = ITSExpand.expand_aam_with_its_report(
        source,
        preserve_older_map=True,
        fallback_to_other_side=True,
        require_constitution_preservation=True,
        fold_unmapped_explicit_hydrogens=True,
        ignore_stereochemistry=True,
        explicit_hydrogen=True,
        preserve_radical_state=_has_radical(source),
        constitutional_only=True,
    )
    return result.rsmi


def _synkit_general_noanchors(source: str) -> str:
    """Run the same guarded completion without preserving supplied map labels."""
    result = ITSExpand.expand_aam_with_its_report(
        source,
        preserve_older_map=False,
        fallback_to_other_side=True,
        require_constitution_preservation=True,
        fold_unmapped_explicit_hydrogens=True,
        ignore_stereochemistry=True,
        explicit_hydrogen=True,
        preserve_radical_state=_has_radical(source),
        constitutional_only=True,
    )
    return result.rsmi


def _synkit_general_rsmi(source: str) -> str:
    """Time only the minimal RSMI adapter around mapped RC reconstruction.

    Independent ITS/RC and completion gates are still evaluated by the common
    benchmark harness after the generation timer stops. This profile excludes
    anchor retention, H folding, stereo normalization, radical transport,
    opposite-side fallback, and the post-expansion constitution guard.
    """
    return ITSExpand.expand_rsmi(source)


def _synkit_radical(source: str) -> str:
    result = complete_radical_aam(source)
    if not result.usable or result.mapped_reaction is None:
        raise ValueError(result.failure_reason or "SynKit radical completion failed")
    return result.mapped_reaction


def load_methods(
    suite: str,
    selected: list[str],
    dependencies: Path,
    gmapache: Path,
    partialaams: Path,
) -> dict[str, Callable[[str], str]]:
    methods: dict[str, Callable[[str], str]] = {
        "synkit": _synkit_radical if suite == "radical" else _synkit_general,
        "synkit_rsmi": _synkit_general_rsmi,
        "synkit_noanchors": (
            _synkit_radical if suite == "radical" else _synkit_general_noanchors
        ),
    }
    if suite == "radical" and "synkit_rsmi" in selected:
        raise ValueError("synkit_rsmi is defined only for the general corpus")
    if set(selected).issubset({"synkit", "synkit_rsmi", "synkit_noanchors"}):
        return methods
    for path in (dependencies, gmapache, partialaams):
        if not path.exists():
            raise FileNotFoundError(path)
        if str(path) not in sys.path:
            sys.path.append(str(path))
    from partialaams.extender import Extender
    from partialaams.gm_expand import gm_extend_aam_from_rsmi

    extender = Extender()
    methods.update(
        gm=gm_extend_aam_from_rsmi,
        rb1=extender.fit,
        rb2=lambda source: extender.fit(source, use_gm=True),
    )
    return methods


def benchmark_method(
    name: str,
    function: Callable[[str], str],
    rows: list[dict[str, Any]],
    suite: str,
    case_handle,
    progress_every: int,
    case_timeout: float,
) -> dict[str, Any]:
    validator = AAMValidator(strip_unbalanced_maps=True) if suite == "general" else None
    counts: Counter[str] = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        record_id = int(row["record_id"])
        source = str(row["source"])
        case: dict[str, Any] = {"method": name, "record_id": record_id}
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, case_timeout)
        generation_started = time.perf_counter()
        generation_seconds: float | None = None
        try:
            candidate = function(source)
            generation_seconds = time.perf_counter() - generation_started
            timings["generation"].append(generation_seconds)
            if not isinstance(candidate, str) or candidate.count(">>") != 1:
                raise ValueError("Method did not return a reaction SMILES")
            counts["output"] += 1
            validation_started = time.perf_counter()
            gates = completion_gates(source, candidate)
            for gate, passed in gates.items():
                counts[f"{gate}:{str(passed).lower()}"] += 1
            verdicts: dict[str, bool] | None = None
            if validator is not None:
                verdicts = {
                    key: bool(value)
                    for key, value in validator.smiles_checks(
                        candidate,
                        str(row["reference"]),
                        constitutional_only=True,
                    ).items()
                    if key in {"ITS", "RC"}
                }
                for check, passed in verdicts.items():
                    counts[f"aam_validator_{check.lower()}:{str(passed).lower()}"] += 1
            validation_seconds = time.perf_counter() - validation_started
            timings["validation"].append(validation_seconds)
            extension_valid_without_anchors = all(
                gate_passed
                for gate, gate_passed in gates.items()
                if gate != "source_anchors_preserved"
            )
            anchor_preserving_extension_valid = all(gates.values())
            counts[
                "extension_valid_without_anchors:"
                f"{str(extension_valid_without_anchors).lower()}"
            ] += 1
            counts[
                "anchor_preserving_extension_valid:"
                f"{str(anchor_preserving_extension_valid).lower()}"
            ] += 1
            passed = (
                extension_valid_without_anchors
                if verdicts is None
                else all(verdicts.values())
            )
            counts[f"accepted:{str(passed).lower()}"] += 1
            case.update(
                status="PASS" if passed else "FAIL",
                generation_seconds=generation_seconds,
                validation_seconds=validation_seconds,
                gates=gates,
                reference_checks=verdicts,
                extension_valid_without_anchors=extension_valid_without_anchors,
                anchor_preserving_extension_valid=anchor_preserving_extension_valid,
            )
            if not passed:
                case["candidate"] = candidate
        except Exception as exc:
            if generation_seconds is None:
                generation_seconds = time.perf_counter() - generation_started
                timings["generation"].append(generation_seconds)
            counts["error"] += 1
            if isinstance(exc, CaseTimeout):
                counts["timeout"] += 1
            case.update(
                status="ERROR",
                error_type=type(exc).__name__,
                message=str(exc),
                generation_seconds=generation_seconds,
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
        case_handle.write(json.dumps(case, sort_keys=True) + "\n")
        if progress_every and index % progress_every == 0:
            print(
                f"expansion {suite}/{name}: {index}/{len(rows)}",
                file=sys.stderr,
                flush=True,
            )
    wall = time.perf_counter() - started
    return {
        "method": name,
        "label": METHOD_LABELS[name],
        "counts": dict(sorted(counts.items())),
        "timing_seconds": {
            "wall": wall,
            "rows_per_second": len(rows) / wall if wall else None,
            "stages": {
                key: timing_summary(values) for key, values in sorted(timings.items())
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("general", "radical"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dependencies", type=Path, default=DEFAULT_DEPENDENCIES)
    parser.add_argument("--gmapache", type=Path, default=DEFAULT_GMAPACHE)
    parser.add_argument("--partialaams", type=Path, default=DEFAULT_PARTIALAAMS)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()

    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        parser.error("offset must be nonnegative and limit must be positive")
    RDLogger.DisableLog("rdApp.*")
    dataset = (
        args.dataset or (RADICAL_DATASET if args.suite == "radical" else POLAR_DATASET)
    ).resolve()
    all_rows = (
        _radical_rows(dataset) if args.suite == "radical" else _general_rows(dataset)
    )
    rows = all_rows[
        args.offset : None if args.limit is None else args.offset + args.limit
    ]
    methods = load_methods(
        args.suite,
        args.methods,
        args.dependencies.resolve(),
        args.gmapache.resolve(),
        args.partialaams.resolve(),
    )
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    with open_text(args.cases, "wt") as case_handle:
        reports = [
            benchmark_method(
                name,
                methods[name],
                rows,
                args.suite,
                case_handle,
                args.progress_every,
                args.case_timeout,
            )
            for name in args.methods
        ]
    report = {
        "schema": "synkit.partial-aam-method-comparison/1",
        "suite": args.suite,
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "selection": {
            "offset": args.offset,
            "limit": args.limit,
            "rows": len(rows),
            "all_dataset_rows": len(all_rows),
        },
        "reference_policy": (
            {
                "independent_full_aam": True,
                "reference_field": "smart",
                "checks": ["AAMValidator.ITS", "AAMValidator.RC"],
            }
            if args.suite == "general"
            else {
                "independent_full_aam": False,
                "checks": [
                    "structural completion gates excluding source-anchor preservation",
                    "radical-state preservation",
                ],
                "separate_metric": "source_anchors_preserved",
                "warning": "Radical results are completion validity, not AAM accuracy.",
            }
        ),
        "case_timeout_seconds": args.case_timeout,
        "methods": reports,
        "case_file": str(args.cases.resolve()),
        "environment": {
            "python": platform.python_version(),
            "synkit": _synkit_version(),
            "rdkit": _package_version("rdkit"),
            "networkx": _package_version("networkx"),
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
