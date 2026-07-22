#!/usr/bin/env python3
"""Compare minimal partial-AAM expansion on general or radical corpora."""

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

METHODS = ("synkit", "gm", "rb1", "rb2")
METHOD_LABELS = {
    "synkit": "SynKit minimal ITS expansion",
    "gm": "GM",
    "rb1": "RB1 (PartialAAMs extend)",
    "rb2": "RB2 (PartialAAMs extend_g)",
}
DEFAULT_DEPENDENCIES = Path("/tmp/synkit-partialaam-deps")
DEFAULT_GMAPACHE = Path("/tmp/synkit-gmapache")
DEFAULT_PARTIALAAMS = Path("/tmp/PartialAAMs")


class CaseTimeout(TimeoutError):
    """Raised when one expansion exceeds the declared wall-time ceiling."""


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
    raise CaseTimeout("Partial-AAM expansion exceeded the per-case time ceiling")


def _side_maps(side: str) -> tuple[dict[int, tuple[int, int]], bool, bool]:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(side, parser)
    if molecule is None:
        raise ValueError(f"RDKit rejected endpoint {side!r}")
    atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0]
    maps = [atom.GetAtomMapNum() for atom in atoms]
    positive = [atom_map for atom_map in maps if atom_map > 0]
    identities = {
        atom.GetAtomMapNum(): (atom.GetAtomicNum(), atom.GetIsotope())
        for atom in atoms
        if atom.GetAtomMapNum() > 0
    }
    return identities, len(positive) == len(set(positive)), all(maps)


def _canonical_radical_endpoint(side: str) -> str:
    """Return a map-independent endpoint key retaining radical state."""
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(side, parser)
    if molecule is None:
        raise ValueError(f"RDKit rejected endpoint {side!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def completion_gates(
    source: str,
    candidate: str,
    *,
    require_radical_state: bool = False,
) -> dict[str, bool]:
    """Evaluate map completeness and endpoint preservation."""
    if source.count(">>") != 1 or candidate.count(">>") != 1:
        raise ValueError("Expected one reaction separator")
    candidate_results = [_side_maps(side) for side in candidate.split(">>")]
    candidate_maps = [item[0] for item in candidate_results]
    complete = all(item[2] for item in candidate_results)
    unique = complete and all(item[1] for item in candidate_results)
    balanced = unique and set(candidate_maps[0]) == set(candidate_maps[1])
    identity = balanced and all(
        candidate_maps[0][atom_map] == candidate_maps[1][atom_map]
        for atom_map in candidate_maps[0]
    )
    gates = {
        "all_atoms_mapped": complete,
        "unique_atom_maps": unique,
        "balanced_atom_maps": balanced,
        "atom_identity_preserved": identity,
        "endpoint_constitution_preserved": ITSExpand.endpoint_constitutions_match(
            source, candidate
        ),
    }
    if require_radical_state:
        gates["radical_state_preserved"] = all(
            _canonical_radical_endpoint(source_side)
            == _canonical_radical_endpoint(candidate_side)
            for source_side, candidate_side in zip(
                source.split(">>"), candidate.split(">>")
            )
        )
    return gates


def general_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "record_id": int(row["R-id"]),
            "source": str(row["partial"]),
            "reference": str(row["smart"]),
        }
        for row in read_json(path)
    ]


def radical_rows(path: Path) -> list[dict[str, Any]]:
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


def _synkit_expand(source: str, *, preserve_radical_state: bool = False) -> str:
    return ITSExpand.expand_rsmi(
        source,
        preserve_radical_state=preserve_radical_state,
    )


def load_methods(
    suite: str,
    selected: list[str],
    dependencies: Path,
    gmapache: Path,
    partialaams: Path,
) -> dict[str, Callable[[str], str]]:
    if suite == "radical":
        synkit_expand = lambda source: _synkit_expand(  # noqa: E731
            source,
            preserve_radical_state=True,
        )
    else:
        synkit_expand = _synkit_expand
    methods: dict[str, Callable[[str], str]] = {"synkit": synkit_expand}
    if set(selected) == {"synkit"}:
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


def _reference_checks(
    validator: AAMValidator,
    candidate: str,
    reference: str,
) -> dict[str, bool]:
    return {
        key: bool(value)
        for key, value in validator.smiles_checks(
            candidate,
            reference,
            constitutional_only=True,
        ).items()
        if key in {"ITS", "RC"}
    }


def benchmark_method(
    suite: str,
    name: str,
    function: Callable[[str], str],
    rows: list[dict[str, Any]],
    case_handle,
    progress_every: int,
    case_timeout: float,
) -> dict[str, Any]:
    validator = AAMValidator(strip_unbalanced_maps=True)
    counts: Counter[str] = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    wall_started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        case: dict[str, Any] = {"method": name, "record_id": row["record_id"]}
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, case_timeout)
        generation_started = time.perf_counter()
        try:
            candidate = function(str(row["source"]))
            generation_seconds = time.perf_counter() - generation_started
            timings["generation"].append(generation_seconds)
            if not isinstance(candidate, str) or candidate.count(">>") != 1:
                raise ValueError("Method did not return a reaction SMILES")
            counts["output"] += 1
            validation_started = time.perf_counter()
            gates = completion_gates(
                str(row["source"]),
                candidate,
                require_radical_state=suite == "radical",
            )
            checks = (
                _reference_checks(validator, candidate, str(row["reference"]))
                if suite == "general"
                else {}
            )
            for key, passed in {**gates, **checks}.items():
                prefix = "aam_validator_" if key in {"ITS", "RC"} else ""
                counts[f"{prefix}{key.lower()}:{str(passed).lower()}"] += 1
            accepted = all(gates.values()) and all(checks.values())
            counts[f"accepted:{str(accepted).lower()}"] += 1
            validation_seconds = time.perf_counter() - validation_started
            timings["validation"].append(validation_seconds)
            case.update(
                status="PASS" if accepted else "FAIL",
                generation_seconds=generation_seconds,
                validation_seconds=validation_seconds,
                gates=gates,
                reference_checks=checks,
            )
            if not accepted:
                case["candidate"] = candidate
        except Exception as exc:
            generation_seconds = time.perf_counter() - generation_started
            if (
                not timings["generation"]
                or generation_seconds != timings["generation"][-1]
            ):
                timings["generation"].append(generation_seconds)
            counts["error"] += 1
            counts["timeout"] += isinstance(exc, CaseTimeout)
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
            print(f"expansion {name}: {index}/{len(rows)}", file=sys.stderr, flush=True)
    wall = time.perf_counter() - wall_started
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
    parser.add_argument("--suite", choices=("general", "radical"), default="general")
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
        radical_rows(dataset) if args.suite == "radical" else general_rows(dataset)
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
                args.suite,
                name,
                methods[name],
                rows,
                case_handle,
                args.progress_every,
                args.case_timeout,
            )
            for name in args.methods
        ]
    report = {
        "schema": f"synkit.partial-aam-{args.suite}-comparison/1",
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
                "checks": ["completion gates", "radical-state preservation"],
                "warning": "Radical results are structural coverage, not AAM accuracy.",
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
