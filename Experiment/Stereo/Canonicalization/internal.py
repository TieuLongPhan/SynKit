#!/usr/bin/env python3
"""Audit all configured stereo families and public-input atom permutations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import networkx as nx
from rdkit import Chem
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.acs_molecular_chirality import (  # noqa: E402
    EXPECTED_SHA256,
    load_dataset,
)
from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    OctahedralStereo,
    PlanarBondStereo,
    PlanarChiralityStereo,
    SquarePlanarStereo,
    StereoSpecification,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    canonicalize_configured_stereograph,
    canonicalize_rdkit_configured_stereograph,
    local_configuration_classes,
    mirror_configured_descriptor,
)
from synkit.Graph.Stereo.orbits import (  # noqa: E402
    SHAPE_DEFINITIONS,
    StereoConfiguration,
)

_DEFAULT_CASE_IDS = ("VS021", "VS060", "VS061", "VS066")
_EXPECTED_CONFIGURATION_COUNTS = {
    "tetrahedral": 2,
    "square_planar": 3,
    "trigonal_bipyramidal": 20,
    "octahedral": 30,
    "planar_bond": 2,
    "atrop_bond": 2,
    "cumulene_axis": 2,
    "extended_cis_trans": 2,
    "helical": 2,
    "planar_chirality": 2,
}
_EXPECTED_MIRROR_FIXED_COUNTS = {
    family: (
        _EXPECTED_CONFIGURATION_COUNTS[family]
        if family
        in {
            "square_planar",
            "planar_bond",
            "extended_cis_trans",
        }
        else 0
    )
    for family in _EXPECTED_CONFIGURATION_COUNTS
}


@dataclass(frozen=True)
class _Fixture:
    family: str
    graph: nx.Graph
    seed: Any


class _CaseTimeout(TimeoutError):
    pass


class _case_time_limit:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        if not hasattr(signal, "setitimer"):
            return
        self.previous_handler = signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_error: Any) -> None:
        if not hasattr(signal, "setitimer"):
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    @staticmethod
    def _expired(_signum: int, _frame: Any) -> None:
        raise _CaseTimeout


def _star(arity: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    for reference in range(1, arity + 1):
        graph.add_node(reference, color=f"ligand:{reference}")
        graph.add_edge(0, reference, color="single")
    return graph


def _bond_graph(*, central_order: str) -> nx.Graph:
    graph = nx.Graph()
    for node in range(6):
        graph.add_node(node, color=f"atom:{node}")
    for edge in ((0, 2), (1, 2), (3, 4), (3, 5)):
        graph.add_edge(*edge, color="single")
    graph.add_edge(2, 3, color=central_order)
    return graph


def _cumulene_graph() -> nx.Graph:
    graph = nx.Graph()
    for node in range(7):
        graph.add_node(node, color=f"atom:{node}")
    for edge in ((0, 2), (1, 2), (4, 5), (4, 6)):
        graph.add_edge(*edge, color="single")
    graph.add_edge(2, 3, color="double")
    graph.add_edge(3, 4, color="double")
    return graph


def _extended_cumulene_graph() -> nx.Graph:
    graph = nx.Graph()
    for node in range(8):
        graph.add_node(node, color=f"atom:{node}")
    for edge in ((0, 2), (1, 2), (5, 6), (5, 7)):
        graph.add_edge(*edge, color="single")
    for edge in ((2, 3), (3, 4), (4, 5)):
        graph.add_edge(*edge, color="double")
    return graph


def _path_graph() -> nx.Graph:
    graph = nx.path_graph(5)
    nx.set_node_attributes(
        graph,
        {node: f"path:{node}" for node in graph},
        "color",
    )
    nx.set_edge_attributes(graph, "single", "color")
    return graph


def _plane_graph() -> nx.Graph:
    graph = nx.cycle_graph(4)
    graph.add_node(4, color="pilot")
    nx.set_node_attributes(
        graph,
        {
            0: "plane:A",
            1: "plane:B",
            2: "plane:C",
            3: "plane:D",
            4: "pilot",
        },
        "color",
    )
    nx.set_edge_attributes(graph, "plane", "color")
    graph.add_edge(0, 4, color="pilot_link")
    return graph


def _fixture_catalog() -> tuple[_Fixture, ...]:
    return (
        _Fixture(
            "tetrahedral",
            _star(4),
            TetrahedralStereo((0, 1, 2, 3, 4), 1),
        ),
        _Fixture(
            "square_planar",
            _star(4),
            SquarePlanarStereo((0, 1, 2, 3, 4), 0),
        ),
        _Fixture(
            "trigonal_bipyramidal",
            _star(5),
            TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1),
        ),
        _Fixture(
            "octahedral",
            _star(6),
            OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1),
        ),
        _Fixture(
            "planar_bond",
            _bond_graph(central_order="double"),
            PlanarBondStereo((0, 1, 2, 3, 4, 5), 0),
        ),
        _Fixture(
            "atrop_bond",
            _bond_graph(central_order="single"),
            AtropBondStereo((0, 1, 2, 3, 4, 5), 1),
        ),
        _Fixture(
            "cumulene_axis",
            _cumulene_graph(),
            CumuleneAxisStereo((2, 3, 4), ((0, 1), (5, 6)), 1),
        ),
        _Fixture(
            "extended_cis_trans",
            _extended_cumulene_graph(),
            ExtendedCisTransStereo(
                (2, 3, 4, 5),
                ((0, 1), (6, 7)),
                0,
            ),
        ),
        _Fixture(
            "helical",
            _path_graph(),
            HelicalStereo((0, 1, 2, 3, 4), 1),
        ),
        _Fixture(
            "planar_chirality",
            _plane_graph(),
            PlanarChiralityStereo((0, 1, 2, 3), 4, 1),
        ),
    )


def _clear_nonsemantic_caches(molecule: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        for name in ("_CIPCode", "_CIPRank"):
            if atom.HasProp(name):
                atom.ClearProp(name)
    return copy


def _exact(
    graph: nx.Graph,
    descriptor: Any,
    *,
    timeout_seconds: float,
) -> tuple[Any | None, str | None, float]:
    started = time.perf_counter()
    try:
        with _case_time_limit(timeout_seconds):
            result = canonicalize_configured_stereograph(
                graph,
                (descriptor,),
                atom_color="color",
                bond_color="color",
            )
    except _CaseTimeout:
        return None, "case_timeout", time.perf_counter() - started
    except Exception as error:
        return (
            None,
            f"{type(error).__name__}:{error}",
            time.perf_counter() - started,
        )
    return result, None, time.perf_counter() - started


def _local_arrangement_audit(fixture: _Fixture) -> dict[str, Any]:
    if fixture.family in SHAPE_DEFINITIONS:
        definition = SHAPE_DEFINITIONS[fixture.family]
        frames = tuple(
            permutation.apply(fixture.seed.configuration.frame)
            for permutation in definition.unspecified_group.elements
        )
        forms = {
            StereoConfiguration(
                fixture.family,
                frame,
                StereoSpecification.FIXED,
            ).canonical_form()
            for frame in frames
        }
        return {
            "raw_arrangements_checked": len(frames),
            "local_configuration_classes": len(forms),
            "method": "complete_unspecified_group_frame_enumeration",
        }
    if fixture.family == "helical":
        raw = tuple(
            descriptor
            for parity in (1, -1)
            for descriptor in (
                HelicalStereo(fixture.seed.path, parity),
                HelicalStereo(tuple(reversed(fixture.seed.path)), parity),
            )
        )
    else:
        plane = fixture.seed.plane_atoms
        raw_values = []
        for parity in (1, -1):
            reverse = tuple(reversed(plane))
            for offset in range(len(plane)):
                raw_values.append(
                    PlanarChiralityStereo(
                        plane[offset:] + plane[:offset],
                        fixture.seed.pilot,
                        parity,
                    )
                )
                raw_values.append(
                    PlanarChiralityStereo(
                        reverse[offset:] + reverse[:offset],
                        fixture.seed.pilot,
                        -parity,
                    )
                )
        raw = tuple(raw_values)
    return {
        "raw_arrangements_checked": len(raw),
        "local_configuration_classes": len(
            {descriptor.canonical_form() for descriptor in raw}
        ),
        "method": "complete_path_or_plane_equivalent_representation_enumeration",
    }


def _mapping_for_order(order: tuple[int, ...]) -> dict[int, int]:
    return {old: new for new, old in enumerate(order)}


def _deterministic_orders(
    size: int,
    *,
    count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        return ()
    base = tuple(range(size))
    candidates = [
        tuple(reversed(base)),
        base[1:] + base[:1],
        base[2:] + base[:2],
    ]
    generator = random.Random(seed)
    while len(candidates) < max(count * 3, 12):
        values = list(base)
        generator.shuffle(values)
        candidates.append(tuple(values))
    unique = []
    for order in candidates:
        if order != base and order not in unique:
            unique.append(order)
        if len(unique) == count:
            break
    return tuple(unique)


def _family_audit(
    fixture: _Fixture,
    *,
    class_relabelings: int,
    representative_samples: int,
    exhaustive_atom_limit: int,
    exhaustive_all_classes: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    local = _local_arrangement_audit(fixture)
    expected = _EXPECTED_CONFIGURATION_COUNTS[fixture.family]
    classes = local_configuration_classes(fixture.seed)
    class_records = []
    baseline_by_code: dict[str, int] = {}
    mirror_codes: list[str] = []
    failures = []
    exact_checks = 0
    timeouts = 0
    started_family = time.perf_counter()
    for index, descriptor in enumerate(classes):
        baseline, issue, duration = _exact(
            fixture.graph,
            descriptor,
            timeout_seconds=timeout_seconds,
        )
        exact_checks += 1
        if issue is not None or baseline is None:
            timeouts += issue == "case_timeout"
            failures.append(f"class:{index}:baseline:{issue}")
            class_records.append(
                {
                    "class_index": index,
                    "baseline_issue": issue,
                    "seconds": duration,
                }
            )
            continue
        baseline_by_code.setdefault(baseline.canonical_code, index)
        relabel_failures = []
        relabel_seconds = 0.0
        if exhaustive_all_classes:
            orders: Iterable[tuple[int, ...]] = permutations(
                range(fixture.graph.number_of_nodes())
            )
            relabeling_protocol = "exhaustive_n_factorial"
        else:
            orders = _deterministic_orders(
                fixture.graph.number_of_nodes(),
                count=class_relabelings,
                seed=1729 + index,
            )
            relabeling_protocol = "fixed_seed_deterministic_sample"
        relabelings_checked = 0
        for order in orders:
            relabelings_checked += 1
            mapping = _mapping_for_order(order)
            relabeled_graph = nx.relabel_nodes(
                fixture.graph,
                mapping,
                copy=True,
            )
            observed, relabel_issue, elapsed = _exact(
                relabeled_graph,
                descriptor.relabel(mapping),
                timeout_seconds=timeout_seconds,
            )
            exact_checks += 1
            relabel_seconds += elapsed
            if relabel_issue == "case_timeout":
                timeouts += 1
            if (
                relabel_issue is not None
                or observed is None
                or not baseline.same_stereograph(observed)
            ):
                relabel_failures.append(
                    {
                        "order": list(order),
                        "issue": relabel_issue or "certificate_mismatch",
                    }
                )
        mirrored = mirror_configured_descriptor(descriptor)
        mirror, mirror_issue, mirror_seconds = _exact(
            fixture.graph,
            mirrored,
            timeout_seconds=timeout_seconds,
        )
        exact_checks += 1
        if mirror_issue == "case_timeout":
            timeouts += 1
        if mirror_issue is not None or mirror is None:
            failures.append(f"class:{index}:mirror:{mirror_issue}")
        else:
            mirror_codes.append(mirror.canonical_code)
        double = mirror_configured_descriptor(mirrored)
        double_mirror_invariant = descriptor == double
        if relabel_failures:
            failures.append(f"class:{index}:relabel")
        if not double_mirror_invariant:
            failures.append(f"class:{index}:double_mirror")
        class_records.append(
            {
                "class_index": index,
                "canonical_digest": baseline.canonical_digest,
                "relabeling_protocol": relabeling_protocol,
                "relabelings_checked": relabelings_checked,
                "relabeling_failures": relabel_failures,
                "mirror_digest": (
                    mirror.canonical_digest if mirror is not None else None
                ),
                "mirror_issue": mirror_issue,
                "mirror_fixed": (
                    baseline.same_stereograph(mirror) if mirror is not None else None
                ),
                "double_mirror_invariant": double_mirror_invariant,
                "seconds": duration + relabel_seconds + mirror_seconds,
            }
        )

    size = fixture.graph.number_of_nodes()
    if exhaustive_all_classes:
        representative_orders = ()
        representative_protocol = "covered_by_exhaustive_class_relabeling"
    elif size <= exhaustive_atom_limit:
        representative_orders: Iterable[tuple[int, ...]] = permutations(range(size))
        representative_protocol = "exhaustive_n_factorial"
    else:
        representative_orders = (tuple(range(size)),) + _deterministic_orders(
            size,
            count=max(0, representative_samples - 1),
            seed=8675309,
        )
        representative_protocol = "fixed_seed_deterministic_sample"
    representative = classes[0]
    representative_baseline, baseline_issue, _elapsed = _exact(
        fixture.graph,
        representative,
        timeout_seconds=timeout_seconds,
    )
    representative_failures = []
    representative_checked = 0
    for order in representative_orders:
        mapping = _mapping_for_order(tuple(order))
        observed, issue, _duration = _exact(
            nx.relabel_nodes(fixture.graph, mapping, copy=True),
            representative.relabel(mapping),
            timeout_seconds=timeout_seconds,
        )
        exact_checks += 1
        representative_checked += 1
        if issue == "case_timeout":
            timeouts += 1
        if (
            baseline_issue is not None
            or representative_baseline is None
            or issue is not None
            or observed is None
            or not representative_baseline.same_stereograph(observed)
        ):
            representative_failures.append(
                {
                    "order": list(order),
                    "issue": issue or baseline_issue or "certificate_mismatch",
                }
            )
    failures.extend(
        f"representative:{item['issue']}" for item in representative_failures
    )
    exact_classes = len(baseline_by_code)
    mirror_closed = set(mirror_codes) <= set(baseline_by_code)
    mirror_fixed = sum(record.get("mirror_fixed") is True for record in class_records)
    if exact_classes != expected:
        failures.append(f"exact_class_count:{exact_classes}:expected:{expected}")
    if local["local_configuration_classes"] != expected:
        failures.append(
            "local_class_count:"
            f"{local['local_configuration_classes']}:expected:{expected}"
        )
    if len(classes) != expected:
        failures.append(f"enumerated_class_count:{len(classes)}:expected:{expected}")
    if not mirror_closed:
        failures.append("mirror_not_closed_over_classes")
    expected_mirror_fixed = _EXPECTED_MIRROR_FIXED_COUNTS[fixture.family]
    if mirror_fixed != expected_mirror_fixed:
        failures.append(
            f"mirror_fixed_count:{mirror_fixed}:expected:{expected_mirror_fixed}"
        )
    return {
        "family": fixture.family,
        "atoms": size,
        "expected_configuration_classes": expected,
        **local,
        "enumerated_descriptor_classes": len(classes),
        "exact_certificate_classes": exact_classes,
        "certificate_collisions": len(classes) - exact_classes,
        "mirror_closed_over_classes": mirror_closed,
        "expected_mirror_fixed_classes": expected_mirror_fixed,
        "mirror_fixed_classes": mirror_fixed,
        "class_relabelings_checked": sum(
            record.get("relabelings_checked", 0) for record in class_records
        ),
        "representative_relabeling_protocol": representative_protocol,
        "representative_relabelings_checked": representative_checked,
        "representative_relabeling_failures": representative_failures,
        "exact_canonicalizations": exact_checks,
        "timeouts": timeouts,
        "failures": failures,
        "passed": not failures,
        "classes": class_records,
        "seconds": time.perf_counter() - started_family,
    }


def _public_audit(
    path: Path,
    *,
    case_ids: tuple[str, ...],
) -> dict[str, Any]:
    rows = {row["ID"]: row for row in load_dataset(path)}
    missing = sorted(set(case_ids) - rows.keys())
    if missing:
        raise ValueError(f"Unknown ACS case identifiers: {missing}")
    records = []
    started_all = time.perf_counter()
    for identifier in case_ids:
        molecule = Chem.MolFromSmiles(rows[identifier]["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected ACS case {identifier}")
        baseline = canonicalize_rdkit_configured_stereograph(molecule)
        cleared = canonicalize_rdkit_configured_stereograph(
            _clear_nonsemantic_caches(molecule)
        )
        failures = []
        permutation_count = 0
        started = time.perf_counter()
        for order in permutations(range(molecule.GetNumAtoms())):
            renumbered = Chem.RenumberAtoms(molecule, order)
            result = canonicalize_rdkit_configured_stereograph(renumbered)
            permutation_count += 1
            if not baseline.same_stereograph(result):
                failures.append(list(order))
        records.append(
            {
                "id": identifier,
                "atoms": molecule.GetNumAtoms(),
                "configured_certificate_digest": baseline.canonical_digest,
                "permutations_checked": permutation_count,
                "permutation_failures": failures,
                "all_permutations_invariant": not failures,
                "cip_cache_invariant": baseline.same_stereograph(cleared),
                "seconds": time.perf_counter() - started,
            }
        )
    return {
        "dataset": {
            "name": "ACS StereoMolGraph validation corpus",
            "records": len(rows),
            "audited_sha256": EXPECTED_SHA256,
            "selected_ids": list(case_ids),
            "selection_rule": (
                "small configured public cases spanning tetrahedral, E/Z, "
                "and two-center stereo, permitting exhaustive n! relabeling"
            ),
        },
        "atom_permutations_checked": sum(
            record["permutations_checked"] for record in records
        ),
        "permutation_failures": sum(
            len(record["permutation_failures"]) for record in records
        ),
        "cases_fully_invariant": sum(
            record["all_permutations_invariant"] for record in records
        ),
        "cip_cache_invariant_cases": sum(
            record["cip_cache_invariant"] for record in records
        ),
        "records": records,
        "seconds": time.perf_counter() - started_all,
    }


def benchmark_exact_canonicalization(
    path: Path,
    *,
    case_ids: tuple[str, ...] = _DEFAULT_CASE_IDS,
    family_names: tuple[str, ...] = (),
    class_relabelings: int = 3,
    representative_samples: int = 16,
    exhaustive_atom_limit: int = 5,
    exhaustive_all_classes: bool = False,
    timeout_seconds: float = 20.0,
    include_public: bool = True,
) -> dict[str, Any]:
    if class_relabelings < 0 or representative_samples < 1:
        raise ValueError("Relabeling sample counts are invalid.")
    catalog = _fixture_catalog()
    known = {fixture.family for fixture in catalog}
    unknown = sorted(set(family_names) - known)
    if unknown:
        raise ValueError(f"Unknown configured families: {unknown}")
    selected = tuple(
        fixture
        for fixture in catalog
        if not family_names or fixture.family in family_names
    )
    started = time.perf_counter()
    families = [
        _family_audit(
            fixture,
            class_relabelings=class_relabelings,
            representative_samples=representative_samples,
            exhaustive_atom_limit=exhaustive_atom_limit,
            exhaustive_all_classes=exhaustive_all_classes,
            timeout_seconds=timeout_seconds,
        )
        for fixture in selected
    ]
    public = (
        _public_audit(path, case_ids=case_ids)
        if include_public
        else {
            "dataset": None,
            "atom_permutations_checked": 0,
            "permutation_failures": 0,
            "cases_fully_invariant": 0,
            "cip_cache_invariant_cases": 0,
            "records": [],
            "seconds": 0.0,
        }
    )
    expected_classes = sum(
        record["expected_configuration_classes"] for record in families
    )
    if exhaustive_all_classes:
        whole_graph_claim = (
            "Every atom permutation is checked for every local configuration "
            "class in every selected family."
        )
    else:
        whole_graph_claim = (
            "Whole-graph n! relabeling is exhaustive only for one "
            "representative of fixtures with at most the declared atom "
            "limit; larger representatives and the remaining classes use "
            "reported fixed-seed samples."
        )
    return {
        "schema": "synkit.exact-canonicalization-benchmark/2",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "task": (
            "all-family local configuration quotienting and exact "
            "configured-stereograph certificate invariance"
        ),
        "protocol": {
            "selected_families": [record["family"] for record in families],
            "class_relabelings_per_configuration": class_relabelings,
            "representative_samples_for_large_fixtures": (representative_samples),
            "exhaustive_atom_permutation_limit": exhaustive_atom_limit,
            "exhaustive_all_configuration_classes": (exhaustive_all_classes),
            "case_timeout_seconds": timeout_seconds,
            "public_acs_included": include_public,
        },
        "totals": {
            "families": len(families),
            "expected_configuration_classes": expected_classes,
            "local_configuration_classes": sum(
                record["local_configuration_classes"] for record in families
            ),
            "exact_certificate_classes": sum(
                record["exact_certificate_classes"] for record in families
            ),
            "raw_local_arrangements_checked": sum(
                record["raw_arrangements_checked"] for record in families
            ),
            "class_relabelings_checked": sum(
                record["class_relabelings_checked"] for record in families
            ),
            "representative_relabelings_checked": sum(
                record["representative_relabelings_checked"] for record in families
            ),
            "synthetic_exact_canonicalizations": sum(
                record["exact_canonicalizations"] for record in families
            ),
            "synthetic_timeouts": sum(record["timeouts"] for record in families),
            "family_failures": sum(not record["passed"] for record in families),
            "public_atom_permutations_checked": public["atom_permutations_checked"],
            "public_permutation_failures": public["permutation_failures"],
        },
        "families": families,
        "public_external_audit": public,
        "passed": (
            all(record["passed"] for record in families)
            and public["permutation_failures"] == 0
        ),
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "All raw local arrangements are exhaustively quotiented for the "
            "seven finite-orbit shapes, with complete equivalent path/plane "
            "representations for the two variable-support families. Every "
            "resulting local class is encoded by an exact whole-stereograph "
            "certificate and checked under deterministic relabeling and "
            f"mirror closure. {whole_graph_claim} Synthetic family coverage "
            "is not public chemical validation."
        ),
    }
