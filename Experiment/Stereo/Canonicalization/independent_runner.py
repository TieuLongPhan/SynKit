"""Compare the independent reference partitions with full production certificates."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from math import prod
from pathlib import Path
import platform
import subprocess
import time

import networkx as nx

from Experiment.Stereo.Canonicalization.independent_fixtures import Fixture, fixtures
from Experiment.Stereo.Canonicalization.exact_code_store import ExactCodeStore
from Experiment.Stereo.Canonicalization.runtime import peak_rss_mib, wall_time_limit
from Experiment.Stereo.Canonicalization.independent_model import (
    formal_states, from_descriptor, to_descriptor, transport,
)
from Experiment.Stereo.Canonicalization.independent_mutations import run_mutations
from Experiment.Stereo.Canonicalization.independent_orbits import (
    ATOM_KEYS, BOND_KEYS, ROOT, partition_agreement, partition_from_labels, reference_partition,
)


def _code(graph, descriptors, options, timeout_seconds):
    """Production result: a full exact code, not its finite digest."""
    from synkit.Graph.Stereo import canonicalize_configured_stereograph

    with wall_time_limit(timeout_seconds, TimeoutError("exact_computation_timeout")):
        result = canonicalize_configured_stereograph(
            graph, descriptors, enumerate_automorphism_group=False, **options,
        )
    if not result.complete or not result.exact:
        raise RuntimeError("Production did not return a completed exact result.")
    return result.canonical_code


def _production_assignments(fixture, choices, reference, options, production_timeout):
    """Bound certificate storage without replacing exact equality by a hash."""
    from synkit.Graph.Stereo import mirror_configured_descriptor

    mapping = {node: 10000 + index for index, node in enumerate(reversed(tuple(fixture.graph)))}
    renamed = nx.relabel_nodes(fixture.graph, mapping)
    observed, mirrors, relabeling_checks, issues = [], [], [], []
    with ExactCodeStore() as store:
        for assignment in reference["assignments"]:
            states = tuple(local[index] for local, index in zip(choices, assignment, strict=True))
            descriptors = tuple(to_descriptor(state) for state in states)
            try:
                identifier = store.intern(_code(fixture.graph, descriptors, options, production_timeout))
                mirrored = tuple(mirror_configured_descriptor(value) for value in descriptors)
                mirror_id = store.intern(_code(fixture.graph, mirrored, options, production_timeout))
                moved = tuple(to_descriptor(transport(state, mapping)) for state in states)
                relabeled_id = store.intern(_code(renamed, moved, options, production_timeout))
                if peak_rss_mib() > 1024:
                    raise MemoryError("production_process_peak_rss_limit:1024MiB")
                observed.append(identifier)
                mirrors.append(mirror_id)
                relabeling_checks.append(identifier == relabeled_id)
            except Exception as error:
                issues.append({"assignment": assignment, "reason": f"{type(error).__name__}: {error}"})
                break
        indices = [store.index_digest(identifier) for identifier in observed]
        bytes_on_disk = store.bytes_on_disk
    return observed, mirrors, relabeling_checks, issues, indices, bytes_on_disk


def audit_fixture(item):
    fixture, internal_colors, oracle_timeout, production_timeout = item
    started = time.monotonic()
    options = {"atom_color": "color", "bond_color": "color"} if internal_colors else {}
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    reference = reference_partition(
        fixture.graph, choices, timeout_seconds=oracle_timeout,
        atom_color="color" if internal_colors else ATOM_KEYS,
        bond_color="color" if internal_colors else BOND_KEYS,
    )
    record = {"case_id": fixture.name, "purpose": fixture.purpose,
              "families": [seed.family for seed in fixture.seeds], "reference": reference,
              "atom_count": len(fixture.graph), "carrier_count": len(choices),
              "production_timeout_per_certificate_seconds": production_timeout,
              "passed": False, "issues": []}
    if reference["status"] != "complete":
        return {**record, "status": "oracle_incomplete", "seconds": time.monotonic() - started}
    observed, mirrors, relabeling_checks, issues, indices, disk_size = _production_assignments(
        fixture, choices, reference, options, production_timeout,
    )
    record["issues"] = issues
    complete = len(observed) == len(reference["assignments"]) and not record["issues"]
    agreement = complete and partition_agreement(reference, reference["assignments"], observed)
    mirror_agreement = False
    if complete:
        codes = dict(zip(reference["assignments"], observed, strict=True))
        expected = [codes[assignment] for assignment in reference["mirror_assignment_images"]]
        mirror_agreement = mirrors == expected
    record.update({
        "status": "passed" if agreement and mirror_agreement and all(relabeling_checks) else "failed",
        "passed": agreement and mirror_agreement and all(relabeling_checks),
        "production_assignments_completed": len(observed), "full_partition_agreement": agreement,
        "production_mirror_agreement": mirror_agreement,
        "renumbering_checks": len(relabeling_checks), "renumbering_failures": relabeling_checks.count(False),
        "production_partition": partition_from_labels(reference["assignments"], observed) if complete else None,
        "certificate_sha256_indices": indices,
        "temporary_full_certificate_store_bytes": disk_size,
        "process_peak_rss_kib": peak_rss_mib() * 1024,
        "seconds": time.monotonic() - started,
    })
    print(json.dumps({"case": fixture.name, "status": record["status"],
                      "assignments": len(observed), "seconds": round(record["seconds"], 3)}), flush=True)
    return record


def public_fixtures(cip_path):
    """Freeze existing public-case inventory; do not claim independent perception."""
    from rdkit import Chem
    from Experiment.Stereo.Canonicalization.configuration_free_global import (
        ACS_DATASET, ROTA, _CASES, _carrier_inventory, _selected_rows,
    )

    rows, integrity = _selected_rows(_CASES, acs_path=ACS_DATASET, cip_path=cip_path, rota_path=ROTA)
    selected = []
    for spec in _CASES:
        molecule = Chem.MolFromSmiles(rows[(spec.source, spec.record_id)])
        if molecule is None:
            raise ValueError(f"Public fixture parse failure: {spec.source}:{spec.record_id}")
        graph, carriers, _ = _carrier_inventory(molecule)
        seeds = tuple(from_descriptor(carrier["class_representatives"][0]) for carrier in carriers)
        actual_composition = tuple(sorted(Counter(seed.family for seed in seeds).items()))
        formal_count = prod(len(formal_states(seed)) for seed in seeds)
        raw_count = prod(carrier["raw_local_permutations"] for carrier in carriers)
        raw_exhaustive = all(carrier["raw_inventory_exhaustive"] for carrier in carriers)
        if (actual_composition != spec.composition or formal_count != spec.formal_assignments
                or raw_count != spec.raw_product or not raw_exhaustive):
            raise ValueError(f"Public frozen inventory/enumeration contract changed: {spec.source}:{spec.record_id}")
        selected.append(Fixture(f"public:{spec.budget}:{spec.source}:{spec.record_id}", graph, seeds,
                                "Independent assignments conditional on the supplied/perceived frozen inventory"))
    return tuple(selected), integrity


def provenance():
    import rdkit

    paths = [
        "Experiment/Stereo/Canonicalization/" + name for name in (
            "independent_model.py", "independent_orbits.py", "independent_fixtures.py",
            "independent_runner.py", "independent_mutations.py", "theory_audit.py",
            "configuration_free_global.py", "configuration_free_cases.py", "exact_code_store.py",
            "independent_graph.py",
            "independent_transport.py",
        )
    ] + [str(path.relative_to(ROOT)) for path in sorted((ROOT / "synkit/Graph/Stereo").glob("*.py"))]
    paths += ["synkit/Graph/Canon/exact.py", "synkit/Chem/Molecule/stereo_perception.py"]
    return {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "python": platform.python_version(), "rdkit": rdkit.__version__,
        "networkx": nx.__version__, "platform": platform.platform(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--include-public-abc", action="store_true")
    parser.add_argument("--cip-path", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--oracle-timeout", type=float, default=10)
    parser.add_argument("--production-timeout", type=float, default=5)
    args = parser.parse_args()
    if args.jobs < 1 or args.oracle_timeout <= 0 or args.production_timeout <= 0:
        parser.error("Jobs and computation time limits must be positive.")
    start_provenance = provenance()
    items = [(fixture, True, args.oracle_timeout, args.production_timeout) for fixture in fixtures()]
    integrity = {}
    if args.include_public_abc:
        public, integrity = public_fixtures(args.cip_path)
        items += [(fixture, False, args.oracle_timeout, args.production_timeout) for fixture in public]
    if args.case:
        names = {fixture.name for fixture, *_ in items}
        if set(args.case) - names:
            parser.error(f"Unknown case identifiers: {sorted(set(args.case) - names)}")
        items = [item for item in items if item[0].name in args.case]
    if len({item[0].name for item in items}) != len(items):
        raise ValueError("Each audit case must have one unique identifier.")
    if args.jobs == 1:
        records = [audit_fixture(item) for item in items]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            records = list(executor.map(audit_fixture, items))
    mutations = run_mutations()
    from Experiment.Stereo.Canonicalization.theory_audit import action_audit

    production_actions = action_audit()
    source_unchanged = provenance()["source_sha256"] == start_provenance["source_sha256"]
    report = {
        "schema": "synkit.independent-stereograph-audit/1", "provenance": start_provenance,
        "source_integrity": integrity,
        "protocol": {
            "truth": ("Explicit geometries plus exhaustive attributed constitutional automorphisms "
                      "preserving a frozen inventory"),
            "comparison": ("Complete partitions from full production certificate strings; "
                           "SHA256 values are indices only"),
            "carrier_perception_independently_validated": False,
            "constructed_fixtures_are_chemical_holdout": False,
            "shared_contract": "Raw descriptor fields, declared attribute selectors and production input constructors",
            "local_group_builders": "Independent A4/D4, integer-coordinate TBP/OH and terminal sign relations",
            "constitutional_map_enumerator": (
                "NetworkX VF2++ with exact equitable typed-color refinement; original edge colors checked on every map"
            ),
            "production_group_output": "Generators rather than complete automorphism materialization",
            "certificate_storage": "Temporary SQLite full-byte equality; digest is only a candidate index",
            "production_memory_guard": "1GiB peak RSS checked between exact computations; native calls may overshoot",
            "jobs": args.jobs, "selected_case_ids": [item[0].name for item in items],
            "oracle_incomplete_is_pass": False,
        },
        "records": records, "mutation_controls": mutations,
        "production_slot_actions": production_actions, "sources_unchanged_during_run": source_unchanged,
        "summary": {"cases": len(records), "passed": sum(record["passed"] for record in records),
                    "oracle_incomplete": sum(record["status"] == "oracle_incomplete" for record in records),
                    "mutations_rejected": sum(record["rejected"] for record in mutations["records"])},
        "passed": (bool(records) and all(record["passed"] for record in records) and mutations["passed"]
                   and source_unchanged and production_actions["action_failures"] == 0
                   and production_actions["cocycle_failures"] == 0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1
