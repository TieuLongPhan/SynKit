#!/usr/bin/env python
"""Reproduce the reaction paper's focused software-conformance evidence.

Run from any directory with the repository's scientific Python environment:
    python Experiment/StereoReaction/paper_audit.py

This is an explicit selected-suite audit, not the full maintained suite or a
prediction-accuracy benchmark. Frozen corpus inventories and historical RX14
results remain distinct from measurements made by this invocation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Experiment/Lewis/mech_path/Data/MechanismBench"
TEST_SELECTION = (
    "Test/Graph/Stereo/test_reaction_relations.py",
    "Test/Graph/Stereo/test_reaction_transport_calculus.py",
    "Test/Graph/Stereo/test_reaction_composition.py",
    "Test/Graph/Stereo/test_reaction_wire.py",
    "Test/Graph/Stereo/test_reaction_interchange.py",
    "Test/Graph/Stereo/test_reaction_semantics.py",
    "Test/Graph/Stereo/test_migration_closeout.py",
    "Test/Graph/Stereo/test_reaction_witness_basis.py",
    "Test/Graph/ITS/test_stereo_layers.py",
    "Test/Graph/Fusion/test_proof_v2.py",
    "Test/Graph/Fusion/test_proof_document_digest.py",
    "Test/Graph/Fusion/test_interface_constructor.py",
    "Test/Graph/Fusion/test_identity_search.py",
    "Test/Synthesis/Reactor/test_stereo_conformance_data.py",
    "Test/Synthesis/Reactor/test_promoted_stereo_rewrites.py",
    "Test/Synthesis/Reactor/test_non_tetrahedral_rewrite_conformance.py",
    "Test/Synthesis/Reactor/test_stereo_coupling_corpus.py",
    "Test/Mechanism/test_electrocyclic_machinery.py",
    "Test/Mechanism/test_stereo_branch_coverage.py",
    "Test/Mechanism/test_stereo_workflow.py",
    "Test/Rule/test_generic_stereo_six_geometry.py",
    "Test/Benchmark/test_stereo_rxn_validation.py",
)
FOCUSED_TEST_NODES = (
    "Test/Synthesis/Reactor/test_syn_reactor_stereo_cases.py::test_generic_pi_rule_semihydrogenates_but_2_yne_with_syn_anti_coupling",
    "Test/Synthesis/Reactor/test_syn_reactor_stereo_cases.py::test_one_generic_syn_pi_rule_reduces_alkyne_and_alkene",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def portable(value):
    """Retain comparison payloads as explicit JSON values, including descriptors."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return portable(value.value)
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    if hasattr(value, "to_dict"):
        return portable(value.to_dict())
    raise TypeError("No comparison serialization for " + type(value).__name__)


def migration_report():
    """Instrument the actual closeout fixture without changing its decisions."""
    from synkit.Graph.Stereo.legacy import StereoSemanticComparison, summarize_stereo_comparisons

    module = module_from_file("reaction_paper_migration", ROOT / TEST_SELECTION[6])
    original = StereoSemanticComparison.create
    current = {}
    comparisons, records = [], []

    class TaggedDescriptors:
        def __iter__(self):
            for descriptor in descriptors:
                current["descriptor"] = descriptor
                yield descriptor

    def observe(*args, **kwargs):
        record = original(*args, **kwargs)
        comparisons.append(record)
        records.append({
            "comparison_index": len(records) + 1,
            "descriptor_family": current["descriptor"].descriptor_class,
            **portable(record.to_dict()),
        })
        return record

    descriptors = module.DESCRIPTORS
    with patch.object(module, "DESCRIPTORS", TaggedDescriptors()):
        with patch.object(StereoSemanticComparison, "create", side_effect=observe):
            module.test_complete_six_geometry_migration_audit_has_no_unregistered_divergence()
    return {
        "schema": "synkit.stereo-rxn-current-migration/1",
        "fixture": TEST_SELECTION[6],
        "instrumentation": "Observe comparison creation; preserve the fixture and all assertions unchanged.",
        "summary": summarize_stereo_comparisons(comparisons),
        "records": records,
        "boundary": "Sixty-two operation comparisons across six designed descriptor fixtures; not 62 reactions.",
    }


def witness_basis_report():
    """Enumerate preserving bases for one declared effect in each native family.

    The retired formula is evaluated explicitly as a mathematical diagnostic;
    it is not a second execution of old released code or a performance baseline.
    """
    from synkit.Graph.Stereo import StereoChange, StereoRelationKind
    from synkit.Graph.Stereo.orbits import StereoConfiguration

    module = module_from_file("reaction_paper_basis", ROOT / TEST_SELECTION[6])
    cases, records = [], []
    coordination = {"square_planar", "trigonal_bipyramidal", "octahedral"}
    for before in module.DESCRIPTORS:
        after = before.invert()
        if before.descriptor_class in coordination:
            for permutation in before.configuration.definition.unspecified_group.elements:
                candidate = replace(before, atoms=permutation.apply(before.configuration.frame))
                if before.relation_to(candidate).kind is StereoRelationKind.RECONFIGURED:
                    after = candidate
                    break
            else:
                raise AssertionError("No reconfiguration fixture for " + before.descriptor_class)
        effect = StereoChange.from_endpoints(before, after)
        target = after.configuration
        case_records = []
        for index, basis in enumerate(before.configuration.definition.preserving_group.elements, 1):
            substrate = replace(before, atoms=basis.apply(before.configuration.frame))
            assert substrate.configuration == before.configuration
            output = effect.apply_to(substrate)
            retired_frame = effect.relation.witness.apply(substrate.configuration.frame)
            retired_output = StereoConfiguration(before.descriptor_class, retired_frame)
            row = {
                "descriptor_family": before.descriptor_class,
                "basis_index": index,
                "basis_permutation": list(basis.image),
                "substrate_frame": list(substrate.configuration.frame),
                "current_output_frame": list(output.configuration.frame),
                "retired_formula_output_frame": list(retired_frame),
                "current_output_matches_declared_target": output.configuration == target,
                "retired_formula_matches_declared_target": retired_output == target,
            }
            records.append(row)
            case_records.append(row)
        cases.append({
            "descriptor_family": before.descriptor_class,
            "before_frame": list(before.configuration.frame),
            "after_frame": list(target.frame),
            "effect_relation": effect.relation.kind.value,
            "effect_witness": list(effect.relation.witness.permutation.image),
            "preserving_bases": len(case_records),
            "current_target_matches": sum(r["current_output_matches_declared_target"] for r in case_records),
            "retired_formula_target_matches": sum(r["retired_formula_matches_declared_target"] for r in case_records),
        })
    return {
        "schema": "synkit.stereo-rxn-witness-basis-audit/1",
        "selection": ("One first-admissible RECONFIGURED effect for each coordination family; "
                      "inverse controls otherwise."),
        "retired_formula": "Apply the template witness directly to every equivalent substrate frame without rebasing.",
        "boundary": ("Complete preserving-group bases for six designed effects; "
                     "no random sampling or chemical accuracy claim."),
        "cases": cases,
        "records": records,
        "all_current_outputs_match": all(r["current_output_matches_declared_target"] for r in records),
    }


def source_inventory():
    paths = {Path(__file__).resolve(), ROOT / "pyproject.toml"}
    paths.update(ROOT / name for name in TEST_SELECTION)
    paths.add(ROOT / "Test/Synthesis/Reactor/test_syn_reactor_stereo_cases.py")
    paths.update((ROOT / "Experiment/StereoReaction").glob("*.py"))
    for name in ("Graph/Stereo", "Graph/ITS", "Graph/Fusion", "Graph/Morphism", "Rule", "Mechanism",
                 "Synthesis/Reactor"):
        paths.update((ROOT / "synkit" / name).rglob("*.py"))
    return {relative(path): digest(path) for path in sorted(paths)}


def corpus_inventory():
    baseline = load(DATA / "stereo.json")
    positive = [c for c in baseline["cases"] if c["case_kind"] == "transformation"]
    negative = [c for c in baseline["cases"] if c["case_kind"] == "negative_assertion"]
    counts = Counter(c["representation"] for c in positive)
    deferred = [c["case_id"] for c in positive if c["status"] == "deferred_isotope_support"]
    coverage = load(DATA / "stereo_branch_coverage.json")
    extensions = {}
    for name in ("stereo_couplings.json", "electrocyclic_machinery.json"):
        payload = load(DATA / name)
        observed = Counter(c["kind"] for c in payload["cases"])
        assert observed["positive"] == payload["positive_case_count"]
        assert observed["negative"] == payload["negative_fixture_count"]
        extensions[name] = {"schema": payload["schema"], "counts": dict(observed),
                            "case_ids": [c["case_id"] for c in payload["cases"]]}
    assert len(positive) == baseline["positive_case_count"] == 80
    assert len(negative) == baseline["negative_fixture_count"] == 8
    assert len(deferred) == 3
    return {
        "baseline": {
            "schema": baseline["schema"], "positive_records": len(positive),
            "negative_specification_assertions": len(negative), "positive_by_representation": dict(counts),
            "direct_reactor_records": len(positive) - counts["mechanism_replay"] - len(deferred),
            "mechanism_replay_records": counts["mechanism_replay"],
            "deferred_isotope_record_ids": deferred,
            "cases": [{"case_id": c["case_id"], "kind": c["case_kind"],
                       "status": c["status"], "representation": c.get("representation")}
                      for c in baseline["cases"]],
            "boundary": "80 reviewed records include 3 deferred cases; 8 negatives are specification assertions.",
        },
        "extensions": extensions,
        "branch_index": {
            "schema": coverage["schema"], "branches_by_axis": {k: len(v) for k, v in coverage["coverage"].items()},
            "total_indexed_branches": sum(len(v) for v in coverage["coverage"].values()),
            "coverage": coverage["coverage"], "serialization_variants_excluded": coverage["excluded_from_case_count"],
            "boundary": ("Reference-index completeness, not independent chemical decisions "
                         "or joint-factorial execution."),
        },
    }


def input_inventory():
    names = ("stereo.json", "stereo_couplings.json", "electrocyclic_machinery.json",
             "stereo_branch_coverage.json", "polar.json", "radical.json", "evidence/mechanismbench_evidence.json")
    paths = [DATA / name for name in names]
    paths.extend(ROOT / "evidence" / name for name in ("stereo_rxn_closeout.json", "stereo_rxn_validation.json"))
    return {relative(path): {"sha256": digest(path), "schema": load(path)["schema"]} for path in paths}


def run_command(argv, log, timeout=600):
    start = perf_counter()
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return {"argv": argv, "cwd": str(ROOT), "returncode": completed.returncode,
            "wall_seconds": round(perf_counter() - start, 6), "log": relative(log),
            "log_sha256": digest(log)}, completed


def junit_report(path):
    document = ET.parse(path).getroot()
    suites = list(document.iter("testsuite"))
    cases = []
    for case in document.iter("testcase"):
        status = next((name for name in ("failure", "error", "skipped") if case.find(name) is not None), "passed")
        cases.append({"classname": case.get("classname"), "name": case.get("name"),
                      "status": status, "seconds": float(case.get("time", "0"))})
    return {"file": relative(path), "sha256": digest(path),
            "suite_counts": {key: sum(int(s.get(key, "0")) for s in suites)
                             for key in ("tests", "failures", "errors", "skipped")},
            "recorded_testcases_by_status": dict(Counter(c["status"] for c in cases)), "testcases": cases,
            "counting_boundary": ("JUnit testcase records include any emitted pytest subtest records; "
                                  "not chemical cases.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/stereo_rxn/results/current_validation.json")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--worker", choices=("migration", "witness-basis"))
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.worker:
        print(json.dumps(migration_report() if args.worker == "migration" else witness_basis_report(), indent=2))
        return 0
    import synkit

    output = args.output.resolve()
    artifacts = (args.artifacts or output.parent / "current_validation_artifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    sources_before, inputs_before = source_inventory(), input_inventory()
    commands = []
    workflows, _ = run_command([sys.executable, "Experiment/StereoReaction/workflows.py"], artifacts / "workflows.log")
    commands.append(workflows)
    bounded, completed = run_command([sys.executable, "Experiment/StereoReaction/validation.py"],
                                     artifacts / "bounded_validation.log", timeout=60)
    commands.append(bounded)
    bounded_payload = json.loads(completed.stdout) if completed.returncode == 0 else None
    workers = {}
    for name in ("migration", "witness-basis"):
        command, completed = run_command(
            [sys.executable, "Experiment/StereoReaction/paper_audit.py", "--worker", name],
            artifacts / (name + ".log"), timeout=60,
        )
        commands.append(command)
        workers[name] = json.loads(completed.stdout) if completed.returncode == 0 else None
    junit = artifacts / "selected_tests.xml"
    selection, _ = run_command([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *TEST_SELECTION,
                                *FOCUSED_TEST_NODES,
                                "--junitxml=" + str(junit)], artifacts / "selected_tests.log")
    commands.append(selection)
    historic = load(ROOT / "evidence/stereo_rxn_closeout.json")
    historic_input_checks = {
        name: digest(DATA / name) == historic["checksums"][name]
        for name in ("stereo.json", "stereo_couplings.json", "electrocyclic_machinery.json",
                     "stereo_branch_coverage.json")
    }
    sources_after, inputs_after = source_inventory(), input_inventory()
    checks = {
        "all_commands_succeeded": all(c["returncode"] == 0 for c in commands),
        "source_inventory_unchanged_during_audit": sources_before == sources_after,
        "input_inventory_unchanged_during_audit": inputs_before == inputs_after,
        "four_corpus_checksums_match_historical_closeout": all(historic_input_checks.values()),
        "bounded_workload_passed": bounded_payload is not None and bounded_payload["status"] == "PASS",
        "migration_no_unexpected_divergences": workers["migration"] is not None
        and workers["migration"]["summary"]["unexpected_divergences"] == 0,
        "witness_bases_all_current_outputs_match": workers["witness-basis"] is not None
        and workers["witness-basis"]["all_current_outputs_match"],
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    packages = {}
    for name in ("networkx", "rdkit", "numpy", "sympy", "pytest", "pytest-subtests"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema": "synkit.stereo-rxn-paper-audit/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "environment": {"synkit": synkit.__version__, "python": platform.python_version(),
                        "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
                        "cpu_count": os.cpu_count(), "packages": packages},
        "repository": {"head": head, "branch": branch, "tracked_worktree_dirty": bool(dirty),
                       "binding": ("Source hashes identify the tested worktree; "
                                   "HEAD alone does not bind uncommitted fixes.")},
        "source_sha256": sources_after, "inputs": inputs_after, "corpora": corpus_inventory(),
        "commands": commands, "selected_test_modules": list(TEST_SELECTION),
        "focused_test_nodes": list(FOCUSED_TEST_NODES),
        "selected_tests": junit_report(junit) if junit.exists() else None,
        "bounded_validation": bounded_payload, "migration": workers["migration"],
        "witness_basis": workers["witness-basis"],
        "historical_snapshot": {"path": "evidence/stereo_rxn_closeout.json", "date": historic["date"],
                                "package_version": historic["package_version"],
                                "reported_full_suite": historic["tests"],
                                "current_corpus_checksum_matches": historic_input_checks,
                                "full_suite_rerun_in_this_audit": False},
        "interpretation_limits": [
            ("Counts assess software conformance to declared source/rule/mechanism annotations, "
             "not reaction prediction accuracy."),
            "A selected test suite does not update the historical 4055-pass full-suite claim.",
            ("One bounded serialization workload is not a scaling study "
             "or a statistically replicated performance estimate."),
            "A template-witness formula diagnostic is not a comparison of two released implementations.",
            ("The 167-case frozen mechanism report mixes 160 polar/radical arrow cases "
             "with only 7 typed stereo cases."),
            ("On Linux the bounded validation reports the maximum of sampled resident RSS values; "
             "not the operating-system high-water RSS over the full process lifetime."),
            ("Bounded proof_bytes refers to one normalized reaction-stereo value, "
             "not a fusion-proof-v2 size distribution."),
            ("No mechanism preference, kinetics, selectivity, yield, "
             "or physical configurational stability is inferred."),
        ],
        "external_actions_performed": False,
    }
    write(output, report)
    print(json.dumps({"status": report["status"], "output": relative(output), "checks": checks}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
