"""Independently rescore every output and audit frozen paired comparisons."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from Experiment.Synister.global_milp import doubled_distance
from Experiment.Synister.propagation_comparison import save, summarize
from Experiment.Synister.mapping_check import check_mappings
from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction


def _prepared_seeds(directory, rows):
    """Verify shared seeds against their immutable preparation records."""
    prepared = directory / "prepared_seeds.json"
    seeds = json.loads(prepared.read_text()) if prepared.exists() else None
    if seeds is not None:
        root = Path(__file__).resolve().parents[2]
        assert set(seeds) == {r["benchmark_id"] for r in rows}
        for seed in seeds.values():
            origin = root / seed["source"]
            assert sha256(origin.read_bytes()).hexdigest() == seed["source_sha256"]
            evidence = json.loads(origin.read_text())
            mapping = evidence.get("seed")
            if mapping is None:
                mapping = evidence["prediction"]["mapping"]
            assert list(mapping) == seed["mapping"]
    return seeds


def _stage_records(directory, key, name, result, seed):
    """Bind accepted output to durable solver, output and validation records."""
    prefix = directory / (key + "." + name)
    report = json.loads(Path(str(prefix) + ".solver.json").read_text())
    assert report["benchmark_id"] == key and report["seed"] == seed
    solver = report["methods"][name]
    for field in (
        "complete",
        "minimum_doubled_cd",
        "termination",
        "seconds",
        "mapping_count",
    ):
        assert solver[field] == result[field]
    output = json.loads(Path(str(prefix) + ".output.json").read_text())
    validation = json.loads(Path(str(prefix) + ".validation.json").read_text())
    assert result["validation_complete"] and validation["complete"]
    assert (
        result["mapping_sha256"]
        == output["mapping_sha256"]
        == validation["mapping_sha256"]
    )
    assert validation["mapping_count"] == result["mapping_count"]


def _protocol(directory):
    """Check immutable input, runtime and shared-seed artifact identities."""
    protocol = json.loads((directory / "protocol.json").read_text())
    inputs = directory / "inputs.json"
    assert sha256(inputs.read_bytes()).hexdigest() == protocol["inputs_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert (
            sha256((directory / "frozen_source" / name).read_bytes()).hexdigest()
            == expected
        )
    external_hash = protocol.get("external_input_manifest_sha256")
    if external_hash is not None:
        external_path = directory / "external_input_manifest.json"
        assert sha256(external_path.read_bytes()).hexdigest() == external_hash
        from Experiment.Synister.propagation_external_cohort import verify_manifest

        manifest = json.loads(external_path.read_text())
        assert verify_manifest(manifest, Path(__file__).resolve().parents[2]) == json.loads(
            inputs.read_text()
        )
    if protocol.get("saved_seed_artifact_sha256"):
        assert (
            sha256((directory / "prepared_seeds.json").read_bytes()).hexdigest()
            == protocol["saved_seed_artifact_sha256"]
        )
    if protocol.get("target_mode") == "specific_cd":
        target_path = directory / "specific_cd_target_manifest.json"
        assert sha256(target_path.read_bytes()).hexdigest() == protocol[
            "specific_cd_target_manifest_sha256"
        ]
        targets = json.loads(target_path.read_text())
        assert targets["inputs_sha256"] == protocol["specific_cd_source_inputs_sha256"]
        assert targets["prepared_seeds_sha256"] == protocol[
            "specific_cd_source_seeds_sha256"
        ]
        assert protocol["specific_cd_source_inputs_sha256"] == protocol[
            "original_cohort_sha256"
        ]
    return protocol, inputs


def audit(directory):
    """Validate cohort, source hashes, bijections, costs and comparable sets."""
    protocol, inputs = _protocol(directory)
    rows = json.loads(inputs.read_text())
    seeds = _prepared_seeds(directory, rows)
    targets = None
    if protocol.get("target_mode") == "specific_cd":
        targets = json.loads(
            (directory / "specific_cd_target_manifest.json").read_text()
        )["targets"]
    total = protocol.get("cohort_size", 100)
    assert len(rows) == total
    assert len({r["benchmark_id"] for r in rows}) == total
    expected_records = {r["benchmark_id"] + ".result.json" for r in rows}
    assert {
        p.name for p in directory.glob("reaction_*.result.json")
    } == expected_records
    checked = 0
    errors = []
    for row in rows:
        key = row["benchmark_id"]
        record = json.loads((directory / (key + ".result.json")).read_text())
        assert record["benchmark_id"] == key
        if "error" in record:
            errors.append(key)
            continue
        r, p = parse_reaction(row["reaction"])
        seed = record.get("seed")
        if seed is None:
            assert seeds is not None
            assert all(
                m["termination"] == "parent_time_limit"
                for m in record["methods"].values()
            )
            seed = seeds[key]["mapping"]
        if seeds is not None:
            assert seeds[key]["reaction"] == row["reaction"]
            assert seeds[key]["mapping"] == seed
        extract_label(r, p, seed)
        if "seed_doubled_cd" in record:
            assert doubled_distance(r, p, seed) == record["seed_doubled_cd"]
        if targets is not None:
            frozen_target = targets[key]
            assert frozen_target["reaction_sha256"] == sha256(
                row["reaction"].encode()
            ).hexdigest()
            assert frozen_target["seed_mapping"] == seed
            assert frozen_target["target_doubled_cd"] == doubled_distance(r, p, seed)
        sets = {}
        for name in ("legacy", "synister_cp"):
            result = record["methods"][name]
            if result.get("solver_outcome_known") is False:
                raise ValueError(
                    "Unknown solver outcome requires explicit investigation: "
                    + key
                    + "."
                    + name
                )
            if record.get("schema_version") == 2:
                _stage_records(directory, key, name, result, seed)
            export = directory / (key + "." + name + ".maps.json")
            assert sha256(export.read_bytes()).hexdigest() == result["mapping_sha256"]
            maps = json.loads(export.read_text())
            sets[name] = set(map(tuple, maps))
            assert len(sets[name]) == len(maps) == result["mapping_count"]
            expected_cd = (
                result["minimum_doubled_cd"]
                if targets is None
                else frozen_target["target_doubled_cd"]
            )
            if targets is not None:
                assert result["target_mode"] == "specific_cd"
                assert result["target_doubled_cd"] == expected_cd
            else:
                assert result["target_mode"] == "minimal"
            checked += check_mappings(r, p, maps, expected_cd)
        a, b = (record["methods"][name] for name in ("legacy", "synister_cp"))
        if targets is not None:
            assert a["target_doubled_cd"] == b["target_doubled_cd"]
            if a["complete"]:
                assert sets["synister_cp"] <= sets["legacy"]
            if b["complete"]:
                assert sets["legacy"] <= sets["synister_cp"]
        elif a["minimum_doubled_cd"] is not None and b["minimum_doubled_cd"] is not None:
            assert a["minimum_doubled_cd"] == b["minimum_doubled_cd"]
            if a["complete"]:
                assert sets["synister_cp"] <= sets["legacy"]
            if b["complete"]:
                assert sets["legacy"] <= sets["synister_cp"]
    summary = summarize(directory, total)
    assert summary == json.loads((directory / "summary.json").read_text())
    result = {
        "all_recorded_outputs_consistent": True,
        "independently_rescored_maps": checked,
        "paired_solver_records": total - len(errors),
        "preparation_or_worker_errors": errors,
        "summary_sha256": sha256((directory / "summary.json").read_bytes()).hexdigest(),
    }
    save(directory / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory)), flush=True)
