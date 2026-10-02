"""C1 pre-outcome execution lock and fail-closed launch checks."""
import argparse
import json
from pathlib import Path
import sys

from Experiment.Synister.development import ROOT, digest, encoded, environment, save
from Experiment.Synister.freeze_environment import ROOTS, closure


SETTINGS = {"limit": 1000, "workers": 4, "search_seconds": 60.0,
            "score_seconds": 30.0, "prediction_deadline": 60.0,
            "seed_policy": "best-frozen", "score_backend": "support-stabilizer"}


def source_contents():
    paths = sorted((ROOT / "synkit").rglob("*.py"))
    paths += sorted((ROOT / "Experiment/Synister").glob("*.py"))
    paths += sorted((ROOT / "Experiment/Synister/tests").glob("*.py"))
    return {str(p.relative_to(ROOT)): p.read_text() for p in paths}


def check_settings(settings):
    for name, expected in SETTINGS.items():
        if settings.get(name) != expected:
            raise ValueError(f"C1 locked setting mismatch: {name}")


def validate_sources(lock):
    if digest(encoded(source_contents())) != lock["all_sources_sha256"]:
        raise ValueError("C1 implementation changed after execution lock")


def make_lock(selection, env, protocol, replay, output):
    read = lambda path: json.loads(path.read_text())
    selected = read(selection / "manifest.json")
    assert selected["selected"] == 1000
    assert selected["scope"] == "C1_outcome_blind_confirmation_selection"
    assert selected["protocol_sha256"] == digest(protocol.read_bytes())
    audit = read(selection / "selection_audit.json")
    assert audit["status"] == "verified" and audit["manifest_sha256"] == digest((selection / "manifest.json").read_bytes())
    for key, filename in (("dataset", "inputs.csv.gz"), ("selection", "selection.json"),
                          ("frame", "frame.json"), ("accounting", "accounting.json"), ("references", "references.json")):
        assert selected[f"{key}_sha256"] == digest((selection / filename).read_bytes())
    expected = read(env / "manifest.json")
    assert expected["packages"] == closure(ROOTS)
    assert expected["requirements_sha256"] == digest((env / "requirements.lock").read_bytes())
    smoke = read(env / "smoke/summary.json")
    assert smoke["status"] == "verified"
    assert "include-system-site-packages = false" in (Path(sys.prefix) / "pyvenv.cfg").read_text().lower()
    current = environment()
    assert current["rxnmapper_resource_sha256"] == smoke["environment"]["rxnmapper_resource_sha256"]
    assert read(replay / "summary.json")["all_verified"]
    assert read(replay / "summary.json")["attempts"] == 100
    output.mkdir(parents=True, exist_ok=False)
    sources = source_contents()
    save(output / "all_sources.json", sources)
    with (output / "protocol.md").open("x") as stream:
        stream.write(protocol.read_text())
    lock = {"protocol": "identifiability-c1-confirmation-v1", "scope": "pre-outcome_execution_lock",
            "settings": SETTINGS, "margin": "1/50", "selection_manifest_sha256": digest((selection / "manifest.json").read_bytes()),
            "dataset_sha256": selected["dataset_sha256"], "protocol_sha256": digest(protocol.read_bytes()),
            "environment_manifest_sha256": digest((env / "manifest.json").read_bytes()),
            "clean_install_report_sha256": digest((env / "clean_install_report.json").read_bytes()),
            "smoke_sha256": digest((env / "smoke/summary.json").read_bytes()),
            "d1_replay_summary_sha256": digest((replay / "summary.json").read_bytes()),
            "all_sources_sha256": digest(encoded(sources)), "packages": expected["packages"],
            "rxnmapper_resource_sha256": current["rxnmapper_resource_sha256"],
            "secondary_analysis": "separately locked; not run by primary campaign",
            "policy": "all predictions frozen before searches; full-joint export; no retry replacement; no reference mappings read"}
    save(output / "lock.json", lock)
    print(json.dumps({"lock_sha256": digest((output / "lock.json").read_bytes()),
                      "sources_sha256": lock["all_sources_sha256"]}, indent=2))


def validate_launch(args):
    lock = json.loads(args.confirmation_lock.read_text())
    if lock["protocol"] == "identifiability-c2-rhea-v1":
        expected = dict(SETTINGS, limit=500)
        assert lock["settings"] == expected
        for name, value in expected.items():
            if getattr(args, name) != value:
                raise ValueError(f"C2 locked setting mismatch: {name}")
    else:
        assert lock["protocol"] == "identifiability-c1-confirmation-v1"
        check_settings(vars(args))
        assert lock["settings"] == SETTINGS
    assert args.selection_manifest is not None
    assert digest(args.selection_manifest.read_bytes()) == lock["selection_manifest_sha256"]
    assert digest(args.dataset.read_bytes()) == lock["dataset_sha256"]
    assert closure(ROOTS) == lock["packages"]
    assert environment()["rxnmapper_resource_sha256"] == lock["rxnmapper_resource_sha256"]
    validate_sources(lock)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("selection", "environment", "protocol", "replay", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    make_lock(args.selection, args.environment, args.protocol, args.replay, args.output)
