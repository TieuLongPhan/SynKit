"""Result files retain scientific evidence without build fingerprints."""

from copy import deepcopy
import json

from Experiment.CRN.tools.reports import result_payload, save_result


def test_result_keeps_certificates_and_drops_nested_fingerprints(tmp_path):
    report = {
        "schema": "synkit.crn-example/1",
        "status": "PASS",
        "environment": {"python": "3.12", "crn_backend": {"library_sha256": "hidden"}},
        "source_sha256": {"source.py": "hidden"},
        "models": [{"name": "network", "sha256": "hidden", "rank": 3, "certificate": ["r1", "r2"], "seconds": 0.125}],
        "checks": {"replay_ok": True},
    }
    original = deepcopy(report)
    path = tmp_path / "result.json"
    save_result(path, report)
    saved = json.loads(path.read_text())
    assert saved["models"] == [{"name": "network", "rank": 3, "certificate": ["r1", "r2"], "seconds": 0.125}]
    assert saved["checks"] == report["checks"]
    assert "sha256" not in path.read_text() and "environment" not in saved
    assert report == original


def test_run_summary_keeps_software_versions_without_build_paths():
    saved = result_payload(
        {
            "schema": "synkit.crn-experiments/2",
            "environment": {
                "python": "3.12",
                "synkit": "1.7.0",
                "cpu_count": 22,
                "crn_backend": {
                    "production_backend": "cpp",
                    "library_path": "/tmp/native.so",
                    "build_sha256": "hidden",
                },
            },
        }
    )
    assert saved["environment"] == {
        "python": "3.12",
        "synkit": "1.7.0",
        "cpu_count": 22,
        "crn_backend": {"production_backend": "cpp"},
    }


def test_snapshot_verdict_preserves_a_failed_repeat():
    report = {
        "runs": [
            {"backend": "native", "depth": 1, "snapshot_sha256": "expected", "audits": [{"snapshot_sha256": "changed"}]}
        ],
        "schedule": [{"backend": "native", "depth": 1, "snapshot_sha256": "changed"}],
    }
    saved = result_payload(report)
    assert saved["runs"][0]["snapshot_matches"] is True
    assert saved["runs"][0]["audits"][0]["snapshot_matches"] is False
    assert saved["schedule"][0]["snapshot_matches"] is False
    assert result_payload(saved) == saved
