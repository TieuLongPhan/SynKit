"""Exercise reproducibility failures and self-contained route controls."""

import hashlib
import json

import pytest

from Experiment.CRN import run_all
from Experiment.CRN.benchmarks import backends
from Experiment.CRN.benchmarks.glycolysis import load_glycolysis
from Experiment.CRN.studies import biomodels, glycolysis_donors
from Experiment.CRN.tools.check_reports import ALL_STUDIES, check_reports
from synkit.CRN import SynCRN, crnt_summary
from synkit.CRN.backend import python_backend


def test_cached_glycolysis_preserves_dimensions_and_inventory_outcomes():
    model, equations = load_glycolysis()
    assert len(equations) == 34
    assert (model.n_species, model.n_reactions, crnt_summary(model).rank) == (
        40,
        34,
        23,
    )
    with python_backend():
        report = glycolysis_donors.donor_report()
    assert report["status"] == "PASS"
    assert [row["outcome"]["status"] for row in report["routes"]] == [
        "unrealizable",
        "realizable",
        "realizable",
        "realizable",
        "unrealizable",
        "realizable",
    ]


def test_biomodels_rejects_a_different_cached_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(biomodels, "CACHE_DIR", tmp_path)
    (tmp_path / "BIOMD0000000001.xml").write_text("<sbml/>")
    with pytest.raises(ValueError, match="checksum"):
        biomodels.fetch("BIOMD0000000001", offline=True)


def test_runner_records_exceptions_and_incomplete_results(tmp_path):
    def broken():
        raise ValueError("unavailable input")

    assert run_all._run("broken", broken, tmp_path)["status"] == "FAIL"
    assert (
        "unavailable input"
        in json.loads((tmp_path / "crn-broken.json").read_text())["error"]
    )

    def incomplete():
        return {"schema": "example/1", "status": "PASS", "complete": False}

    assert run_all._run("incomplete", incomplete, tmp_path)["status"] == "FAIL"
    assert (
        run_all._run("next", lambda: {"status": "PASS"}, tmp_path)["status"] == "PASS"
    )


def test_runner_includes_local_controls_without_external_documents(
    tmp_path, monkeypatch
):
    calls = []

    def report(name):
        def build(*args, **kwargs):
            calls.append(name)
            return {
                "schema": f"example.{name}/1",
                "status": "PASS",
                "checks": {"ok": True},
            }

        return build

    for module, attribute, name in (
        (run_all.validation, "validation_report", "validation"),
        (run_all.scaling, "scaling_report", "scaling"),
        (run_all.kegg_case_study, "kegg_report", "kegg"),
        (run_all.formose_case_study, "formose_report", "formose"),
        (run_all.representation, "representation_report", "representation"),
        (run_all.supplementary, "supplementary_report", "supplementary"),
        (run_all.glycolysis_donors, "donor_report", "glycolysis-donors"),
    ):
        monkeypatch.setattr(module, attribute, report(name))
    summary = tmp_path / "summary.json"
    assert (
        run_all.main(
            [
                "--backend",
                "python",
                "--quick",
                "--offline",
                "--output-dir",
                str(tmp_path / "reports"),
                "--summary",
                str(summary),
            ]
        )
        == 0
    )
    assert set(calls) == ALL_STUDIES - {
        "native-check",
        "biomodels",
        "external-crnt4sbml",
    }
    assert check_reports(summary) == []
    assert check_reports(summary, require_all=True)


@pytest.fixture
def saved_summary(tmp_path):
    rows = []
    for name in sorted(ALL_STUDIES):
        report = {
            "schema": f"example.{name}/1",
            "status": "PASS",
            "checks": {"ok": True},
        }
        path = tmp_path / f"crn-{name}.json"
        path.write_text(json.dumps(report))
        rows.append(
            {
                **report,
                "name": name,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "synkit.crn-experiments/2",
                "status": "PASS",
                "complete": True,
                "profile": "default",
                "all_studies_included": True,
                "omitted_studies": {},
                "studies": rows,
            }
        )
    )
    return summary


def test_report_verifier_rejects_missing_and_changed_reports(saved_summary):
    assert check_reports(saved_summary, require_all=True) == []
    report = saved_summary.parent / "crn-validation.json"
    report.write_text(report.read_text() + "\n")
    assert any("checksum" in error for error in check_reports(saved_summary))
    report.unlink()
    assert check_reports(saved_summary)


def test_all_results_verifier_rejects_reduced_workloads(saved_summary):
    data = json.loads(saved_summary.read_text())
    data["profile"] = "quick"
    saved_summary.write_text(json.dumps(data))
    assert check_reports(saved_summary) == []
    assert any(
        "default workload" in error
        for error in check_reports(saved_summary, require_all=True)
    )


@pytest.mark.parametrize(
    "outcome", ["complete", "limited", "mismatch", "partial_repeat"]
)
def test_backend_ratios_require_agreement_and_every_repeat(
    tmp_path, monkeypatch, outcome
):
    monkeypatch.setattr(
        backends,
        "build_kegg_crn",
        lambda *a, **kw: SynCRN.from_reaction_strings(["A>>B"]),
    )
    library = tmp_path / "library.so"
    library.with_suffix(".json").write_text("{}")
    count = {}

    def sample(network, backend, *args):
        count[backend] = count.get(backend, 0) + 1
        complete = outcome != "limited" and not (
            outcome == "partial_repeat" and count[backend] == 2
        )
        return {
            "seconds": 0.1 if backend == "cpp" else 0.2,
            "complete": complete,
            "error": None,
            "result_sha256": backend if outcome == "mismatch" else "same",
        }

    monkeypatch.setattr(backends, "_sample", sample)
    report = backends.benchmark_report(
        library=library, modules=["M00001"], repeats=2, tasks=["siphons"]
    )
    if outcome == "complete":
        assert report["status"] == "PASS"
        assert report["comparisons"][0]["speedup"] == 2
    else:
        assert report["status"] == "FAIL"
        assert report["comparisons"][0]["speedup"] is None


def test_empty_benchmark_does_not_report_success():
    report = backends.benchmark_report(modules=[], synthetic_sizes=[])
    assert report["status"] == "FAIL"
    assert report["complete"] is False
