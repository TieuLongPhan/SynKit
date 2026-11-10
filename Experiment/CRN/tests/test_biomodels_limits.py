"""BioModels reports must distinguish exhausted and completed enumerations."""

import pytest

from Experiment.CRN.studies import biomodels
from synkit.CRN import native_backend


def test_model175_records_small_limit_failure_and_complete_larger_search(exact_library):
    model_id = "BIOMD0000000175"
    if not (biomodels.CACHE_DIR / f"{model_id}.xml").is_file():
        pytest.skip("Optional BioModels XML cache is not distributed")

    with native_backend(exact_library):
        limited = biomodels.biomodels_report([model_id], offline=True, budget=120, siphon_max_nodes=200_000)
        complete = biomodels.biomodels_report([model_id], offline=True, budget=120, siphon_max_nodes=1_000_000)

    assert limited["status"] == "FAIL"
    assert limited["checks"]["every_structural_analysis_complete"] is False
    failed = limited["models"][0]
    assert failed["n_minimal_siphons"] is None
    assert "SiphonSearchLimit" in failed["siphon_error"]
    assert failed["persistent"] is None
    assert failed["persistence_error"] == "required enumeration incomplete"
    assert limited["parameters"]["siphon_max_nodes"] == 200_000

    assert complete["status"] == "PASS"
    assert all(complete["checks"].values())
    model = complete["models"][0]
    assert model["n_reactions"] == 194
    assert model["n_minimal_siphons"] == 145
    assert model["roundtrip_ok"] is True
    assert complete["parameters"]["siphon_max_nodes"] == 1_000_000
