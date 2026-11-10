"""The formose study must execute the selected reaction engine truthfully."""

import pytest

from Experiment.CRN.studies.formose_case_study import expand
from Experiment.CRN.tools.common import environment


@pytest.fixture(scope="module")
def generation_engines(tmp_path_factory):
    from Experiment.CRN.tools.common import _default_rdkit_sdk
    from synkit.CRN.Construct.native_expand import build_native_expander
    from synkit.CRN.Construct.native_dpo import build_native_dpo_expander

    sdk = _default_rdkit_sdk()
    if sdk is None:
        pytest.skip("An RDKit C++ SDK is required for the generation-engine regression")
    directory = tmp_path_factory.mktemp("formose-engines")
    return (
        build_native_expander(directory / "rdkit", rdkit_dir=sdk),
        build_native_dpo_expander(directory / "dpo", rdkit_dir=sdk),
    )


def test_formose_uses_dpo_when_both_native_contexts_are_active(generation_engines, monkeypatch):
    from synkit.CRN.Construct import native_expand as rdkit_engine
    from synkit.CRN.Construct.native_dpo import native_dpo_expander

    rdkit, dpo = generation_engines

    def unexpected_rdkit_execution(*args, **kwargs):
        raise AssertionError("The default formose study must execute SynReactor DPO")

    monkeypatch.setattr(rdkit_engine, "run_native_expansion", unexpected_rdkit_execution)
    record = {}
    with rdkit_engine.native_expander(rdkit), native_dpo_expander(dpo):
        network = expand(4, record)
        metadata = environment()
    assert (network.n_species, network.n_reactions) == (37, 44)
    assert record["rule_backend"] == "synreactor-native"
    assert record["engine"] == "synkit_dpo_cpp"
    assert record["reaction_engine"] == "SynReactor DPO"
    assert record["controller"] == "cpp" and record["kernel"] == "synkit_dpo_cpp"
    assert record["implicit_h"] is True and record["radical_policy"] == "ignore"
    assert metadata["crn_expander"]["reaction_engine"] == "RDKit reaction SMARTS"
    assert metadata["crn_dpo_expander"]["engine"] == "synkit_dpo_cpp"
    assert len(metadata["crn_dpo_expander"]["kernel_library_sha256"]) == 64


def test_legacy_context_alone_does_not_silently_select_rdkit(generation_engines, monkeypatch):
    from synkit.CRN.Construct import native_expand as rdkit_engine

    rdkit, _dpo = generation_engines

    def unexpected_rdkit_execution(*args, **kwargs):
        raise AssertionError("RDKit generation requires an explicit control selection")

    monkeypatch.setattr(rdkit_engine, "run_native_expansion", unexpected_rdkit_execution)
    record = {}
    with rdkit_engine.native_expander(rdkit):
        network = expand(4, record)
    assert (network.n_species, network.n_reactions) == (37, 44)
    assert record["rule_backend"] == "synreactor"
    assert record["reaction_engine"] == "SynReactor DPO"
    assert record["controller"] == "python"


def test_rdkit_control_remains_explicitly_available(generation_engines):
    from synkit.CRN.Construct.native_expand import native_expander

    rdkit, _dpo = generation_engines
    record = {}
    with native_expander(rdkit):
        network = expand(4, record, rule_backend="rdkit-native")
    assert (network.n_species, network.n_reactions) == (37, 44)
    assert record["rule_backend"] == "native"
    assert record["engine"] == "synkit_crn_expand"
    assert record["reaction_engine"] == "RDKit reaction SMARTS"
    assert record["controller"] == "cpp" and record["kernel"] == "rdkit_cpp"
