"""Authorized molecule-level extended stereo integration for Sprint 30."""

import json

import pytest
from rdkit import Chem

from synkit.Chem.Molecule.chirality import (
    MolecularChirality,
    MolecularChiralityOutcome,
    UnspecifiedMolecularStereoError,
    assess_molecular_chirality,
    classify_molecular_chirality,
)
from synkit.Chem.Molecule.stereo_evidence import (
    ExtendedStereoStability,
    MolecularStereoConfiguration,
    MolecularStereoConfigurationSet,
    StereoEvidenceSource,
    StereoPopulationStatus,
)
from synkit.Graph.Stereo import (
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
)


def _cumulene(parity: int | None) -> CumuleneAxisStereo:
    return CumuleneAxisStereo(
        (1, 2, 3),
        ((0, "@H:1"), (4, 5)),
        parity,
        "declared_sidecar",
    )


def _configuration(
    parity: int,
    *,
    fraction: float | None = None,
) -> MolecularStereoConfiguration:
    return MolecularStereoConfiguration(
        (_cumulene(parity),),
        StereoEvidenceSource.DECLARED_SIDECAR,
        ExtendedStereoStability.UNASSESSED,
        fraction,
    )


def test_fixed_cumulene_sidecar_promotes_only_its_matching_potential_locus() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C(Cl)Br")
    assert molecule is not None
    configuration = _configuration(1)

    result = classify_molecular_chirality(
        molecule,
        require_specified=True,
        stereo_configuration=configuration,
    )

    assert result.classification is MolecularChirality.CHIRAL
    assert result.input_stereo_status == "specified"
    assert result.unspecified_stereo_loci == ()
    assert result.configured_extended_descriptor_count == 1
    assert result.stereo_evidence_source == "declared_sidecar"
    assert result.extended_stability_status == "unassessed"
    assert len(result.potential_stereo_loci) == 1


def test_orientation_free_cumulene_input_still_fails_closed() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C(Cl)Br")
    assert molecule is not None

    with pytest.raises(UnspecifiedMolecularStereoError):
        classify_molecular_chirality(molecule, require_specified=True)
    assessment = assess_molecular_chirality(molecule)
    assert assessment.outcome is MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE
    assert assessment.evaluated_isomer_count == 0


def test_fixed_extended_cis_trans_sidecar_is_mirror_fixed() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C=CCl")
    assert molecule is not None
    descriptor = ExtendedCisTransStereo(
        (1, 2, 3, 4),
        ((0, "@H:1"), (5, "@H:4")),
        0,
        "declared_sidecar",
    )
    configuration = MolecularStereoConfiguration(
        (descriptor,),
        StereoEvidenceSource.DECLARED_SIDECAR,
    )

    result = classify_molecular_chirality(
        molecule,
        require_specified=True,
        stereo_configuration=configuration,
    )

    assert result.classification is MolecularChirality.ACHIRAL
    assert result.unspecified_stereo_loci == ()


def test_unspecified_extended_descriptor_is_not_authorized_evidence() -> None:
    with pytest.raises(ValueError, match="requires fixed orientation"):
        MolecularStereoConfiguration(
            (_cumulene(None),),
            StereoEvidenceSource.DECLARED_SIDECAR,
        )


def test_mixed_cumulene_population_enumerates_both_declared_enantiomers() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C(Cl)Br")
    assert molecule is not None
    configuration_set = MolecularStereoConfigurationSet(
        (_configuration(1, fraction=0.5), _configuration(-1, fraction=0.5)),
        StereoPopulationStatus.MIXED,
    )

    assessment = assess_molecular_chirality(
        molecule,
        stereo_configurations=configuration_set,
        use_cache=False,
    )

    assert assessment.outcome is MolecularChiralityOutcome.NECESSARILY_CHIRAL
    assert assessment.observed_classifications == (MolecularChirality.CHIRAL,)
    assert assessment.theoretical_isomer_upper_bound == 2
    assert assessment.evaluated_isomer_count == 2
    assert assessment.enumeration_complete
    assert assessment.configured_alternative_count == 2
    assert assessment.configured_population_status == "mixed"


def test_extended_enumeration_cap_never_promotes_one_observation() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C(Cl)Br")
    assert molecule is not None
    configuration_set = MolecularStereoConfigurationSet(
        (_configuration(1), _configuration(-1)),
        StereoPopulationStatus.MIXED,
    )

    assessment = assess_molecular_chirality(
        molecule,
        max_isomers=1,
        stereo_configurations=configuration_set,
        use_cache=False,
    )

    assert assessment.outcome is MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE
    assert assessment.evaluated_isomer_count == 1
    assert assessment.theoretical_isomer_upper_bound == 2
    assert not assessment.enumeration_complete


def test_configuration_set_sidecar_round_trip_preserves_classification() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C(Cl)Br")
    assert molecule is not None
    configuration_set = MolecularStereoConfigurationSet(
        (_configuration(-1),),
        StereoPopulationStatus.PURE,
    )
    payload = json.loads(json.dumps(configuration_set.to_dict()))
    restored = MolecularStereoConfigurationSet.from_dict(payload)

    before = classify_molecular_chirality(
        molecule,
        stereo_configuration=configuration_set.configurations[0],
    )
    after = classify_molecular_chirality(
        molecule,
        stereo_configuration=restored.configurations[0],
    )
    assert restored == configuration_set
    assert before.classification is after.classification is MolecularChirality.CHIRAL
    assert before.stereo_evidence_source == after.stereo_evidence_source


def test_helical_sidecar_enters_exact_mirror_comparison_without_inference() -> None:
    molecule = Chem.MolFromSmiles("CCCC")
    assert molecule is not None
    descriptor = HelicalStereo((0, 1, 2, 3), 1, "validated_geometry")
    configuration = MolecularStereoConfiguration(
        (descriptor,),
        StereoEvidenceSource.VALIDATED_GEOMETRY,
        ExtendedStereoStability.VALIDATED,
    )

    unconfigured = classify_molecular_chirality(molecule, stereo_complete=False)
    configured = classify_molecular_chirality(
        molecule,
        stereo_complete=False,
        stereo_configuration=configuration,
    )
    assert unconfigured.classification is MolecularChirality.ACHIRAL
    assert unconfigured.configured_extended_descriptor_count == 0
    assert configured.classification is MolecularChirality.CHIRAL
    assert configured.extended_stability_status == "validated"


def test_declared_support_must_exist_in_the_molecular_topology() -> None:
    molecule = Chem.MolFromSmiles("CC.CC")
    assert molecule is not None
    descriptor = HelicalStereo((0, 1, 2, 3), 1, "declared_sidecar")
    configuration = MolecularStereoConfiguration(
        (descriptor,), StereoEvidenceSource.DECLARED_SIDECAR
    )

    with pytest.raises(ValueError, match="continuous molecular path"):
        classify_molecular_chirality(
            molecule,
            stereo_configuration=configuration,
        )


def test_population_and_stability_evidence_validate_independently() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        MolecularStereoConfigurationSet(
            (_configuration(1, fraction=0.2), _configuration(-1, fraction=0.2)),
            StereoPopulationStatus.MIXED,
        )
    with pytest.raises(ValueError, match="exactly one"):
        MolecularStereoConfigurationSet(
            (_configuration(1), _configuration(-1)),
            StereoPopulationStatus.PURE,
        )
