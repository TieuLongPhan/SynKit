"""Whole-molecule chirality classification and published conformance."""

from collections import Counter
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.acs_molecular_chirality import (  # noqa: E402
    DATASET,
    load_dataset,
)
from synkit.Chem.Molecule.chirality import (  # noqa: E402
    MolecularChirality,
    PotentialStereoLocusType,
    classify_molecular_chirality,
    detect_potential_stereo_loci,
    is_molecular_chiral,
)
from synkit.Graph.Stereo import AxisStereoSupport  # noqa: E402

METADATA = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "ACS-StereoMolGraph"
    / "ci5c02523_si_002.metadata.json"
)
REPORT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "ACS-StereoMolGraph"
    / "published_chirality_benchmark.json"
)


def test_published_chirality_fixture_is_exact_and_attributed() -> None:
    rows = load_dataset(DATASET)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    assert metadata["records"] == len(rows) == 258
    assert metadata["dataset"]["doi"] == "10.1021/acs.jcim.5c02523.s002"
    assert metadata["license"]["name"] == "CC BY-NC 4.0"
    assert Counter(row["manual"] for row in rows) == {
        "Chiral": 164,
        "Achiral": 94,
    }
    assert all(row["manual"] == row["StereoMolGraph"] for row in rows)


def test_classifier_handles_local_global_and_resonance_symmetry_cases() -> None:
    rows = {row["ID"]: row for row in load_dataset(DATASET)}
    expected = {
        "VS001": "Achiral",
        "VS014": "Chiral",
        "VS042": "Achiral",
        "VS044": "Achiral",
        "VS170": "Achiral",
        "VS215": "Achiral",
        "VS280": "Chiral",
        "VS300": "Chiral",
    }

    for identifier, prediction in expected.items():
        molecule = Chem.MolFromSmiles(rows[identifier]["Input SMILES"])
        assert molecule is not None
        observed = classify_molecular_chirality(molecule)
        assert observed.classification.value == prediction
        assert observed.is_chiral is (prediction == "Chiral")
        assert is_molecular_chiral(molecule) is (prediction == "Chiral")


def test_stereo_complete_probes_explain_globally_chiral_cage() -> None:
    rows = {row["ID"]: row for row in load_dataset(DATASET)}
    molecule = Chem.MolFromSmiles(rows["VS280"]["Input SMILES"])
    assert molecule is not None

    incomplete = classify_molecular_chirality(molecule, stereo_complete=False)
    complete = classify_molecular_chirality(molecule, stereo_complete=True)

    assert incomplete.classification is MolecularChirality.ACHIRAL
    assert complete.classification is MolecularChirality.CHIRAL
    assert complete.completed_tetrahedral_centers


def test_isotope_is_part_of_molecular_identity() -> None:
    isotope_defined = Chem.MolFromSmiles("C[C@](CC)(CO)[13CH2]C")
    isotope_erased = Chem.MolFromSmiles("C[C@](CC)(CO)CC")
    assert isotope_defined is not None and isotope_erased is not None

    isotope_result = classify_molecular_chirality(isotope_defined)
    erased_result = classify_molecular_chirality(isotope_erased)

    assert isotope_result.classification is MolecularChirality.CHIRAL
    assert erased_result.classification is MolecularChirality.ACHIRAL
    assert isotope_result.identity_profile == ("element-isotope-hydrogen-connectivity")


def test_even_cumulene_axis_is_potential_not_invented_configuration() -> None:
    molecule = Chem.MolFromSmiles("ClC=C=CCl")
    achiral_control = Chem.MolFromSmiles("C=C=CCl")
    assert molecule is not None and achiral_control is not None

    result = classify_molecular_chirality(molecule)
    control_result = classify_molecular_chirality(achiral_control)

    assert result.classification is MolecularChirality.ACHIRAL
    assert result.completed_extended_tetrahedral_axes == ()
    assert len(result.potential_stereo_loci) == 1
    locus = result.potential_stereo_loci[0]
    assert locus.locus_type is PotentialStereoLocusType.CUMULENE_AXIS
    assert locus.support == AxisStereoSupport((1, 2, 3), ((0, "@H:1"), (4, "@H:3")))
    assert locus.atom_indices == (1, 2, 3)
    assert locus.orientation_state.value == "unspecified"
    assert locus.stability_status.value == "unassessed"
    assert control_result.classification is MolecularChirality.ACHIRAL
    assert control_result.potential_stereo_loci == ()


def test_biaryl_axis_detection_does_not_assign_handedness_or_stability() -> None:
    atropisomer = Chem.MolFromSmiles("O=C(O)c1cccc(Br)c1-c1c(Br)cccc1C(=O)O")
    symmetric_control = Chem.MolFromSmiles("c1ccccc1-c1ccccc1")
    assert atropisomer is not None and symmetric_control is not None

    atrop_result = classify_molecular_chirality(atropisomer)
    control_result = classify_molecular_chirality(symmetric_control)

    assert atrop_result.classification is MolecularChirality.ACHIRAL
    assert atrop_result.completed_biaryl_atrop_axes == ()
    assert len(atrop_result.potential_stereo_loci) == 1
    locus = atrop_result.potential_stereo_loci[0]
    assert locus.locus_type is PotentialStereoLocusType.ATROP_AXIS
    assert locus.orientation_state.value == "unspecified"
    assert locus.evidence_provenance == "two_dimensional_connectivity"
    assert locus.stability_status.value == "unassessed"
    assert control_result.classification is MolecularChirality.ACHIRAL


def test_potential_locus_detection_is_deterministic_and_non_mutating() -> None:
    molecule = Chem.MolFromSmiles("ClC=C=CCl")
    assert molecule is not None
    before = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)

    first = detect_potential_stereo_loci(molecule)
    second = detect_potential_stereo_loci(molecule)

    assert first == second
    assert tuple(locus.identifier for locus in first) == ("cumulene_axis:1-2-3",)
    assert Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True) == before


def test_removing_stereo_flags_makes_chiral_and_achiral_inputs_indistinguishable() -> (
    None
):
    rows = {row["ID"]: row for row in load_dataset(DATASET)}
    achiral = Chem.MolFromSmiles(rows["VS196"]["Input SMILES"])
    chiral = Chem.MolFromSmiles(rows["VS197"]["Input SMILES"])
    assert achiral is not None and chiral is not None

    Chem.RemoveStereochemistry(achiral)
    Chem.RemoveStereochemistry(chiral)

    assert Chem.MolToSmiles(
        achiral, canonical=True, isomericSmiles=False
    ) == Chem.MolToSmiles(chiral, canonical=True, isomericSmiles=False)
    achiral_result = classify_molecular_chirality(achiral)
    chiral_result = classify_molecular_chirality(chiral)
    assert achiral_result.classification is chiral_result.classification
    assert achiral_result.input_stereo_status == "underspecified"
    assert chiral_result.input_stereo_status == "underspecified"
    assert achiral_result.unspecified_stereo_loci


def test_frozen_benchmark_separates_published_labels_from_live_reproduction() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["published_stereomolgraph"]["correct"] == 258
    assert report["synkit"]["correct"] == 258
    assert report["synkit"]["confusion"] == {
        "false_achiral": 0,
        "false_chiral": 0,
        "true_achiral": 94,
        "true_chiral": 164,
    }
    assert (
        report["failure_analysis"]["assigned_local_plus_lewis_identity"]["correct"]
        == 235
    )
    assert report["stereo_stripped_input"]["correct"] == 223
    assert report["stereo_stripped_input"]["mixed_label_constitution_count"] == 6
    assert (
        report["stereo_stripped_input"]["information_theoretic_maximum_correct"] == 250
    )
    assert report["live_stereomolgraph"]["correct"] == 254
    assert report["live_stereomolgraph"][
        "published_stereomolgraph_disagreement_ids"
    ] == ["VS246", "VS247", "VS248", "VS299"]
