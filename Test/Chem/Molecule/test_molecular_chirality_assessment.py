"""Sprint 23 configuration-aware whole-molecule chirality tests."""

from pathlib import Path
import sys

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Test.Chem.Molecule.benchmark_molecular_chirality import (  # noqa: E402
    load_dataset,
)
from synkit.Chem.Molecule.chirality import (  # noqa: E402
    MolecularChirality,
    MolecularChiralityOutcome,
    UnspecifiedMolecularStereoError,
    assess_molecular_chirality,
    classify_molecular_chirality,
)


def test_strict_binary_mode_rejects_unassigned_stereo() -> None:
    molecule = Chem.MolFromSmiles("FC(Cl)C(Br)I")
    assert molecule is not None

    with pytest.raises(UnspecifiedMolecularStereoError) as error:
        classify_molecular_chirality(molecule, require_specified=True)

    assert error.value.loci == (
        "Atom_Tetrahedral:1",
        "Atom_Tetrahedral:3",
    )


@pytest.mark.parametrize(
    ("smiles", "outcome", "upper_bound", "observed"),
    [
        (
            "FC=CCl",
            MolecularChiralityOutcome.NECESSARILY_ACHIRAL,
            2,
            (MolecularChirality.ACHIRAL,),
        ),
        (
            "FC(Cl)C(Br)I",
            MolecularChiralityOutcome.NECESSARILY_CHIRAL,
            4,
            (MolecularChirality.CHIRAL,),
        ),
    ],
)
def test_exhaustive_supported_completions_prove_necessary_outcomes(
    smiles: str,
    outcome: MolecularChiralityOutcome,
    upper_bound: int,
    observed: tuple[MolecularChirality, ...],
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    assessment = assess_molecular_chirality(molecule, use_cache=False)

    assert assessment.outcome is outcome
    assert assessment.observed_classifications == observed
    assert assessment.theoretical_isomer_upper_bound == upper_bound
    assert assessment.enumeration_complete
    assert assessment.is_definitive


def test_stereo_free_constitution_can_be_configuration_dependent() -> None:
    rows = {row["ID"]: row for row in load_dataset()}
    molecule = Chem.MolFromSmiles(rows["VS196"]["Input SMILES"])
    assert molecule is not None
    Chem.RemoveStereochemistry(molecule)

    assessment = assess_molecular_chirality(molecule, use_cache=False)

    assert assessment.outcome is MolecularChiralityOutcome.CONFIGURATION_DEPENDENT
    assert assessment.observed_classifications == (
        MolecularChirality.ACHIRAL,
        MolecularChirality.CHIRAL,
    )
    assert assessment.theoretical_isomer_upper_bound == 64
    assert assessment.evaluated_isomer_count < 9
    assert not assessment.enumeration_complete
    assert assessment.is_definitive
    assert len(assessment.representative_isomers) == 2


def test_cap_never_promotes_one_sample_to_necessary_conclusion() -> None:
    molecule = Chem.MolFromSmiles("FC(Cl)C(Br)I")
    assert molecule is not None

    assessment = assess_molecular_chirality(
        molecule,
        max_isomers=1,
        use_cache=False,
    )

    assert assessment.outcome is MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE
    assert assessment.theoretical_isomer_upper_bound == 4
    assert assessment.evaluated_isomer_count == 1
    assert not assessment.enumeration_complete
    assert not assessment.is_definitive


@pytest.mark.parametrize("max_isomers", [0, -1, 1.5, True])
def test_assessment_rejects_invalid_enumeration_caps(max_isomers: object) -> None:
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None

    with pytest.raises(ValueError, match="positive integer"):
        assess_molecular_chirality(molecule, max_isomers=max_isomers)  # type: ignore[arg-type]
