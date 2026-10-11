import pytest

from synkit.Chem.Mapper.prediction_adapter import (
    align_mapped_prediction,
    unmapped_input,
)


def test_maps_removed_before_input_serialization():
    a = unmapped_input("[CH3:80][OH:2]>>[CH3:80][OH:2]")
    b = unmapped_input("[CH3:1][OH:900]>>[CH3:1][OH:900]")
    assert a == b == "CO>>CO"


def test_output_reordering_without_reference_alignment():
    result = align_mapped_prediction("CO>>CO", "[OH:12][CH3:7]>>[CH3:7][OH:12]")
    assert result.mapping == (0, 1)
    assert result.reactant_to_output == (1, 0)
    assert result.product_to_output == (0, 1)


@pytest.mark.parametrize(
    "output",
    [
        "[CH3:1]O>>[CH3:1][OH:2]",  # missing map
        "[CH3:1][OH:1]>>[CH3:1][OH:2]",  # duplicate map
        "[CH3:1][OH:2]>>[CH3:1][OH:3]",  # different map inventories
        "[CH3:1][OH:2]>>[CH3:2][OH:1]",  # element mismatch
        "[CH2:1]=[O:2]>>[CH2:1]=[O:2]",  # changed endpoints
        "[CH3:1][OH:2].[ClH:3]>>[CH3:1][OH:2].[ClH:3]",  # added atoms
    ],
)
def test_invalid_outputs_rejected(output):
    with pytest.raises(ValueError):
        align_mapped_prediction("CO>>CO", output)
