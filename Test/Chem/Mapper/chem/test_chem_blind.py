import pytest

from synkit.Chem.Mapper import blinded_mapped_reaction_problem
from synkit.Chem.Mapper.slap.lap import chemical_distance

REACTION = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"


def test_blinded_problem_is_reproducible_and_breaks_reference_alignment():
    first = blinded_mapped_reaction_problem(REACTION, blind_seed="blind-test")
    second = blinded_mapped_reaction_problem(REACTION, blind_seed="blind-test")

    assert first.reactant_atom_maps == second.reactant_atom_maps
    assert first.product_atom_maps == second.product_atom_maps
    assert first.reference_mapping == second.reference_mapping
    assert first.reactant_atom_maps != first.product_atom_maps
    assert first.reference_mapping != tuple(range(3))
    assert all(
        isinstance(label, int)
        for graph in first.lgp
        for label in graph.props["atomic numbers"]
    )
    assert chemical_distance(first.lgp, first.reference_mapping, binary=False) == 0


def test_blinded_problem_rejects_unbalanced_map_inventories():
    with pytest.raises(ValueError, match="atom inventories differ"):
        blinded_mapped_reaction_problem("[CH3:1]>>[CH3:2]")
