from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.slap.lap import chemical_distance, solve_lap


def test_lap_helpers_solve_assignment_and_distance():
    _, _, value = solve_lap([[0.0, 2.0], [2.0, 0.0]])
    lgp = smiles2lgp("CC>>CC", add_Hs=False)

    assert value == 0.0
    assert chemical_distance(lgp, [0, 1], binary=True) == 0.0
