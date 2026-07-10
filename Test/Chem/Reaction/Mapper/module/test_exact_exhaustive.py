from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.exact.exhaustive import ExactMapper


def test_exhaustive_mapper_solves_identity_mapping():
    lgp = smiles2lgp("C>>C", add_Hs=False)
    mapper = ExactMapper()

    result = mapper.solve(lgp[0], lgp[1])

    assert result.cost == 0.0
    assert result.mapping == [0]
