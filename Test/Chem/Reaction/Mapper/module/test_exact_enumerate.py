from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.exact.enumerate import mapping_to_lgp
from synkit.Chem.Reaction.Mapper.slap.lap import recover_mapping


def test_enumerate_mapping_to_lgp_round_trips_mapping():
    lgp = smiles2lgp("CC>>CC", add_Hs=False)
    result_lgp = mapping_to_lgp(lgp, [0, 1])

    assert recover_mapping(result_lgp) == [0, 1]
