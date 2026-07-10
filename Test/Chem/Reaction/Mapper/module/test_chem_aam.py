from synkit.Chem.Reaction.Mapper.chem.aam import AAMapper


def test_aam_mapper_smoke_maps_simple_reaction():
    mapper = AAMapper(binary=True)

    mapper.map_smiles("CCO>>CC=O", add_Hs=False)

    assert mapper.results
    assert "smiles" in mapper.results[0]
    assert "cd" in mapper.results[0]
