from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.slap.sequential import GraphMatcher


def test_sequential_graph_matcher_finds_identity_mapping():
    lgp = smiles2lgp("C>>C", add_Hs=False)
    matcher = GraphMatcher(binary=True)

    matcher.get_maps(lgp)

    assert matcher.results
    assert matcher.results[0]["val"] == 0
