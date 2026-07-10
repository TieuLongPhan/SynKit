from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp


def test_smiles2lgp_builds_balanced_graph_pair():
    left, right = smiles2lgp("CCO>>CC=O", add_Hs=False)

    assert len(left.labels) == len(right.labels)
    assert left.props["atomic numbers"]
    assert right.props["atomic numbers"]
