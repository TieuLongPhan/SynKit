from rdkit import Chem

from synkit.Chem.Mapper.chem.smiles import (
    smiles2lgp,
    standardize_reaction_center_hydrogens,
)


def test_smiles2lgp_builds_balanced_graph_pair():
    left, right = smiles2lgp("CCO>>CC=O", add_Hs=False)

    assert len(left.labels) == len(right.labels)
    assert left.props["atomic numbers"]
    assert right.props["atomic numbers"]


def test_standardize_hydrogens_preserves_heavy_aam():
    reaction = (
        "[C:1]([H:3])([H:4])([H:5])[O:2][H:6]"
        ">>"
        "[C:1]([H:3])([H:4])([H:5])[O:2][H:6]"
    )

    standardized = standardize_reaction_center_hydrogens(reaction)
    assert "[H:" not in standardized
    for side in standardized.split(">>"):
        mol = Chem.MolFromSmiles(side)
        assert {
            atom.GetAtomMapNum()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1
        } == {1, 2}
