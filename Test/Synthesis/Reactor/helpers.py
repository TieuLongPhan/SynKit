from rdkit import Chem


def canonical_fragments(side, *, stereo=False):
    """Canonical component multiset, preserving charges, isotopes and counts."""
    molecule = Chem.MolFromSmiles(side)
    if molecule is None:
        raise ValueError("An endpoint could not be parsed")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    if not stereo:
        Chem.RemoveStereochemistry(molecule)
    return tuple(
        sorted(Chem.MolToSmiles(fragment, isomericSmiles=True) for fragment in Chem.GetMolFrags(molecule, asMols=True))
    )
