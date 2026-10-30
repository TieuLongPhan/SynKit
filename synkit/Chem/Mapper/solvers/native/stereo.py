"""Preserve supplied geometry and assign complete modern CIP descriptors."""

import json

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


def assign_stereo(mol):
    """Assign true CIP labels without legacy cleanup of supplied chiral tags.

    Highly symmetric recursion is bounded and raises explicitly rather than
    returning a silently incomplete stereo palette. No H atoms are created.
    """
    for obj in list(mol.GetAtoms()) + list(mol.GetBonds()):
        if obj.HasProp("_CIPCode"):
            obj.ClearProp("_CIPCode")
    rdCIPLabeler.AssignCIPLabels(mol, maxRecursiveIterations=1000000)
    return mol


def stereo_smiles(mol, *, canonical=True):
    """Serialize existing stereo without RDKit's lossy legacy cleanup pass."""
    parameters = Chem.SmilesWriteParams()
    parameters.canonical = canonical
    parameters.doIsomericSmiles = True
    parameters.cleanStereo = False
    return Chem.MolToSmiles(mol, parameters)


def endpoint_identity_key(mol):
    """Canonical colored-graph identity, avoiding stereo-SMILES rank ambiguity.

    Modern CIP is encoded as an ordinary node color. Bond state is encoded by
    colored auxiliary vertices; these are identity-code gadgets, not H atoms or
    chemical participants. The original palette accompanies the graph code.
    """
    assign_stereo(mol)
    node_tokens = [
        json.dumps(
            (
                "a",
                a.GetAtomicNum(),
                a.GetIsotope(),
                a.GetFormalCharge(),
                a.GetNumRadicalElectrons(),
                a.GetTotalNumHs(),
                a.GetIsAromatic(),
                a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "",
            )
        )
        for a in mol.GetAtoms()
    ]
    edge_tokens = [
        json.dumps(("b", str(b.GetBondType()), str(b.GetStereo())))
        for b in mol.GetBonds()
    ]
    directed = {Chem.BondType.DATIVE, Chem.BondType.DATIVEL, Chem.BondType.DATIVER}
    tokens = sorted(set(node_tokens + edge_tokens + ['"tail"', '"head"']))
    colors = {token: i + 1 for i, token in enumerate(tokens)}
    graph = Chem.RWMol()

    def add(token):
        atom = Chem.Atom(0)
        atom.SetNoImplicit(True)
        atom.SetIsotope(colors[token])
        return graph.AddAtom(atom)

    for token in node_tokens:
        add(token)
    for bond, token in zip(mol.GetBonds(), edge_tokens):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        middle = add(token)
        if bond.GetBondType() in directed:
            tail, head = add('"tail"'), add('"head"')
            graph.AddBond(i, tail, Chem.BondType.SINGLE)
            graph.AddBond(tail, middle, Chem.BondType.SINGLE)
            graph.AddBond(middle, head, Chem.BondType.SINGLE)
            graph.AddBond(head, j, Chem.BondType.SINGLE)
        else:
            graph.AddBond(i, middle, Chem.BondType.SINGLE)
            graph.AddBond(middle, j, Chem.BondType.SINGLE)
    return json.dumps(
        (tokens, Chem.MolToSmiles(graph, canonical=True, isomericSmiles=True)),
        separators=(",", ":"),
    )
