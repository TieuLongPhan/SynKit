"""Declared RDKit resonance closure and profile-aware phosphate equivalence."""

from __future__ import annotations

from typing import Any, Literal

import networkx as nx

from .canonical import _rdkit_graph

MirrorIdentityProfile = Literal["chemical", "lewis_state", "acs_topology"]
_MAX_RESONANCE_FORMS = 4096


def _resonance_graphs(molecule: Any) -> tuple[nx.Graph, ...]:
    """Return RDKit resonance forms closed under the declared phosphate move.

    This is an explicit computational identity convention, not a guarantee
    of exhaustive enumeration of every chemically possible resonance form.
    The terminal phosphate move is needed because RDKit's supplier does not
    supply those charge/bond-order exchanges itself.
    """
    from rdkit import Chem

    supplier = Chem.ResonanceMolSupplier(
        molecule,
        flags=(Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION | Chem.ResonanceFlags.KEKULE_ALL),
        maxStructs=_MAX_RESONANCE_FORMS,
    )
    forms = tuple(supplier)
    if len(forms) >= _MAX_RESONANCE_FORMS:
        raise RuntimeError(
            "Exact resonance-family enumeration reached the "
            f"{_MAX_RESONANCE_FORMS}-form safety limit."
        )
    if not forms:
        forms = (Chem.Mol(molecule),)
    pending = list(forms)
    seen = set()
    graphs = []
    while pending:
        form = pending.pop()
        drawing = (
            tuple(atom.GetFormalCharge() for atom in form.GetAtoms()),
            tuple(str(bond.GetBondType()) for bond in form.GetBonds()),
        )
        if drawing in seen:
            continue
        seen.add(drawing)
        if len(seen) >= _MAX_RESONANCE_FORMS:
            raise RuntimeError(
                "Exact resonance-family closure reached the "
                f"{_MAX_RESONANCE_FORMS}-form safety limit."
            )
        normalized = Chem.Mol(form)
        Chem.SetAromaticity(normalized)
        graphs.append(_rdkit_graph(normalized))
        for center, left, right in _terminal_phosphate_pairs(form):
            alternate = Chem.RWMol(form)
            alternate.GetAtomWithIdx(left).SetFormalCharge(-1)
            alternate.GetAtomWithIdx(right).SetFormalCharge(0)
            alternate.GetBondBetweenAtoms(center, left).SetBondType(Chem.BondType.SINGLE)
            alternate.GetBondBetweenAtoms(center, right).SetBondType(Chem.BondType.DOUBLE)
            alternate = alternate.GetMol()
            Chem.SanitizeMol(alternate)
            pending.append(alternate)
    return tuple(graphs)


def _terminal_phosphate_pairs(molecule: Any) -> tuple[tuple[int, int, int], ...]:
    """Locate the narrowly declared terminal phosphate resonance move.

    Only neutral, nonradical, four-coordinate P with a terminal neutral O
    double bond and a terminal anionic O single bond is included.  Both O
    atoms have no hydrogens or radicals.  The move exchanges charge and bond
    order while preserving atom identity, isotope and total charge/valence.
    """
    from rdkit import Chem

    pairs = []
    for atom in molecule.GetAtoms():
        if (
            atom.GetAtomicNum() != 15
            or atom.GetFormalCharge() != 0
            or atom.GetDegree() != 4
            or atom.GetTotalValence() != 5
            or atom.GetNumRadicalElectrons() != 0
            or atom.GetTotalNumHs() != 0
        ):
            continue
        double = []
        single = []
        for oxygen in atom.GetNeighbors():
            if (
                oxygen.GetAtomicNum() != 8
                or oxygen.GetDegree() != 1
                or oxygen.GetTotalNumHs() != 0
                or oxygen.GetNumRadicalElectrons() != 0
                or oxygen.GetIsAromatic()
            ):
                continue
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), oxygen.GetIdx())
            if oxygen.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                double.append(oxygen.GetIdx())
            if oxygen.GetFormalCharge() == -1 and bond.GetBondType() == Chem.BondType.SINGLE:
                single.append(oxygen.GetIdx())
        pairs.extend((atom.GetIdx(), left, right) for left in double for right in single)
    return tuple(pairs)


def _is_terminal_phosphate_resonance_locus(
    molecule: Any,
    element: Any,
    *,
    identity_profile: MirrorIdentityProfile,
) -> bool:
    """Return whether an apparent P center differs only by localized resonance.

    A neutral tetra-coordinate phosphorus with terminal ``P=O`` and
    ``P-[O-]`` ligands contains two drawings of the same delocalized phosphate
    oxygen environment.  Treating bond order and formal charge as ligand
    identity makes this look tetrahedral even though exchanging those terminal
    oxygens is resonance-equivalent in the molecular-identity task.
    """
    if identity_profile == "lewis_state":
        return False
    support = element.support
    if not hasattr(support, "center"):
        return False
    # Degree-one ligands have no remaining environment.  Their transposition
    # fixes every other atom, and the declared move supplies exactly the
    # exchanged charge/bond-order drawing in the chemical profile.  The ACS
    # profile omits those attributes outright.  Isotope remains identity in
    # both profiles; atom-map labels remain transport handles only.
    return any(
        center == support.center
        and molecule.GetAtomWithIdx(left).GetIsotope()
        == molecule.GetAtomWithIdx(right).GetIsotope()
        for center, left, right in _terminal_phosphate_pairs(molecule)
    )
