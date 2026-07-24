"""Constitution-only topology gates for tetrahedral carrier atoms."""

from __future__ import annotations

from rdkit import Chem

from synkit.Graph.Stereo.supports import AtomStereoSupport


def hidden_hydrogen_count(atom: Chem.Atom) -> int:
    """Return represented explicit and implicit hydrogens on one atom."""
    return int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())


def lone_pair_count(atom: Chem.Atom) -> int:
    """Return one represented stereochemical lone-pair slot when supported."""
    coordination = atom.GetDegree() + hidden_hydrogen_count(atom)
    if (
        atom.GetAtomicNum() in {7, 15, 16}
        and coordination == 3
        and atom.GetFormalCharge() <= 0
    ):
        return 1
    return 0


def is_tetrahedral_carrier(atom: Chem.Atom) -> bool:
    """Return whether the represented constitution has four tetrahedral slots."""
    if atom.GetIsAromatic():
        return False
    slots = (
        atom.GetDegree()
        + hidden_hydrogen_count(atom)
        + lone_pair_count(atom)
    )
    if slots != 4:
        return False
    if atom.GetAtomicNum() not in {7, 15, 16}:
        return (
            atom.GetHybridization() == Chem.HybridizationType.SP3
            and all(
                bond.GetBondType() == Chem.BondType.SINGLE
                for bond in atom.GetBonds()
            )
        )
    return atom.GetHybridization() in {
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
    }


def detect_tetrahedral_carriers(
    molecule: Chem.Mol,
) -> tuple[AtomStereoSupport, ...]:
    """Return four-slot tetrahedral carriers before symmetry refinement."""
    if molecule is None:
        raise ValueError("Tetrahedral-carrier detection requires a molecule.")
    working = Chem.Mol(molecule)
    return tuple(
        AtomStereoSupport(atom.GetIdx())
        for atom in working.GetAtoms()
        if is_tetrahedral_carrier(atom)
    )


__all__ = [
    "detect_tetrahedral_carriers",
    "hidden_hydrogen_count",
    "is_tetrahedral_carrier",
    "lone_pair_count",
]
