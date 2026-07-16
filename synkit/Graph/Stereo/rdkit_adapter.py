"""RDKit conversion for SynKit relative stereo descriptors."""

from __future__ import annotations

from typing import Iterable

from rdkit import Chem

from .descriptors import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    virtual_reference,
)


# RDKit's square-planar permutation numbers describe arrangements relative to
# the atom's current local neighbor order; they are not global stereo labels.
# These position maps are the SP1/SP2/SP3 ligand-numbering table from RDKit's
# non-tetrahedral SMILES documentation.  SynKit stores the resulting cyclic
# ligand order, whose identity is invariant under D4 rotations/reflections.
_SQUARE_PLANAR_POSITION_MAPS = {
    1: (0, 1, 2, 3),
    2: (0, 2, 1, 3),
    3: (0, 1, 3, 2),
}

# A/B are the two axial positions and C/D/E are the cyclic equatorial
# positions in RDKit's documented TB table.  Each value maps RDKit's current
# local neighbor order into that positional SynKit representation.
_TRIGONAL_BIPYRAMIDAL_POSITION_MAPS = {
    1: (0, 4, 1, 2, 3),
    2: (0, 4, 1, 3, 2),
    3: (0, 3, 1, 2, 4),
    4: (0, 3, 1, 4, 2),
    5: (0, 2, 1, 3, 4),
    6: (0, 2, 1, 4, 3),
    7: (0, 1, 2, 3, 4),
    8: (0, 1, 2, 4, 3),
    9: (1, 4, 0, 2, 3),
    10: (1, 3, 0, 2, 4),
    11: (1, 4, 0, 3, 2),
    12: (1, 3, 0, 4, 2),
    13: (1, 2, 0, 3, 4),
    14: (1, 2, 0, 4, 3),
    15: (2, 4, 0, 1, 3),
    16: (2, 3, 0, 1, 4),
    17: (3, 4, 0, 1, 2),
    18: (3, 4, 0, 2, 1),
    19: (2, 3, 0, 4, 1),
    20: (2, 4, 0, 3, 1),
}

# A/B are an opposite pair and C/D/E/F are the cyclic square in RDKit's
# documented OH table.  As for SP/TBP, these are local-order transforms, not
# global labels and not SynKit descriptor identity.
_OCTAHEDRAL_POSITION_MAPS = {
    1: (0, 5, 1, 2, 3, 4),
    2: (0, 5, 1, 4, 3, 2),
    3: (0, 4, 1, 2, 3, 5),
    4: (0, 5, 1, 2, 4, 3),
    5: (0, 4, 1, 2, 5, 3),
    6: (0, 3, 1, 2, 4, 5),
    7: (0, 3, 1, 2, 5, 4),
    8: (0, 5, 1, 3, 2, 4),
    9: (0, 4, 1, 3, 2, 5),
    10: (0, 5, 1, 4, 2, 3),
    11: (0, 4, 1, 5, 2, 3),
    12: (0, 3, 1, 4, 2, 5),
    13: (0, 3, 1, 5, 2, 4),
    14: (0, 5, 1, 3, 4, 2),
    15: (0, 4, 1, 3, 5, 2),
    16: (0, 4, 1, 5, 3, 2),
    17: (0, 3, 1, 4, 5, 2),
    18: (0, 3, 1, 5, 4, 2),
    19: (0, 2, 1, 3, 4, 5),
    20: (0, 2, 1, 3, 5, 4),
    21: (0, 2, 1, 4, 3, 5),
    22: (0, 2, 1, 5, 3, 4),
    23: (0, 2, 1, 4, 5, 3),
    24: (0, 2, 1, 5, 4, 3),
    25: (0, 1, 2, 3, 4, 5),
    26: (0, 1, 2, 3, 5, 4),
    27: (0, 1, 2, 4, 3, 5),
    28: (0, 1, 2, 5, 3, 4),
    29: (0, 1, 2, 4, 5, 3),
    30: (0, 1, 2, 5, 4, 3),
}


def _square_planar_permutations(mol: Chem.Mol) -> dict[int, int]:
    """Return RDKit square-planar permutations keyed by center atom index."""
    return {
        int(info.centeredOn): int(info.permutation)
        for info in Chem.FindPotentialStereo(mol)
        if info.type == Chem.StereoType.Atom_SquarePlanar
        and mol.GetAtomWithIdx(int(info.centeredOn)).GetChiralTag()
        == Chem.ChiralType.CHI_SQUAREPLANAR
    }


def _trigonal_bipyramidal_permutations(mol: Chem.Mol) -> dict[int, int]:
    """Return RDKit TBP permutations keyed by center atom index."""
    return {
        int(info.centeredOn): int(info.permutation)
        for info in Chem.FindPotentialStereo(mol)
        if info.type == Chem.StereoType.Atom_TrigonalBipyramidal
        and mol.GetAtomWithIdx(int(info.centeredOn)).GetChiralTag()
        == Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL
    }


def _octahedral_permutations(mol: Chem.Mol) -> dict[int, int]:
    """Return RDKit octahedral permutations keyed by center atom index."""
    return {
        int(info.centeredOn): int(info.permutation)
        for info in Chem.FindPotentialStereo(mol)
        if info.type == Chem.StereoType.Atom_Octahedral
        and mol.GetAtomWithIdx(int(info.centeredOn)).GetChiralTag()
        == Chem.ChiralType.CHI_OCTAHEDRAL
    }


def _non_tetrahedral_local_references(
    atom: Chem.Atom,
    ids: dict[int, int],
    center: int,
    coordination: int,
    descriptor_name: str,
) -> tuple[int | str, ...]:
    """Resolve a local ligand order without inventing vacant sites."""
    refs: list[int | str] = [
        ids[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()
    ]
    if len(refs) == coordination:
        return tuple(refs)
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
    if len(refs) + hidden_hydrogens != coordination:
        raise ValueError(
            f"{descriptor_name} center {center} requires exactly "
            f"{coordination} represented "
            "or hydrogen ligands; vacant or otherwise missing coordination "
            "sites cannot be inferred safely."
        )
    refs.extend(virtual_reference("H", center) for _ in range(hidden_hydrogens))
    return tuple(refs)


def _atrop_end_references(
    atom: Chem.Atom,
    other_index: int,
    ids: dict[int, int],
    center: int,
) -> tuple[int | str, int | str]:
    """Resolve two local atrop substituents without guessing vacant sites."""
    refs: list[int | str] = [
        ids[neighbor.GetIdx()]
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != other_index
    ]
    if len(refs) == 2:
        return tuple(refs)  # type: ignore[return-value]
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
    if len(refs) + hidden_hydrogens != 2:
        raise ValueError(
            f"Atrop-bond end {center} requires exactly two represented or "
            "hydrogen substituents."
        )
    refs.extend(virtual_reference("H", center) for _ in range(hidden_hydrogens))
    return tuple(refs)  # type: ignore[return-value]


def _atom_ids(mol: Chem.Mol, *, require_maps: bool) -> dict[int, int]:
    values = {}
    for atom in mol.GetAtoms():
        value = int(atom.GetAtomMapNum()) if require_maps else atom.GetIdx() + 1
        if value <= 0:
            raise ValueError("Stereo conversion requires non-zero atom maps.")
        if value in values.values():
            raise ValueError("Stereo conversion requires unique atom maps.")
        values[atom.GetIdx()] = value
    return values


def _virtual_ligand(atom: Chem.Atom, center: int) -> str:
    """Resolve RDKit's unrepresented ligand without guessing its identity."""
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
    if hidden_hydrogens > 0:
        return virtual_reference("H", center)

    # Keep this import local: MolToGraph calls this adapter while constructing
    # its registry. Its Lewis estimate supplies the electronic distinction
    # that RDKit's absent-neighbor representation itself does not encode.
    from synkit.IO.mol_to_graph import MolToGraph

    if MolToGraph.estimate_lone_pairs(atom) > 0:
        return virtual_reference("LP", center)
    raise ValueError(
        f"Missing stereo ligand at atom {center} is neither an unrepresented "
        "hydrogen nor a Lewis lone pair."
    )


def descriptors_from_rdkit(
    mol: Chem.Mol, *, require_atom_maps: bool = True
) -> dict[str, StereoValue]:
    """Extract SynKit's RDKit-interoperable stereo descriptor subset."""
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    ids = _atom_ids(mol, require_maps=require_atom_maps)
    descriptors: dict[str, StereoValue] = {}
    square_planar_permutations = _square_planar_permutations(mol)
    trigonal_bipyramidal_permutations = _trigonal_bipyramidal_permutations(mol)
    octahedral_permutations = _octahedral_permutations(mol)
    tetra_tags = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    }
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_SQUAREPLANAR:
            center = ids[atom.GetIdx()]
            permutation = square_planar_permutations.get(atom.GetIdx())
            positions = _SQUARE_PLANAR_POSITION_MAPS.get(permutation)
            if positions is None:
                raise ValueError(
                    f"Square-planar center {center} has unsupported RDKit "
                    f"permutation {permutation!r}."
                )
            local_refs = _non_tetrahedral_local_references(
                atom,
                ids,
                center,
                4,
                "Square-planar",
            )
            cyclic_refs = tuple(local_refs[index] for index in positions)
            descriptor = SquarePlanarStereo(
                (center, *cyclic_refs),
                0,
                "rdkit",
            )
            descriptors[descriptor_id(descriptor)] = descriptor
            continue
        if tag == Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL:
            center = ids[atom.GetIdx()]
            permutation = trigonal_bipyramidal_permutations.get(atom.GetIdx())
            positions = _TRIGONAL_BIPYRAMIDAL_POSITION_MAPS.get(permutation)
            if positions is None:
                raise ValueError(
                    f"Trigonal-bipyramidal center {center} has unsupported "
                    f"RDKit permutation {permutation!r}."
                )
            local_refs = _non_tetrahedral_local_references(
                atom,
                ids,
                center,
                5,
                "Trigonal-bipyramidal",
            )
            positional_refs = tuple(local_refs[index] for index in positions)
            descriptor = TrigonalBipyramidalStereo(
                (center, *positional_refs),
                1,
                "rdkit",
            )
            descriptors[descriptor_id(descriptor)] = descriptor
            continue
        if tag == Chem.ChiralType.CHI_OCTAHEDRAL:
            center = ids[atom.GetIdx()]
            permutation = octahedral_permutations.get(atom.GetIdx())
            positions = _OCTAHEDRAL_POSITION_MAPS.get(permutation)
            if positions is None:
                raise ValueError(
                    f"Octahedral center {center} has unsupported RDKit "
                    f"permutation {permutation!r}."
                )
            local_refs = _non_tetrahedral_local_references(
                atom,
                ids,
                center,
                6,
                "Octahedral",
            )
            positional_refs = tuple(local_refs[index] for index in positions)
            descriptor = OctahedralStereo(
                (center, *positional_refs),
                1,
                "rdkit",
            )
            descriptors[descriptor_id(descriptor)] = descriptor
            continue
        if tag not in tetra_tags:
            continue
        center = ids[atom.GetIdx()]
        refs: list[int | str] = [
            ids[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()
        ]
        if len(refs) == 3:
            refs.append(_virtual_ligand(atom, center))
        if len(refs) != 4:
            continue
        descriptor = TetrahedralStereo((center, *refs), tetra_tags[tag], "rdkit")
        descriptors[descriptor_id(descriptor)] = descriptor

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo in {
            Chem.BondStereo.STEREOATROPCW,
            Chem.BondStereo.STEREOATROPCCW,
        }:
            left_index = bond.GetBeginAtomIdx()
            right_index = bond.GetEndAtomIdx()
            left = ids[left_index]
            right = ids[right_index]
            left_refs = _atrop_end_references(
                mol.GetAtomWithIdx(left_index),
                right_index,
                ids,
                left,
            )
            right_refs = _atrop_end_references(
                mol.GetAtomWithIdx(right_index),
                left_index,
                ids,
                right,
            )
            parity = {
                Chem.BondStereo.STEREOATROPCW: 1,
                Chem.BondStereo.STEREOATROPCCW: -1,
            }[stereo]
            descriptor = AtropBondStereo(
                (*left_refs, left, right, *right_refs),
                parity,
                "rdkit",
            )
            descriptors[descriptor_id(descriptor)] = descriptor
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if stereo not in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            continue
        left_idx, right_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left, right = ids[left_idx], ids[right_idx]
        stereo_atoms = tuple(bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            continue
        selected_left, selected_right = stereo_atoms
        left_refs: list[int | str] = [ids[selected_left]]
        left_refs.extend(
            ids[neighbor.GetIdx()]
            for neighbor in mol.GetAtomWithIdx(left_idx).GetNeighbors()
            if neighbor.GetIdx() not in {right_idx, selected_left}
        )
        right_refs: list[int | str] = [ids[selected_right]]
        right_refs.extend(
            ids[neighbor.GetIdx()]
            for neighbor in mol.GetAtomWithIdx(right_idx).GetNeighbors()
            if neighbor.GetIdx() not in {left_idx, selected_right}
        )
        if len(left_refs) == 1:
            left_refs.append(
                _virtual_ligand(mol.GetAtomWithIdx(left_idx), left)
            )
        if len(right_refs) == 1:
            right_refs.append(
                _virtual_ligand(mol.GetAtomWithIdx(right_idx), right)
            )
        if len(left_refs) != 2 or len(right_refs) != 2:
            continue
        if stereo == Chem.BondStereo.STEREOE:
            right_refs.reverse()
        descriptor = PlanarBondStereo(
            (*left_refs, left, right, *right_refs), 0, "rdkit"
        )
        descriptors[descriptor_id(descriptor)] = descriptor
    return descriptors


def apply_stereo_to_rdkit(
    mol: Chem.Mol, descriptors: Iterable[StereoValue]
) -> Chem.Mol:
    """Apply the RDKit-interoperable descriptor subset in place.

    Unsupported descriptors remain valid SynKit graph/rule values, but their
    RDKit projection is not silently discarded: conversion raises until a
    class-specific, round-trip-tested adapter is available.
    """
    ids = _atom_ids(mol, require_maps=True)
    by_map = {atom_map: index for index, atom_map in ids.items()}
    for descriptor in descriptors:
        if isinstance(descriptor, TetrahedralStereo):
            if (
                not isinstance(descriptor.center, int)
                or descriptor.center not in by_map
            ):
                raise ValueError("Tetrahedral center is absent from RDKit molecule.")
            atom = mol.GetAtomWithIdx(by_map[descriptor.center])
            refs = tuple(ids[neighbor.GetIdx()] for neighbor in atom.GetNeighbors())
            if len(refs) == 3:
                refs = (*refs, _virtual_ligand(atom, descriptor.center))
            probe = TetrahedralStereo((descriptor.center, *refs), 1)
            desired = descriptor.canonical_form()[-1]
            probe_parity = probe.canonical_form()[-1]
            raw_parity = None if desired is None else int(desired) * int(probe_parity)
            tag = {
                1: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                -1: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                None: Chem.ChiralType.CHI_UNSPECIFIED,
            }[raw_parity]
            atom.SetChiralTag(tag)
        elif isinstance(descriptor, SquarePlanarStereo):
            if (
                not isinstance(descriptor.center, int)
                or descriptor.center not in by_map
            ):
                raise ValueError("Square-planar center is absent from RDKit molecule.")
            if descriptor.parity is None:
                raise NotImplementedError(
                    "RDKit projection of an unknown square-planar orientation "
                    "would lose the distinction between an unknown descriptor "
                    "and no descriptor."
                )
            atom = mol.GetAtomWithIdx(by_map[descriptor.center])
            local_refs = _non_tetrahedral_local_references(
                atom,
                ids,
                descriptor.center,
                4,
                "Square-planar",
            )
            matching_permutations = []
            for permutation, positions in _SQUARE_PLANAR_POSITION_MAPS.items():
                cyclic_refs = tuple(local_refs[index] for index in positions)
                candidate = SquarePlanarStereo(
                    (descriptor.center, *cyclic_refs),
                    0,
                )
                if candidate == descriptor:
                    matching_permutations.append(permutation)
            if not matching_permutations:
                raise ValueError(
                    "Square-planar descriptor ligands do not match the RDKit "
                    f"coordination sphere at center {descriptor.center}."
                )

            # Duplicate ligands can make more than one raw SP permutation encode
            # the same cyclic identity.  The lowest matching value is a stable
            # serialization choice; descriptor identity does not depend on it.
            permutation = min(matching_permutations)
            atom.SetChiralTag(Chem.ChiralType.CHI_SQUAREPLANAR)
            # RDKit 2026.03 exposes permutation reads through StereoInfo but no
            # Atom setter in Python.  This is RDKit's own integer property; the
            # public FindPotentialStereo() round-trip is exercised by tests.
            atom.SetIntProp("_chiralPermutation", permutation)
        elif isinstance(descriptor, TrigonalBipyramidalStereo):
            if (
                not isinstance(descriptor.center, int)
                or descriptor.center not in by_map
            ):
                raise ValueError(
                    "Trigonal-bipyramidal center is absent from RDKit molecule."
                )
            if descriptor.parity is None:
                raise NotImplementedError(
                    "RDKit projection of an unknown trigonal-bipyramidal "
                    "orientation would lose the distinction between an "
                    "unknown descriptor and no descriptor."
                )
            atom = mol.GetAtomWithIdx(by_map[descriptor.center])
            local_refs = _non_tetrahedral_local_references(
                atom,
                ids,
                descriptor.center,
                5,
                "Trigonal-bipyramidal",
            )
            matching_permutations = []
            for (
                permutation,
                positions,
            ) in _TRIGONAL_BIPYRAMIDAL_POSITION_MAPS.items():
                positional_refs = tuple(local_refs[index] for index in positions)
                candidate = TrigonalBipyramidalStereo(
                    (descriptor.center, *positional_refs),
                    1,
                )
                if candidate == descriptor:
                    matching_permutations.append(permutation)
            if not matching_permutations:
                raise ValueError(
                    "Trigonal-bipyramidal descriptor ligands do not match the "
                    f"RDKit coordination sphere at center {descriptor.center}."
                )

            permutation = min(matching_permutations)
            atom.SetChiralTag(Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL)
            atom.SetIntProp("_chiralPermutation", permutation)
        elif isinstance(descriptor, OctahedralStereo):
            if (
                not isinstance(descriptor.center, int)
                or descriptor.center not in by_map
            ):
                raise ValueError("Octahedral center is absent from RDKit molecule.")
            if descriptor.parity is None:
                raise NotImplementedError(
                    "RDKit projection of an unknown octahedral orientation "
                    "would lose the distinction between an unknown descriptor "
                    "and no descriptor."
                )
            atom = mol.GetAtomWithIdx(by_map[descriptor.center])
            local_refs = _non_tetrahedral_local_references(
                atom,
                ids,
                descriptor.center,
                6,
                "Octahedral",
            )
            matching_permutations = []
            for permutation, positions in _OCTAHEDRAL_POSITION_MAPS.items():
                positional_refs = tuple(local_refs[index] for index in positions)
                candidate = OctahedralStereo(
                    (descriptor.center, *positional_refs),
                    1,
                )
                if candidate == descriptor:
                    matching_permutations.append(permutation)
            if not matching_permutations:
                raise ValueError(
                    "Octahedral descriptor ligands do not match the RDKit "
                    f"coordination sphere at center {descriptor.center}."
                )

            permutation = min(matching_permutations)
            atom.SetChiralTag(Chem.ChiralType.CHI_OCTAHEDRAL)
            atom.SetIntProp("_chiralPermutation", permutation)
        elif isinstance(descriptor, AtropBondStereo):
            left, right = descriptor.atoms[2:4]
            if (
                not isinstance(left, int)
                or not isinstance(right, int)
                or left not in by_map
                or right not in by_map
            ):
                raise ValueError("Atrop-bond centers must be mapped RDKit atoms.")
            bond = mol.GetBondBetweenAtoms(by_map[left], by_map[right])
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                raise ValueError("Atrop descriptor central single bond is absent.")
            if descriptor.parity is None:
                raise NotImplementedError(
                    "RDKit projection of an unknown atrop-bond orientation "
                    "would lose the distinction between an unknown descriptor "
                    "and no descriptor."
                )

            left_index = bond.GetBeginAtomIdx()
            right_index = bond.GetEndAtomIdx()
            local_left = ids[left_index]
            local_right = ids[right_index]
            left_refs = _atrop_end_references(
                mol.GetAtomWithIdx(left_index),
                right_index,
                ids,
                local_left,
            )
            right_refs = _atrop_end_references(
                mol.GetAtomWithIdx(right_index),
                left_index,
                ids,
                local_right,
            )
            candidates = {
                AtropBondStereo(
                    (*left_refs, local_left, local_right, *right_refs),
                    parity,
                ).canonical_form(): stereo
                for stereo, parity in (
                    (Chem.BondStereo.STEREOATROPCW, 1),
                    (Chem.BondStereo.STEREOATROPCCW, -1),
                )
            }
            selected = candidates.get(descriptor.canonical_form())
            if selected is None:
                raise ValueError(
                    "Atrop descriptor substituents do not match the RDKit "
                    f"bond {left}-{right}."
                )
            bond.SetStereo(selected)
        elif isinstance(descriptor, PlanarBondStereo):
            left, right = descriptor.atoms[2:4]
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError("Planar-bond centers must be mapped atoms.")
            bond = mol.GetBondBetweenAtoms(by_map[left], by_map[right])
            if bond is None:
                raise ValueError("Planar descriptor central bond is absent.")
            if descriptor.parity is None:
                # A present descriptor with undefined orientation must remain
                # distinct from a specified E/Z arrangement at the RDKit
                # boundary. Clear neighboring slash/backslash directions too,
                # otherwise AssignStereochemistry reconstructs the old label.
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                for center in (by_map[left], by_map[right]):
                    atom = mol.GetAtomWithIdx(center)
                    for adjacent in atom.GetBonds():
                        if adjacent.GetIdx() != bond.GetIdx():
                            adjacent.SetBondDir(Chem.BondDir.NONE)
                continue
            left_refs = descriptor.atoms[:2]
            right_refs = descriptor.atoms[4:]
            real_left = next(
                (value for value in left_refs if isinstance(value, int)), None
            )
            real_right = next(
                (value for value in right_refs if isinstance(value, int)), None
            )
            if real_left is None or real_right is None:
                continue
            other_left = next(value for value in left_refs if value != real_left)
            other_right = next(value for value in right_refs if value != real_right)
            z_atoms = (real_left, other_left, left, right, real_right, other_right)
            e_atoms = (real_left, other_left, left, right, other_right, real_right)
            candidates = {}
            for stereo, atoms in (
                (Chem.BondStereo.STEREOZ, z_atoms),
                (Chem.BondStereo.STEREOE, e_atoms),
            ):
                candidate = PlanarBondStereo(atoms, descriptor.parity)
                candidates[candidate.canonical_form()] = (stereo, atoms)
            selected = candidates.get(descriptor.canonical_form())
            if selected is None:
                continue
            stereo, atoms = selected
            bond.SetStereoAtoms(by_map[real_left], by_map[real_right])
            bond.SetStereo(stereo)
        else:
            raise NotImplementedError(
                f"RDKit conversion for {descriptor.descriptor_class!r} stereo "
                "is not implemented; keep this descriptor in SynKit graph metadata."
            )
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol
