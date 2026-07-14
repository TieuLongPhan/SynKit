"""RDKit conversion for SynKit relative stereo descriptors."""

from __future__ import annotations

from typing import Iterable

from rdkit import Chem

from .descriptors import (
    PlanarBondStereo,
    StereoValue,
    TetrahedralStereo,
    descriptor_id,
)


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


def _virtual_h(center: int) -> str:
    return f"@H:{center}"


def descriptors_from_rdkit(
    mol: Chem.Mol, *, require_atom_maps: bool = True
) -> dict[str, StereoValue]:
    """Extract SynKit's RDKit-interoperable tetrahedral/planar subset."""
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    ids = _atom_ids(mol, require_maps=require_atom_maps)
    descriptors: dict[str, StereoValue] = {}
    tetra_tags = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    }
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in tetra_tags:
            continue
        center = ids[atom.GetIdx()]
        refs: list[int | str] = [
            ids[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()
        ]
        if len(refs) == 3:
            refs.append(_virtual_h(center))
        if len(refs) != 4:
            continue
        descriptor = TetrahedralStereo((center, *refs), tetra_tags[tag], "rdkit")
        descriptors[descriptor_id(descriptor)] = descriptor

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
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
            left_refs.append(_virtual_h(left))
        if len(right_refs) == 1:
            right_refs.append(_virtual_h(right))
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

    Non-tetrahedral descriptors are valid SynKit graph/rule values, but their
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
                refs = (*refs, _virtual_h(descriptor.center))
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
