"""Typed virtual-reference and direct descriptor-interchange invariants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoInterchangeError,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptors_from_rdkit,
    parse_virtual_reference,
    stereomolgraph_descriptor_to_synkit,
    stereomolgraph_registry_to_synkit,
    synkit_descriptor_to_stereomolgraph,
    virtual_reference,
)
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Mechanism import MechanismModelError, StereoDescriptor


class _ExternalStereo:
    def __init__(self, atoms, parity=None):
        self.atoms = tuple(atoms)
        self.parity = parity


EXTERNAL_TYPES = {
    descriptor_class: type(external_name, (_ExternalStereo,), {})
    for descriptor_class, external_name in {
        "tetrahedral": "Tetrahedral",
        "square_planar": "SquarePlanar",
        "trigonal_bipyramidal": "TrigonalBipyramidal",
        "octahedral": "Octahedral",
        "planar_bond": "PlanarBond",
        "atrop_bond": "AtropBond",
    }.items()
}


@pytest.mark.parametrize(
    "descriptor",
    [
        TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1),
        SquarePlanarStereo((1, 2, 3, 4, "@LP:1"), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, "@H:1"), -1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, "@LP:1"), 1),
        PlanarBondStereo(("@H:3", 2, 3, 4, 5, "@LP:4"), 0),
        AtropBondStereo(("@LP:3", 2, 3, 4, 5, "@H:4"), -1),
    ],
)
def test_all_six_classes_round_trip_through_none_with_typed_sidecar(descriptor):
    external, sidecar = synkit_descriptor_to_stereomolgraph(
        descriptor,
        descriptor_types=EXTERNAL_TYPES,
    )
    restored = stereomolgraph_descriptor_to_synkit(
        external,
        virtual_references=sidecar,
    )

    assert restored == descriptor
    assert restored.canonical_form() == descriptor.canonical_form()
    assert hash(restored) == hash(descriptor)
    assert all(external.atoms[slot] is None for slot, _ in sidecar.references)


def test_hydrogen_and_lone_pair_do_not_collapse_at_transport_boundary():
    hydrogen = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    lone_pair = TetrahedralStereo((1, 2, 3, 4, "@LP:1"), 1)
    external_h, sidecar_h = synkit_descriptor_to_stereomolgraph(
        hydrogen,
        descriptor_types=EXTERNAL_TYPES,
    )
    external_lp, sidecar_lp = synkit_descriptor_to_stereomolgraph(
        lone_pair,
        descriptor_types=EXTERNAL_TYPES,
    )

    assert external_h.atoms == external_lp.atoms
    assert sidecar_h != sidecar_lp
    assert stereomolgraph_descriptor_to_synkit(
        external_h, virtual_references=sidecar_h
    ) != stereomolgraph_descriptor_to_synkit(
        external_lp, virtual_references=sidecar_lp
    )
    with pytest.raises(StereoInterchangeError, match="Unresolved virtual reference"):
        stereomolgraph_descriptor_to_synkit(external_h)


def test_virtual_reference_schema_is_strict_owner_typed_and_relabelable():
    assert str(parse_virtual_reference("@H:7")) == "@H:7"
    assert str(parse_virtual_reference("@LP:7")) == "@LP:7"
    assert parse_virtual_reference("@X:7") is None
    assert virtual_reference("LP", 7) == "@LP:7"
    assert TetrahedralStereo((7, 1, 2, 3, "@LP:7"), -1).relabel(
        {7: 70}
    ).atoms[-1] == "@LP:70"

    with pytest.raises(ValueError, match="canonical"):
        TetrahedralStereo((7, 1, 2, 3, "@X:7"), 1)
    with pytest.raises(ValueError, match="owner"):
        TetrahedralStereo((7, 1, 2, 3, "@LP:8"), 1)
    with pytest.raises(ValueError, match="canonical"):
        TetrahedralStereo((7, 1, 2, 3, None), 1)  # type: ignore[arg-type]


def test_mechanism_envelope_uses_the_same_typed_reference_contract():
    value = StereoDescriptor("tetrahedral", (7, 1, 2, 3, "@LP:7"), 1)

    assert StereoDescriptor.from_dict(value.to_dict()) == value
    with pytest.raises(MechanismModelError, match="canonical"):
        StereoDescriptor(  # type: ignore[arg-type]
            "tetrahedral",
            (7, 1, 2, 3, None),
            1,
        )


@pytest.mark.parametrize(
    ("smiles", "expected_reference"),
    [
        ("[F:1][C@H:2]([Cl:3])[Br:4]", "@H:2"),
        ("C[S@](=O)C=1C=CC=CC1", "@LP:2"),
    ],
)
def test_direct_registry_import_resolves_virtual_ligand_from_chemistry(
    smiles,
    expected_reference,
):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    expected = descriptors_from_rdkit(molecule)
    descriptor = next(iter(expected.values()))
    external, _sidecar = synkit_descriptor_to_stereomolgraph(
        descriptor,
        descriptor_types=EXTERNAL_TYPES,
    )
    external_graph = SimpleNamespace(
        atom_stereo={descriptor.center: external},
        bond_stereo={},
    )

    report = stereomolgraph_registry_to_synkit(
        external_graph,
        source_molecule=molecule,
        lewis_graph=MolToGraph().transform(molecule),
    )

    assert report.lossless
    assert report.exclusions == ()
    assert report.descriptors == expected
    assert expected_reference in descriptor.atoms


def test_rdkit_planar_lone_pair_endpoints_are_typed():
    molecule = Chem.MolFromSmiles("N(=N\\C1=CC=CC=C1)\\C2=CC=CC=C2")
    assert molecule is not None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    descriptor = next(iter(descriptors_from_rdkit(molecule).values()))

    assert isinstance(descriptor, PlanarBondStereo)
    assert "@LP:1" in descriptor.atoms
    assert "@LP:2" in descriptor.atoms


def test_rdkit_2d_projection_accounts_for_out_of_boundary_descriptors():
    molecule = Chem.MolFromSmiles("[F:1][C@H:2]([Cl:3])[Br:4]")
    assert molecule is not None
    expected = next(iter(descriptors_from_rdkit(molecule).values()))
    selected, _sidecar = synkit_descriptor_to_stereomolgraph(
        expected,
        descriptor_types=EXTERNAL_TYPES,
    )
    extra = EXTERNAL_TYPES["tetrahedral"]((1, 2, 3, 4, 5), None)
    external_graph = SimpleNamespace(
        atom_stereo={2: selected, 1: extra},
        bond_stereo={},
    )

    report = stereomolgraph_registry_to_synkit(
        external_graph,
        source_molecule=molecule,
    )

    assert report.lossless
    assert report.descriptors == {"atom:2": expected}
    assert [issue.code for issue in report.exclusions] == [
        "UNSPECIFIED_ATOM_STEREO_EXCLUDED"
    ]


def test_selected_projection_reports_an_unresolvable_reference_as_loss():
    molecule = Chem.MolFromSmiles("[F:1][C@:2]([Cl:3])([Br:4])[I:5]")
    assert molecule is not None
    malformed = EXTERNAL_TYPES["tetrahedral"]((2, 1, 3, 4, None), 1)
    external_graph = SimpleNamespace(
        atom_stereo={2: malformed},
        bond_stereo={},
    )

    report = stereomolgraph_registry_to_synkit(
        external_graph,
        source_molecule=molecule,
    )

    assert not report.lossless
    assert report.descriptors == {}
    assert [(issue.code, issue.locus) for issue in report.losses] == [
        ("UNRESOLVED_REFERENCE", "atom:2")
    ]
