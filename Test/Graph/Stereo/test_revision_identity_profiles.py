"""Cross-operation profile and represented-information contracts."""

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    FrameworkFrame, FrameworkStereo, ResonanceFamilyCanonicalResult,
    StereoisomerRelation, StereographMirrorStatus, TetrahedralStereo,
    canonicalize_rdkit_stereograph, classify_rdkit_stereograph_mirror,
    classify_rdkit_stereoisomer_relation, classify_stereoisomer_relation,
)


@pytest.mark.parametrize(
    "left,right,chemical,lewis",
    [
        ("C1=C2[C@@H]3C(C=NC4=CC=CC(=N1)[C@@H]43)=CC=C2",
         "C1=C2[C@H]3C(C=NC4=CC=CC(=N1)[C@H]43)=CC=C2",
         StereoisomerRelation.IDENTICAL, StereoisomerRelation.ENANTIOMERS),
        ("CO[P@](=O)(N)[O-]", "CO[P@@](=O)(N)[O-]",
         StereoisomerRelation.IDENTICAL, StereoisomerRelation.ENANTIOMERS),
        ("CO[P@](=O)(N)[18O-]", "CO[P@@](=O)(N)[18O-]",
         StereoisomerRelation.ENANTIOMERS, StereoisomerRelation.ENANTIOMERS),
    ],
)
def test_canonical_pairwise_and_mirror_agree_within_profile(left, right, chemical, lewis):
    lmol, rmol = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    for profile, expected in [("chemical", chemical), ("lewis_state", lewis)]:
        lcode = canonicalize_rdkit_stereograph(lmol, identity_profile=profile)
        rcode = canonicalize_rdkit_stereograph(rmol, identity_profile=profile)
        pair = classify_rdkit_stereoisomer_relation(lmol, rmol, identity_profile=profile)
        mirror = classify_rdkit_stereograph_mirror(lmol, identity_profile=profile)
        assert pair.relation is expected
        assert pair.identity_profile == mirror.identity_profile == profile
        assert lcode.same_stereograph(rcode) == (expected is StereoisomerRelation.IDENTICAL)
        assert mirror.status is (
            StereographMirrorStatus.ACHIRAL if expected is StereoisomerRelation.IDENTICAL
            else StereographMirrorStatus.CHIRAL
        )
        assert lcode.same_stereograph(mirror.original_identity)
        assert mirror.original_identity.same_stereograph(mirror.mirror_identity) == (
            mirror.status is StereographMirrorStatus.ACHIRAL
        )


@pytest.mark.parametrize("profile", ["chemical", "lewis_state", "acs_topology"])
def test_profile_identity_survives_atom_renumbering(profile):
    molecule = Chem.MolFromSmiles("CO[P@](=O)(N)[18O-]")
    renumbered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
    left = canonicalize_rdkit_stereograph(molecule, identity_profile=profile)
    right = canonicalize_rdkit_stereograph(renumbered, identity_profile=profile)
    assert left.same_stereograph(right)
    assert left.canonical_digest == right.canonical_digest
    assert classify_rdkit_stereoisomer_relation(
        molecule, renumbered, identity_profile=profile,
    ).relation is StereoisomerRelation.IDENTICAL
    if profile == "chemical":
        assert isinstance(left, ResonanceFamilyCanonicalResult)
        assert len(left.members) == 2


def test_certificate_namespaces_prevent_cross_profile_equality():
    molecule = Chem.MolFromSmiles("CC")
    results = [canonicalize_rdkit_stereograph(molecule, identity_profile=p)
               for p in ("chemical", "lewis_state", "acs_topology")]
    assert len({result.canonical_code for result in results}) == 3
    assert all(not left.same_stereograph(right) for i, left in enumerate(results)
               for right in results[i + 1:])


def test_different_declared_inventory_is_not_diastereomerism():
    graph = nx.star_graph(4)
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    result = classify_stereoisomer_relation(graph, {}, graph, {"center": descriptor})
    assert result.relation is StereoisomerRelation.INFORMATION_MISMATCH
    assert not result.is_definitive


def test_framework_coupling_remains_part_of_information_boundary():
    graph = nx.Graph()
    for center in (0, 5):
        graph.add_node(center, element=f"center:{center}")
        for ligand in range(center + 1, center + 5):
            graph.add_node(ligand, element=f"ligand:{ligand}")
            graph.add_edge(center, ligand, order=1)
    graph.add_edge(0, 5, order=1)
    frames = (FrameworkFrame(0, (1, 2, 3, 4), 1), FrameworkFrame(5, (6, 7, 8, 9), 1))
    changed = (frames[0], FrameworkFrame(5, (6, 7, 8, 9), -1))
    left, right = (FrameworkStereo(frozenset(graph), values, 1) for values in (frames, changed))
    result = classify_stereoisomer_relation(graph, {"f": left}, graph, {"f": right})
    assert result.relation is StereoisomerRelation.INFORMATION_MISMATCH


@pytest.mark.parametrize("operation", [canonicalize_rdkit_stereograph, classify_rdkit_stereograph_mirror])
def test_invalid_profiles_refuse(operation):
    with pytest.raises(ValueError, match="profile"):
        operation(Chem.MolFromSmiles("CC"), identity_profile="undefined")
