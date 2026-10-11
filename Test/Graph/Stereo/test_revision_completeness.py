"""Counterexamples and profile contracts from the manuscript revision audit."""

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    FrameworkFrame,
    FrameworkStereo,
    StereographMirrorStatus,
    canonicalize_configured_stereograph,
    classify_rdkit_configured_stereograph_mirror,
    enumerate_stereograph_assignments,
    focal_stereo_evidence,
    local_configuration_classes,
)


@pytest.mark.parametrize(
    "smiles,status",
    [
        ("C[C@H](Cl)C(Cl)C", StereographMirrorStatus.INCOMPLETE),
        ("C[C@H](Cl)[C@H](Cl)C", StereographMirrorStatus.ACHIRAL),
        ("C[C@H](Cl)[C@@H](Cl)C", StereographMirrorStatus.CHIRAL),
    ],
)
@pytest.mark.parametrize("profile", ["chemical", "lewis_state", "acs_topology"])
def test_partial_to_meso_boundary_survives_renumbering(smiles, status, profile):
    molecule = Chem.MolFromSmiles(smiles)
    for order in (list(range(molecule.GetNumAtoms())), list(reversed(range(molecule.GetNumAtoms())))):
        working = Chem.RenumberAtoms(molecule, order)
        result = classify_rdkit_configured_stereograph_mirror(working, identity_profile=profile)
        assert result.status is status
        assert result.is_definitive == (status is not StereographMirrorStatus.INCOMPLETE)
        if status is StereographMirrorStatus.INCOMPLETE:
            assert len(result.incomplete_loci) == 1
            assert result.original is None and result.mirror is None
        elif status is StereographMirrorStatus.ACHIRAL:
            assert result.atom_mirror_isomorphism is not None


def test_source_declared_partial_graph_has_its_own_boundary():
    result = classify_rdkit_configured_stereograph_mirror(
        Chem.MolFromSmiles("C[C@H](Cl)C(Cl)C"), require_complete=False,
    )
    assert result.status is StereographMirrorStatus.CHIRAL
    assert not result.incomplete_loci


@pytest.mark.parametrize("smiles", ["COP(=O)(N)[18O-]", "COP(=[18O])(N)[O-]"])
@pytest.mark.parametrize("profile", ["chemical", "lewis_state", "acs_topology"])
def test_isotope_distinct_phosphate_is_incomplete(smiles, profile):
    molecule = Chem.MolFromSmiles(smiles)
    for working in (molecule, Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))):
        result = classify_rdkit_configured_stereograph_mirror(working, identity_profile=profile)
        assert result.status is StereographMirrorStatus.INCOMPLETE
        assert result.incomplete_loci


@pytest.mark.parametrize("profile", ["chemical", "lewis_state", "acs_topology"])
@pytest.mark.parametrize("tag", ["@", "@@"])
def test_configured_isotope_phosphate_keeps_handedness(profile, tag):
    result = classify_rdkit_configured_stereograph_mirror(
        Chem.MolFromSmiles(f"CO[P{tag}](=O)(N)[18O-]"), identity_profile=profile,
    )
    assert result.status is StereographMirrorStatus.CHIRAL


@pytest.mark.parametrize(
    "profile,status",
    [("chemical", StereographMirrorStatus.ACHIRAL),
     ("lewis_state", StereographMirrorStatus.INCOMPLETE),
     ("acs_topology", StereographMirrorStatus.ACHIRAL)],
)
def test_unlabelled_phosphate_respects_requested_profile(profile, status):
    result = classify_rdkit_configured_stereograph_mirror(
        Chem.MolFromSmiles("COP(=O)(N)[O-]"), identity_profile=profile,
    )
    assert result.status is status


@pytest.mark.parametrize("orientation", [None, 1, -1])
def test_framework_helpers_refuse_before_unsupported_branches(orientation):
    graph = nx.star_graph(4)
    descriptor = FrameworkStereo(
        frozenset(graph), (FrameworkFrame(0, (1, 2, 3, 4), 1),), orientation,
    )
    with pytest.raises(TypeError, match="framework stereo"):
        local_configuration_classes(descriptor)
    with pytest.raises(TypeError, match="framework stereo"):
        enumerate_stereograph_assignments(graph, {"framework": descriptor})
    with pytest.raises(TypeError, match="framework stereo"):
        focal_stereo_evidence(graph, {"framework": descriptor}, "framework")
    # A helper refusal must not disable the supported configured identity.
    if orientation is None:
        with pytest.raises(ValueError, match="orientation must be fixed"):
            canonicalize_configured_stereograph(graph, (descriptor,))
    else:
        assert canonicalize_configured_stereograph(graph, (descriptor,)).complete
