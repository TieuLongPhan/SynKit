"""Coordinate consensus must be exhaustive and preserve constitutional colors."""

import pytest
from rdkit import Chem

from Experiment.Stereo.Perception import cip_labels


def test_transport_refuses_one_map_over_budget(monkeypatch):
    monkeypatch.setattr(cip_labels, "_MAX_TRANSPORT_MATCHES", 1)
    molecule = Chem.MolFromSmiles("CC")
    with pytest.raises(cip_labels.CoordinateTransportLimitError) as failure:
        cip_labels._constitutional_matches(molecule, molecule)
    assert failure.value.status == "unresolved_coordinate_transport"
    assert failure.value.limit == 1


def test_exactly_at_budget_is_exhaustive(monkeypatch):
    monkeypatch.setattr(cip_labels, "_MAX_TRANSPORT_MATCHES", 2)
    molecule = Chem.MolFromSmiles("CC")
    assert len(cip_labels._constitutional_matches(molecule, molecule)) == 2


@pytest.mark.parametrize("target", ["[13CH4]", "[CH3]", "[CH3-]"])
def test_substructure_match_is_not_full_constitutional_identity(target):
    assert cip_labels._constitutional_matches(
        Chem.MolFromSmiles("C"), Chem.MolFromSmiles(target),
    ) == ()
