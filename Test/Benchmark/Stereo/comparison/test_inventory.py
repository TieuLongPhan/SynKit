"""Mirror comparison respects the exact frozen inventory in the diagnostic."""

import pytest

from Experiment.Stereo.reference.fixtures import fixtures
from Experiment.Stereo.reference.model import to_descriptor
from Experiment.Stereo.comparison.inventory import _mirror_equal


@pytest.mark.parametrize("name,equal", [
    ("tetrahedral:distinct", False), ("tetrahedral:symmetric", True), ("planar_bond:distinct", True),
])
def test_matched_inventory_mirror_equality_has_the_declared_semantics(name, equal):
    fixture = next(value for value in fixtures() if value.name == name)
    assert _mirror_equal(fixture.graph, {"locus": to_descriptor(fixture.seeds[0])}) is equal
