"""Mapped participating hydrogens must survive mechanism comparison."""

import pytest

from synkit.Mechanism import mechanism_equivalent, mechanism_from_legacy_epd


@pytest.mark.parametrize("level", ["net", "events", "trajectory"])
def test_mapped_hydrogen_equivalence_survives_atom_map_renumbering(level):
    reference = mechanism_from_legacy_epd(
        "[NH3+:2][H:4].[O-:1][CH3:3]>>[NH3:2].[O:1]([CH3:3])[H:4]",
        [["LP-/Sigma+", [1], [1, 4]], ["Sigma-/LP+", [2, 4], [2]]],
    )
    prediction = mechanism_from_legacy_epd(
        "[NH3+:12][H:14].[O-:11][CH3:13]>>[NH3:12].[O:11]([CH3:13])[H:14]",
        [["LP-/Sigma+", [11], [11, 14]], ["Sigma-/LP+", [12, 14], [12]]],
    )
    assert mechanism_equivalent(reference, prediction, level=level)
