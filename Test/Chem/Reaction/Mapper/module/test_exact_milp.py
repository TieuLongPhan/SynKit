import pytest

from synkit.Chem.Reaction.Mapper.exact.milp import HAS_PULP, solve_qap


@pytest.mark.skipif(not HAS_PULP, reason="PuLP is not installed")
def test_milp_solve_qap_identity_assignment():
    cost, mapping, proven = solve_qap({(0, 0): 0.0}, {}, 1, {(0, 0)})

    assert cost == 0.0
    assert mapping == {0: 0}
    assert proven
