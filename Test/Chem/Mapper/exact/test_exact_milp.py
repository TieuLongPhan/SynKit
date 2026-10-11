import pytest
from synkit.Chem.Mapper.exact.branching import solve_kernel
from synkit.Chem.Mapper.exact.kernel import Kernel, apply_kernel_solution
from synkit.Chem.Mapper.exact.milp import HAS_PULP, solve_kernel_milp, solve_qap

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.mark.skipif(not HAS_PULP, reason="PuLP is not installed")
def test_milp_solve_qap_identity_assignment():
    cost, mapping, proven = solve_qap({(0, 0): 0.0}, {}, 1, {(0, 0)})

    assert cost == 0.0
    assert mapping == {0: 0}
    assert proven


@pytest.mark.skipif(not HAS_PULP, reason="PuLP is not installed")
def test_milp_respects_restricted_candidate_domains():
    graphs = [LabeledGraph({0: {1: 1}, 1: {0: 1}, 2: {}}, [6, 6, 6]) for _ in range(2)]
    kernel = Kernel(
        r_idx=[0, 1, 2],
        p_idx=[0, 1, 2],
        r_colors=[6, 6, 6],
        p_colors=[6, 6, 6],
        fixed_mapping={},
        lgp=graphs,
        candidate_images=[[2], [1], [0]],
    )
    # Unrestricted identity has cost zero, but the supplied kernel forces two edits.
    result = solve_kernel_milp(kernel)
    reference = solve_kernel(kernel)
    assert result.proven_optimal and result.cost == reference.cost == 2
    assert apply_kernel_solution(kernel, result.sub_mappings[0]) == [2, 1, 0]
