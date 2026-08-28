from synkit.Chem.Mapper.exact.symmetry import (
    orbital_candidate_witnesses,
    permutation_group_order,
    point_stabilizer_generators,
    point_stabilizer_generators_checked,
)


def test_schreier_stabilizer_and_orbital_witnesses():
    rotation = (1, 2, 3, 0)
    reflection = (0, 3, 2, 1)

    stabilizer = point_stabilizer_generators((rotation, reflection), 0)
    assert stabilizer == (reflection,)

    witnesses = orbital_candidate_witnesses((1, 2, 3), stabilizer)
    assert witnesses[1] is None
    assert witnesses[2] is None
    assert witnesses[3][3] == 1
    assert witnesses[3][0] == 0


def test_stabilizer_budget_exit_only_disables_deeper_pruning():
    rotation = (1, 2, 3, 0)
    reflection = (0, 3, 2, 1)

    assert (
        point_stabilizer_generators(
            (rotation, reflection),
            0,
            max_generators=0,
        )
        == ()
    )
    assert orbital_candidate_witnesses((1, 2, 3), ()) == {
        1: None,
        2: None,
        3: None,
    }
    assert point_stabilizer_generators_checked(
        (rotation, reflection),
        0,
        max_generators=0,
    ) == ((), False)


def test_group_order_uses_stabilizer_chain_without_element_expansion():
    rotation = (1, 2, 3, 0)
    reflection = (0, 3, 2, 1)

    assert permutation_group_order((rotation,)) == 4
    assert permutation_group_order((rotation, reflection)) == 8
    assert permutation_group_order(()) == 1
    assert (
        permutation_group_order(
            (rotation, reflection),
            max_generators=0,
        )
        is None
    )
