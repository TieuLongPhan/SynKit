from synkit.Chem.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import (
    chemical_distance,
    dual_lap_lower_bound,
    solve_lap,
)


def test_lap_helpers_solve_assignment_and_distance():
    _, _, value = solve_lap([[0.0, 2.0], [2.0, 0.0]])
    lgp = smiles2lgp("CC>>CC", add_Hs=False)

    assert value == 0.0
    assert chemical_distance(lgp, [0, 1], binary=True) == 0.0


def test_dual_lap_reports_incompatible_element_multiplicities():
    reactant = LabeledGraph({0: {}, 1: {}, 2: {}}, [1, 1, 2])
    product = LabeledGraph({0: {}, 1: {}, 2: {}}, [1, 2, 2])

    assert dual_lap_lower_bound([reactant, product], binary=True) == float("inf")
