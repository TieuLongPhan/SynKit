from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.exact.branching import solve_kernel
from synkit.Chem.Reaction.Mapper.exact.kernel import Kernel


def test_branching_solves_trivial_kernel():
    lgp = smiles2lgp("C>>C", add_Hs=False)
    kernel = Kernel([], [], [], [], {0: 0}, lgp, binary=True, candidate_images=[])

    solution = solve_kernel(kernel)

    assert solution.cost == 0.0
    assert solution.sub_mappings == [{}]
