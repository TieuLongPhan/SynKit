from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.exact.kernel import apply_kernel_solution, Kernel


def test_kernel_apply_solution_combines_fixed_and_uncertain_maps():
    lgp = smiles2lgp("CC>>CC", add_Hs=False)
    kernel = Kernel([1], [1], [6], [6], {0: 0}, lgp, candidate_images=[[1]])

    assert apply_kernel_solution(kernel, {0: 0}) == [0, 1]
