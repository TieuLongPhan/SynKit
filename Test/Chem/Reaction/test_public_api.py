from synkit.Chem import utils
from synkit.Chem.Reaction import remove_explicit_H_from_rsmi


def test_reaction_reexports_remove_explicit_h_from_rsmi():
    assert remove_explicit_H_from_rsmi is utils.remove_explicit_H_from_rsmi
