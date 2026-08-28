import pytest

from synkit.Chem.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.hybrid import (
    enumerate_hybrid_distance_mappings,
)


def _mapping_set(result):
    return {tuple(mapping) for mapping in result.mappings}


def test_auto_selects_edit_support_and_preserves_the_labeled_shell():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    expected = enumerate_distance_mappings(
        lgp,
        CD=2,
        binary=True,
        max_bijections=None,
        compute_minimum_cost=False,
    )

    result = enumerate_hybrid_distance_mappings(
        lgp,
        CD=2,
        binary=True,
        max_bijections=None,
    )

    assert result.complete is True
    assert result.backend == "binary_edit_support"
    assert result.scope == "complete_binary_atom_compatible_cd_shell"
    assert _mapping_set(result) == _mapping_set(expected)
    assert result.selected_labeled_mapping_count == 4
    assert result.backend_statistics["selector"]["edit_support_pair_count"] == 4


def test_selector_falls_back_without_changing_symmetry_or_minimum_semantics():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    minimum = enumerate_hybrid_distance_mappings(
        lgp,
        CD="minimal",
        binary=True,
        max_bijections=None,
    )
    quotient = enumerate_hybrid_distance_mappings(
        lgp,
        CD=0,
        binary=True,
        max_bijections=None,
        symmetry_pruning=True,
    )

    assert minimum.backend == "assignment_branch_and_bound"
    assert minimum.minimum_cost == 0
    assert quotient.backend == "assignment_branch_and_bound"
    assert "lex_leaders" in quotient.scope


def test_selector_limit_and_infeasible_lattice_are_auditable():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    fallback = enumerate_hybrid_distance_mappings(
        lgp,
        CD=2,
        binary=True,
        max_bijections=None,
        max_edit_support_pairs=3,
    )
    empty = enumerate_hybrid_distance_mappings(
        lgp,
        CD=1,
        binary=True,
        max_bijections=None,
    )

    assert fallback.backend == "assignment_branch_and_bound"
    assert "exceed selector limit" in (
        fallback.backend_statistics["selector"]["selection_reason"]
    )
    assert empty.backend == "binary_edit_support"
    assert empty.complete is True
    assert empty.status == "no_solutions"


def test_explicit_edit_support_rejects_incompatible_output_requests():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    with pytest.raises(ValueError, match="labeled mappings only"):
        enumerate_hybrid_distance_mappings(
            lgp,
            CD=0,
            backend="edit_support",
            symmetry_pruning=True,
        )
