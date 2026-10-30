"""Check the exact H-leaf reduction against full explicit-H minimum classes."""

from itertools import permutations

import pytest

from synkit.Chem.Mapper import map_reaction
from synkit.Chem.Mapper.exact.native.hydrogen_quotient import (
    map_reaction_hydrogen_quotient,
)
from synkit.Chem.Mapper.exact.hydrogen import enumerate_minimal_hydrogen_transfers


@pytest.mark.parametrize(
    "reaction,balance,minimum,classes",
    [
        ("CO>>CO", "strict", 0, 1),
        ("N.Cl>>[NH4+].[Cl-]", "strict", 4, 1),
        ("O=CC=O>>OC#CO", "strict", 8, 2),
        ("O>>[OH-]", "dummy", 2, 1),
        ("[OH-]>>O", "dummy", 2, 1),
        ("CO>>C", "dummy", 5, 1),
        ("C>>CO", "dummy", 5, 1),
        ("N.CN.Cl.Br>>[NH4+].C[NH3+].[Cl-].[Br-]", "strict", 8, 2),
    ],
)
def test_quotient_matches_all_explicit_minimum_classes(
    native_library, reaction, balance, minimum, classes
):
    options = dict(library_path=native_library, balance=balance, max_mappings=None)
    explicit = map_reaction(reaction, backend="cpp", **options)
    quotient = map_reaction_hydrogen_quotient(reaction, **options)
    assert explicit.complete and quotient.complete
    assert explicit.minimum_cost == quotient.minimum_cost == minimum
    assert len(quotient.reactions) == classes
    assert {(item.class_id, item.distance) for item in explicit.reactions} == {
        (item.class_id, item.distance) for item in quotient.reactions
    }
    stats = quotient.search_result.backend_statistics
    assert stats["hydrogen_quotient"]
    assert stats["quotient_nodes"] < stats["full_nodes"]


@pytest.mark.parametrize(
    "reaction,expected",
    [
        ("[OH].[H]>>O", "nonneutral_or_radical_hydrogen"),
        ("C=C.[H][H]>>CC", "nonordinary_hydrogen_bond"),
        ("[2H]O>>[2H]O", "isotopic_hydrogen"),
        ("F/C=C/F>>F/C=C\\F", "annotated_stereochemistry"),
    ],
)
def test_unsupported_hydrogen_and_stereo_use_exact_full_solver(
    native_library, reaction, expected
):
    options = dict(library_path=native_library, max_mappings=None)
    explicit = map_reaction(reaction, backend="cpp", **options)
    quotient = map_reaction_hydrogen_quotient(reaction, **options)
    assert explicit.complete and quotient.complete
    assert quotient.minimum_cost == explicit.minimum_cost
    assert {item.class_id for item in quotient.reactions} == {
        item.class_id for item in explicit.reactions
    }
    assert quotient.search_result.backend_statistics["hydrogen_quotient_fallback"] == expected


def test_numeric_shell_includes_nonminimal_hydrogen_transports(native_library):
    # Ethane permits H exchanges costing4 although the minimum H cost is0.
    options = dict(library_path=native_library, CD=4, max_mappings=None)
    explicit = map_reaction("CC>>CC", backend="cpp", **options)
    quotient = map_reaction_hydrogen_quotient("CC>>CC", **options)
    assert explicit.complete and quotient.complete and quotient.reactions
    assert {item.class_id for item in quotient.reactions} == {
        item.class_id for item in explicit.reactions
    }
    assert quotient.search_result.backend_statistics["hydrogen_quotient_fallback"] == "numeric_distance_query"


def test_interrupted_quotient_never_claims_complete_minimum(native_library):
    result = map_reaction_hydrogen_quotient(
        "O=CC=O>>OC#CO", library_path=native_library, time_limit_seconds=0
    )
    assert not result.complete and result.minimum_cost is None
    limited = map_reaction_hydrogen_quotient(
        "O=CC=O>>OC#CO", library_path=native_library, max_hydrogen_plans=1
    )
    assert limited.minimum_cost == 8
    assert not limited.complete and "plan_limit" in limited.incomplete_reason


def test_cap_counts_lifted_candidates_not_only_heavy_mappings(native_library):
    result = map_reaction_hydrogen_quotient(
        "O=CC=O>>OC#CO", library_path=native_library, max_mappings=1
    )
    assert result.minimum_cost == 8 and not result.complete
    assert len(result.reactions) == 1
    assert result.incomplete_reason == "mapping_limit"


@pytest.mark.parametrize(
    "left,right",
    [([2, 2, 0], [0, 0, 1]), ([0, 0, 1], [2, 2, 0]),
     ([2, 1, 0], [0, 1, 2]), ([1, 1, 0], [0, 0, 2]),
     ([0, 0, 0], [1, 2, 0]), ([1, 2, 0], [0, 0, 0])],
)
def test_dummy_hydrogen_transport_orbits_match_independent_permutations(left, right):
    size = len(left)
    counts_r = left + [max(0, sum(right) - sum(left))]
    counts_p = right + [max(0, sum(left) - sum(right))]
    parents_r = [i for i, count in enumerate(counts_r) for _ in range(count)]
    parents_p = [i for i, count in enumerate(counts_p) for _ in range(count)]
    optimum = sum(abs(a - b) for a, b in zip(left, right))
    expected = set()
    for mapping in permutations(range(len(parents_p))):
        cost = 0
        cells = [0] * (size + 1) ** 2
        for atom, image in enumerate(mapping):
            source, target = parents_r[atom], parents_p[image]
            cells[source * (size + 1) + target] += 1
            if source != target:
                cost += int(source < size) + int(target < size)
        if cost == optimum:
            expected.add(tuple(cells))
    plans = enumerate_minimal_hydrogen_transfers(
        counts_r, counts_p, range(size + 1)
    )
    assert plans.complete
    actual = set()
    for plan in plans.plans:
        cells = [0] * (size + 1) ** 2
        for i, count in enumerate(plan.preserved):
            cells[i * (size + 1) + i] = count
        for source, target, count in plan.transfers:
            cells[source * (size + 1) + target] = count
        actual.add(tuple(cells))
    assert actual == expected
