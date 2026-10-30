import pytest

from synkit.Chem.Mapper._native.core import map_reaction_core
from synkit.Chem.Mapper._native.hydrogen import map_reaction_hydrogen


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
    ],
)
def test_native_quotient_matches_complete_explicit_classes(
    symmetry_library, reaction, balance, minimum, classes
):
    options = {
        "library_path": symmetry_library,
        "backend": "cpabs",
        "balance": balance,
        "max_mappings": None,
        "time_limit_seconds": 20,
    }
    quotient = map_reaction_hydrogen(reaction, **options)
    explicit = map_reaction_core(reaction, **options)
    assert quotient.complete and explicit.complete, (
        quotient.incomplete_reason,
        explicit.incomplete_reason,
    )
    assert quotient.minimum_cost == explicit.minimum_cost == minimum
    assert len(quotient.reactions) == classes
    assert {(r.class_id, r.distance) for r in quotient.reactions} == {
        (r.class_id, r.distance) for r in explicit.reactions
    }
    stats = quotient.search_result.backend_statistics
    assert (
        stats["hydrogen_quotient"]
        and stats["hydrogen_statistics"]["transport_complete"]
    )


@pytest.mark.parametrize(
    "reaction,reason",
    [
        ("[OH].[H]>>O", "nonneutral_or_radical_hydrogen"),
        ("C=C.[H][H]>>CC", "nonordinary_hydrogen_bond"),
        ("[2H]O>>[2H]O", "isotopic_hydrogen"),
        (r"F/C=C/F>>F/C=C\F", "annotated_stereochemistry"),
    ],
)
def test_unsupported_states_have_explicit_full_native_fallback(
    symmetry_library, reaction, reason
):
    options = {
        "library_path": symmetry_library,
        "backend": "cpabs",
        "max_mappings": None,
        "time_limit_seconds": 20,
    }
    quotient = map_reaction_hydrogen(reaction, **options)
    explicit = map_reaction_core(reaction, **options)
    assert quotient.complete and explicit.complete
    assert quotient.minimum_cost == explicit.minimum_cost
    assert {r.class_id for r in quotient.reactions} == {
        r.class_id for r in explicit.reactions
    }
    assert (
        quotient.search_result.backend_statistics["hydrogen_quotient_fallback"]
        == reason
    )


@pytest.mark.parametrize(
    "target,mode", [(4, "exact"), (4, "at_most"), (1.25, "exact"), (1.25, "at_most")]
)
def test_numeric_shell_keeps_nonminimal_hydrogen_transports(
    symmetry_library, target, mode
):
    options = {
        "library_path": symmetry_library,
        "backend": "cpabs",
        "CD": target,
        "CD_mode": mode,
        "max_mappings": None,
        "time_limit_seconds": 20,
    }
    quotient = map_reaction_hydrogen("CC>>CC", **options)
    explicit = map_reaction_core("CC>>CC", **options)
    assert quotient.complete and explicit.complete, (
        quotient.incomplete_reason,
        explicit.incomplete_reason,
    )
    assert {(r.class_id, r.distance) for r in quotient.reactions} == {
        (r.class_id, r.distance) for r in explicit.reactions
    }
    if target == 4:
        assert quotient.search_result.backend_statistics["hydrogen_quotient"]
        assert any(r.distance == 4 for r in quotient.reactions)


def test_quotient_caps_and_deadlines_preserve_incomplete(symmetry_library):
    for options in (
        {"time_limit_seconds": 0},
        {"max_transport_plans": 1},
        {"max_mappings": 1},
    ):
        result = map_reaction_hydrogen(
            "O=CC=O>>OC#CO", library_path=symmetry_library, backend="cpabs", **options
        )
        assert not result.complete
