from copy import deepcopy

import pytest

from synkit.Graph.Stereo import StereoOutcome
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import StereoBranchLimitError
from synkit.Synthesis.Reactor import SynReactor

SN2 = "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"
CAPTURE = "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("ignore", (1, 1, 1)),
        ("require", (1, 0, 0)),
        ("propagate", (1, 1, 1)),
        ("strict", (1, 0, 0)),
    ],
)
def test_stereo_mode_matrix_for_exact_inverse_and_absent_input(mode, expected):
    substrates = (
        "C[C@H](F)Cl.[OH-]",
        "C[C@@H](F)Cl.[OH-]",
        "CC(F)Cl.[OH-]",
    )

    observed = tuple(
        SynReactor(
            substrate,
            SN2,
            template_format="tuple",
            explicit_h=False,
            stereo_mode=mode,
        ).mapping_count
        for substrate in substrates
    )

    assert observed == expected


def test_unknown_propagation_remains_one_unknown_not_a_population():
    reactor = SynReactor(
        "CC(F)Cl.[OH-]",
        SN2,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="propagate",
    )

    assert len(reactor.its_list) == 1
    assert reactor.its_list[0].graph["stereo_outcomes"] == {}
    assert (
        reactor.its_list[0].graph["stereo_descriptors"]["product"]["atom:2"].parity
        is None
    )


def test_zero_weight_outcome_branch_is_not_materialized():
    rule = SynRule.from_smart(
        CAPTURE,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"atom:2": StereoOutcome.enantiomeric_mixture(1.0, 0.0)},
    )
    reactor = SynReactor(
        "C[CH+]F.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )

    assert len(reactor.its_list) == 1
    result = reactor.its_list[0]
    assert result.graph["stereo_branch_weight"] == 1.0
    assert result.graph["stereo_branch_path"] == (("atom:2", 0),)


def test_branch_limit_fails_typed_and_without_a_partial_result():
    rule = SynRule.from_smart(
        CAPTURE,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"atom:2": StereoOutcome.racemic()},
    )
    reactor = SynReactor(
        "C[CH+]F.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        stereo_branch_limit=1,
    )

    with pytest.raises(StereoBranchLimitError) as captured:
        _ = reactor.its_list

    assert captured.value.permitted == captured.value.limit == 1
    assert captured.value.requested == captured.value.discovered == 2
    assert reactor._its == []


def test_dedup_accumulates_measure_multiplicity_and_branch_provenance():
    first = rsmi_to_its(SN2, format="tuple")
    second = deepcopy(first)
    first.graph.update(
        stereo_branch_weight=0.7,
        stereo_aggregate_weight=0.7,
        stereo_branch_multiplicity=1,
        stereo_branch_path=(("atom:2", 0),),
    )
    second.graph.update(
        stereo_branch_weight=0.3,
        stereo_aggregate_weight=0.3,
        stereo_branch_multiplicity=1,
        stereo_branch_path=(("atom:2", 1),),
    )

    results = SynReactor._deduplicate_structural_its([first, second])

    assert len(results) == 1
    representative = results[0]
    assert representative.graph["stereo_aggregate_weight"] == pytest.approx(1.0)
    assert representative.graph["stereo_branch_multiplicity"] == 2
    assert [
        contribution["branch_path"]
        for contribution in representative.graph["stereo_branch_contributions"]
    ] == [(("atom:2", 0),), (("atom:2", 1),)]


def test_branch_order_is_repeat_run_deterministic():
    rule = SynRule.from_smart(
        CAPTURE,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"atom:2": StereoOutcome.enantiomeric_mixture(0.7, 0.3)},
    )

    paths = []
    products = []
    for _ in range(3):
        reactor = SynReactor(
            "C[CH+]F.[OH-]",
            rule,
            template_format="tuple",
            explicit_h=False,
            stereo_mode="strict",
            dedup_its=False,
        )
        paths.append(
            [result.graph["stereo_branch_path"] for result in reactor.its_list]
        )
        products.append(reactor.smarts_list)

    assert paths[0] == paths[1] == paths[2]
    assert products[0] == products[1] == products[2]
