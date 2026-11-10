"""Scientific scoring boundaries: whole molecules, extra outputs and candidates."""

from Experiment.Lewis.partial_replay.benchmark import canonical_fragments
from Experiment.Lewis.partial_replay.fragment_metrics import score_candidate, score_outputs


def score(actual, target):
    return score_candidate(canonical_fragments(actual), canonical_fragments(target))


def test_unreported_byproduct_does_not_hide_the_recorded_product():
    result = score("CO.[Br-]", "CO")
    assert not result["exact_set"]
    assert result["all_gt_molecules"]
    assert result["largest_gt_recovered"]
    assert result["max_fragment_match"]


def test_largest_gt_anywhere_differs_from_largest_output_equality():
    result = score("CO.CCCCC", "CO")
    assert result["largest_gt_recovered"]
    assert not result["max_fragment_match"]


def test_substructure_or_wildcard_is_not_exact_molecule_recovery():
    assert not score("CCO", "CO")["any_gt_molecule"]
    assert not score("*CO", "CCO")["any_gt_molecule"]
    assert not score("C[O-]", "CO")["any_gt_molecule"]
    assert not score("[13CH3]O", "CO")["any_gt_molecule"]


def test_spectator_match_does_not_establish_largest_or_all_gt_recovery():
    result = score("O.CN", "CCCCC.O")
    assert result["any_gt_molecule"]
    assert not result["all_gt_molecules"]
    assert not result["largest_gt_recovered"]


def test_component_counts_and_single_candidate_witness_are_required():
    assert not score("C", "C.C")["all_gt_molecules"]
    result = score_outputs([canonical_fragments("CO"), canonical_fragments("CN")],
                           canonical_fragments("CO.CN"))
    assert not result["all_gt_molecules"]
    assert result["gt_molecule_recall"] == 0.5


def test_largest_fragment_ties_and_component_order_do_not_change_score():
    assert score("CN.O", "CO.CN")["max_fragment_match"]
    assert score("O.CN", "CN.CO")["max_fragment_match"]
