from Experiment.Synister.all_distance_oracle import binary_endpoint
from Experiment.Synister.structural_oracle import check_case, independent_its, isomorphic
from synkit.Chem.Mapper.identifiability import Endpoint


def test_all_empty_graph_maps_share_one_class():
    endpoint = binary_endpoint(0)
    report = check_case("empty", endpoint, endpoint)
    assert report["passed"]
    assert report["distances"][0]["maps"] == 24
    assert report["distances"][0]["independent_classes"] == 1
    assert report["distances"][0]["isomorphic_pairs"] == 276


def test_unary_changes_are_not_erased_by_equal_bond_changes():
    endpoint = Endpoint((6, 6), (0, 1), (4, 3), ())
    first = independent_its(endpoint, endpoint, (0, 1))
    second = independent_its(endpoint, endpoint, (1, 0))
    assert not isomorphic(first, second)
    assert check_case("unary", endpoint, endpoint)["distances"][0]["independent_classes"] == 2
    wrong = lambda *args, **kwargs: ("same", "same", None)
    assert not check_case("wrong", endpoint, endpoint, code_function=wrong)["passed"]


def test_incomplete_classification_never_passes():
    endpoint = binary_endpoint(0)
    incomplete = lambda *args, **kwargs: (None, None, "timeout")
    result = check_case("incomplete", endpoint, endpoint, code_function=incomplete)
    assert not result["passed"]
    assert result["distances"][0]["unfinished"] == 24
