import pytest

from Experiment.Synister.all_distance_oracle import binary_endpoint, literal_sets
from Experiment.Synister.mapping_landscape import (
    analyze, fingerprint, partition_orbits, product_automorphisms, toy_endpoints,
)
from Experiment.Synister.structural_oracle import independent_its
from synkit.Chem.Mapper.identifiability import Endpoint


def test_complete_toy_landscape_separates_maps_orbits_and_its():
    r, p = toy_endpoints()
    record = analyze("toy", r, p)
    assert record["compatible_maps"] == 120 and record["product_group_order"] == 2
    assert record["all_classifications_complete"]
    minimum = record["rows"][record["minimum_doubled_cd"]]
    assert (minimum["doubled_cd"], minimum["indexed_maps"], minimum["product_orbits"], minimum["its_classes"]) == (4, 16, 8, 4)
    assert sum(row["indexed_maps"] for row in record["rows"]) == 120
    assert all(row["indexed_maps"] == 2*row["product_orbits"] for row in record["rows"])
    assert all(row["its_classes"] == row["independent_its_classes"] for row in record["rows"])
    assert any(row["indexed_maps"] == 0 for row in record["rows"])


def test_product_group_respects_mutable_endpoint_attributes():
    r = Endpoint((6, 6), (0, 0), (3, 3), ((0, 1, 2),))
    p = Endpoint((6, 6), (0, 1), (3, 2), ((0, 1, 2),))
    assert len(product_automorphisms(r)) == 2
    assert len(product_automorphisms(p)) == 1
    assert sum(map(len, literal_sets(r, p).values())) == 2
    record = analyze("unary", r, p)
    assert record["rows"][0]["indexed_maps"] == record["rows"][0]["product_orbits"] == 2
    assert record["rows"][0]["its_classes"] == 1


def test_group_cap_and_incomplete_orbit_are_not_complete_results():
    endpoint = binary_endpoint(0)
    with pytest.raises(ValueError, match="exceeds"):
        product_automorphisms(endpoint, limit=2)
    with pytest.raises(ValueError, match="preserve"):
        partition_orbits({(0, 1)}, [(0, 1), (1, 0)])


def test_unfinished_canonicalization_is_recorded_not_counted_as_a_class():
    r, p = toy_endpoints()
    record = analyze("interrupted", r, p, code_function=lambda *args, **kwargs: (None, None, "time_limit"))
    assert not record["all_classifications_complete"]
    assert all(row["its_classes"] is None for row in record["rows"] if row["indexed_maps"])


def test_invalid_canonical_merging_is_detected_by_independent_partition():
    r, p = toy_endpoints()
    with pytest.raises(ValueError, match="disagree"):
        analyze("bad", r, p, code_function=lambda *args, **kwargs: (("one_class",), (), None))


def test_fingerprint_does_not_depend_on_vertex_numbers():
    import networkx as nx
    r, p = toy_endpoints()
    graph = independent_its(r, p, (0, 1, 2, 3, 4))
    changed = nx.relabel_nodes(graph, {i: 4-i for i in graph})
    assert fingerprint(graph) == fingerprint(changed)
