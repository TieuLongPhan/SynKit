import pytest

from Experiment.Synister.all_distance_oracle import literal_sets
from Experiment.Synister.classification_worker import classify, verify_cyclic_group
from Experiment.Synister.mapping_landscape import product_automorphisms, toy_endpoints
from Experiment.Synister.worked_oracle import REACTION
from synkit.Chem.Mapper.identifiability import Endpoint, parse_reaction


def test_toy_counts_have_distinct_indexed_orbit_and_its_units():
    r, p = toy_endpoints()
    maps = literal_sets(r, p)[4]
    result = classify(r, p, maps, 4, group=product_automorphisms(p))
    assert result['complete']
    assert (result['indexed_maps'], result['product_orbits'], result['its_classes']) == (16, 8, 4)
    assert result['its_class_lower_bound'] == result['its_class_upper_bound'] == 4
    assert sum(result['bond_pattern_frequencies'].values()) == 16


def test_worked_chemical_minimum_and_mutable_attributes():
    r, p = parse_reaction(REACTION)
    result = classify(r, p, literal_sets(r, p)[12], 12)
    assert result['complete'] and result['indexed_maps'] == 8 and result['its_classes'] == 2
    assert result['indexed_maps'] == result['group_order']*result['product_orbits']
    r = Endpoint((6, 6), (0, 0), (3, 3), ((0, 1, 2),))
    p = Endpoint((6, 6), (0, 1), (3, 2), ((0, 1, 2),))
    result = classify(r, p, literal_sets(r, p)[0], 0)
    assert result['complete'] and result['group_order'] == 1
    assert result['product_orbits'] == 2 and result['its_classes'] == 1
    assert result['joint_pattern_count'] == 2
    with pytest.raises(ValueError, match='attribute'):
        verify_cyclic_group(p, [(0, 1), (1, 0)])


def test_empty_set_and_exhausted_classification_budget():
    r, p = toy_endpoints()
    empty = classify(r, p, [], 0, seconds=0)
    assert empty['complete'] and empty['its_classes'] == empty['product_orbits'] == 0
    partial = classify(r, p, literal_sets(r, p)[4], 4, seconds=0)
    assert not partial['complete'] and not partial['orbit_partition_complete']
    assert partial['its_classes'] is None and partial['its_class_lower_bound'] == 1
    assert partial['its_class_upper_bound'] == partial['product_orbits'] == 16
    assert partial['bond_pattern_count'] is None


def test_canonical_failure_does_not_create_a_class():
    r, p = toy_endpoints()
    result = classify(r, p, literal_sets(r, p)[4], 4,
                      group=product_automorphisms(p),
                      code_function=lambda *args: (None, 'time_limit'))
    assert result['orbit_partition_complete'] and not result['classification_complete']
    assert result['its_classes'] is None and result['observed_canonical_classes'] == 0
    assert result['its_class_lower_bound'] == 1 and result['its_class_upper_bound'] == 8
    assert result['termination'] == 'canonicalization_incomplete'
    assert len(result['failures']) == 8 and result['bond_pattern_count'] > 0


def test_incomplete_or_duplicate_mapping_set_is_rejected():
    r, p = toy_endpoints()
    maps = sorted(literal_sets(r, p)[4])
    with pytest.raises(ValueError, match='Duplicate'):
        classify(r, p, maps+maps[:1], 4)
    # Even cardinality alone cannot establish closure under the subgroup.
    group = product_automorphisms(p)
    first = maps[0]
    mate = tuple(group[1][j] for j in first)
    incomplete = [first, next(m for m in maps if m not in (first, mate))]
    with pytest.raises(ValueError, match='partitioned'):
        classify(r, p, incomplete, 4, group=group)
    with pytest.raises(ValueError, match='requested CD'):
        classify(r, p, maps, 6)
