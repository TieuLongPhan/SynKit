from dataclasses import replace
import itertools
import json

import pytest

from synkit.Chem.Mapper import AAMapper
from synkit.Chem.Mapper.exact import distance as distance_module
from synkit.Chem.Mapper.exact.distance import (
    CertificateVerificationError,
    ExactEnumerationLimitError,
    enumerate_distance_mappings,
    verify_distance_enumeration_certificate,
)
from synkit.Chem.Mapper.exact.branching import solve_kernel
from synkit.Chem.Mapper.exact.exhaustive import ExactMapper
from synkit.Chem.Mapper.exact.kernel import Kernel
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def _cycle_graph(size, permutation=None):
    permutation = list(range(size)) if permutation is None else permutation
    graph = {index: {} for index in range(size)}
    for source in range(size):
        target = (source + 1) % size
        left = permutation[source]
        right = permutation[target]
        graph[left][right] = 1
        graph[right][left] = 1
    return LabeledGraph(graph, [6] * size)


def test_enumerate_smiles_global_minimum_and_exact_shell():
    mapper = AAMapper(binary=True)

    minimum = mapper.enumerate_smiles(
        "CCC>>CCC", CD="minimal", add_Hs=False, unique=False, certify=True
    )
    assert minimum.complete is True
    assert minimum.status == "complete"
    assert minimum.minimum_cost == 0
    assert len(minimum.mappings) == 2
    assert {result["cd"] for result in mapper.results} == {0}
    assert minimum.certificate is not None
    lgp = mapper.results[0]["lgp"]
    # Result LGPs carry resolved labels but retain the original graph and atom
    # types, so they are a valid independent replay input.
    assert verify_distance_enumeration_certificate(
        lgp,
        json.loads(json.dumps(minimum.certificate.as_dict())),
        mappings=minimum.mappings,
    )

    shell = mapper.enumerate_smiles("CCC>>CCC", CD=2, add_Hs=False, unique=False)
    assert shell.complete is True
    assert shell.status == "complete"
    assert shell.minimum_cost == 0
    assert len(shell.mappings) == 4
    assert {result["cd"] for result in mapper.results} == {2}


def test_completed_empty_shell_is_distinct_from_timeout():
    mapper = AAMapper(binary=True)

    empty = mapper.enumerate_smiles(
        "CCC>>CCC", CD=2468, add_Hs=False, unique=False, certify=True
    )
    assert empty.complete is True
    assert empty.status == "no_solutions"
    assert empty.mappings == []
    assert empty.visited_leaves == 0
    assert empty.maximum_cost_upper_bound < 2468
    assert empty.certificate is not None
    from synkit.Chem.Mapper.chem.smiles import smiles2lgp

    assert verify_distance_enumeration_certificate(
        smiles2lgp("CCC>>CCC", add_Hs=False), empty.certificate
    )
    assert mapper.results == []

    timed_out = mapper.enumerate_smiles(
        "CCC>>CCC",
        CD=2,
        add_Hs=False,
        unique=False,
        time_limit_seconds=0,
        certify=True,
    )
    assert timed_out.complete is False
    assert timed_out.status == "timeout"
    assert timed_out.truncation_reason == "time_limit"
    assert timed_out.certificate is not None
    assert timed_out.certificate.frontier_prefixes
    assert verify_distance_enumeration_certificate(
        smiles2lgp("CCC>>CCC", add_Hs=False), timed_out.certificate
    )
    assert mapper.results == []

    minimal_timeout = mapper.enumerate_smiles(
        "CCC>>CCC",
        CD="minimal",
        add_Hs=False,
        unique=False,
        time_limit_seconds=0,
        certify=True,
    )
    assert minimal_timeout.status == "timeout"
    assert minimal_timeout.minimum_cost is None
    assert minimal_timeout.mappings == []
    assert verify_distance_enumeration_certificate(
        smiles2lgp("CCC>>CCC", add_Hs=False), minimal_timeout.certificate
    )


def test_certificate_tampering_fails_closed():
    mapper = AAMapper(binary=True)
    result = mapper.enumerate_smiles(
        "CCC>>CCC", CD=1, add_Hs=False, unique=False, certify=True
    )
    assert result.status == "no_solutions"
    assert result.certificate is not None

    from synkit.Chem.Mapper.chem.smiles import smiles2lgp

    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    assert verify_distance_enumeration_certificate(lgp, result.certificate)
    tampered = replace(result.certificate, selected_mapping_count=1)
    with pytest.raises(CertificateVerificationError, match="digest mismatch"):
        verify_distance_enumeration_certificate(lgp, tampered)

    prefix_field = next(
        field
        for field in (
            "terminal_prefixes",
            "lower_bound_prefixes",
            "upper_bound_prefixes",
            "symmetry_prefixes",
        )
        if getattr(result.certificate, field)
    )
    structurally_invalid = replace(result.certificate, certificate_sha256="")
    structurally_invalid = replace(
        structurally_invalid,
        **{prefix_field: getattr(structurally_invalid, prefix_field)[:-1]},
    )
    structurally_invalid = replace(
        structurally_invalid,
        certificate_sha256=distance_module._certificate_sha256(
            structurally_invalid.as_dict()
        ),
    )
    with pytest.raises(CertificateVerificationError, match="do not cover tree"):
        verify_distance_enumeration_certificate(lgp, structurally_invalid)


def test_exact_enumeration_cap_fails_explicitly():
    mapper = AAMapper(binary=True)
    with pytest.raises(ExactEnumerationLimitError, match="6 atom-compatible"):
        mapper.enumerate_smiles(
            "CCC>>CCC",
            CD="minimal",
            add_Hs=False,
            max_bijections=5,
        )


def test_fixed_assignment_subspace_matches_filtered_full_shell():
    from synkit.Chem.Mapper.chem.smiles import smiles2lgp

    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    full = enumerate_distance_mappings(lgp, CD=2, binary=True, max_bijections=None)
    fixed = enumerate_distance_mappings(
        lgp,
        CD=2,
        binary=True,
        max_bijections=None,
        fixed_mapping={0: 0},
    )

    assert {tuple(mapping) for mapping in fixed.mappings} == {
        tuple(mapping) for mapping in full.mappings if mapping[0] == 0
    }
    assert fixed.total_bijections == 2
    assert fixed.scope == "fixed_assignment_subspace"


def test_fixed_context_uses_product_point_stabilizer_for_symmetry():
    permutation = [0, 2, 4, 1, 3]
    lgp = [_cycle_graph(5), _cycle_graph(5, permutation)]
    labeled = enumerate_distance_mappings(
        lgp,
        CD=0,
        binary=True,
        max_bijections=None,
        fixed_mapping={0: 0},
    )
    quotient = enumerate_distance_mappings(
        lgp,
        CD=0,
        binary=True,
        max_bijections=None,
        fixed_mapping={0: 0},
        symmetry_pruning=True,
    )

    assert len(labeled.mappings) == 2
    assert len(quotient.mappings) == 1
    assert quotient.symmetry_group_order == 2
    assert quotient.symmetry_quotient_complete is True
    assert quotient.selected_labeled_mapping_count == len(labeled.mappings)
    assert quotient.scope == (
        "verified_product_automorphism_lex_leaders_within_" "fixed_assignment_subspace"
    )


def test_fixed_assignment_validation_fails_closed():
    graph = _cycle_graph(5)
    lgp = [graph, graph.copy()]

    with pytest.raises(ValueError, match="product images must be unique"):
        enumerate_distance_mappings(lgp, fixed_mapping={0: 0, 1: 0})
    with pytest.raises(ValueError, match="certificates do not yet support"):
        enumerate_distance_mappings(lgp, fixed_mapping={0: 0}, certify=True)
    with pytest.raises(ValueError, match="expansion does not yet support"):
        enumerate_distance_mappings(
            lgp,
            fixed_mapping={0: 0},
            symmetry_pruning=True,
            expand_symmetry=True,
        )


def test_product_symmetry_pruning_enumerates_verified_lex_leaders():
    mapper = AAMapper(binary=True)
    full = mapper.enumerate_smiles(
        "CCC>>CCC",
        CD=2,
        add_Hs=False,
        unique=False,
        certify=True,
    )
    quotient = mapper.enumerate_smiles(
        "CCC>>CCC",
        CD=2,
        add_Hs=False,
        unique=False,
        certify=True,
        symmetry_pruning=True,
    )

    assert len(full.mappings) == 4
    assert len(quotient.mappings) == 2
    assert quotient.complete is True
    assert quotient.scope == "verified_product_automorphism_lex_leaders"
    assert quotient.symmetry_automorphism_count == 2
    assert quotient.symmetry_group_order == 2
    assert quotient.symmetry_quotient_complete is True
    assert quotient.selected_mapping_count == 2
    assert quotient.selected_labeled_mapping_count == 4
    assert quotient.symmetry_pruned_branches > 0
    assert quotient.visited_nodes < full.visited_nodes
    assert quotient.certificate is not None
    assert quotient.certificate.symmetry_prefixes

    from synkit.Chem.Mapper.chem.smiles import smiles2lgp

    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    assert verify_distance_enumeration_certificate(
        lgp,
        json.loads(json.dumps(quotient.certificate.as_dict())),
        mappings=quotient.mappings,
    )

    legacy_scope = replace(
        quotient.certificate,
        mapping_scope="product_automorphism_orbit_representatives",
        certificate_sha256="",
    )
    legacy_scope = replace(
        legacy_scope,
        certificate_sha256=distance_module._certificate_sha256(legacy_scope.as_dict()),
    )
    assert verify_distance_enumeration_certificate(
        lgp, legacy_scope, mappings=quotient.mappings
    )


def test_symmetry_timeout_certificate_retains_frontier():
    mapper = AAMapper(binary=True)
    result = mapper.enumerate_smiles(
        "CCCC>>CCCC",
        CD=2,
        add_Hs=False,
        unique=False,
        certify=True,
        symmetry_pruning=True,
        max_bijections=None,
        time_limit_seconds=0,
    )
    assert result.status == "timeout"
    assert result.certificate is not None
    assert result.certificate.frontier_prefixes

    from synkit.Chem.Mapper.chem.smiles import smiles2lgp

    assert verify_distance_enumeration_certificate(
        smiles2lgp("CCCC>>CCCC", add_Hs=False),
        result.certificate,
    )


def test_assignment_lower_bound_is_replayable():
    mapper = AAMapper(binary=True)
    result = mapper.enumerate_smiles(
        "CCCC>>CCCC",
        CD=1,
        add_Hs=False,
        unique=False,
        certify=True,
        symmetry_pruning=True,
    )
    assert result.status == "no_solutions"
    assert result.lower_bound_pruned_branches > 0
    assert result.certificate is not None
    assert result.certificate.lower_bound_prefixes

    from synkit.Chem.Mapper.chem.smiles import smiles2lgp

    assert verify_distance_enumeration_certificate(
        smiles2lgp("CCCC>>CCCC", add_Hs=False),
        result.certificate,
    )


def test_relabelled_cycle_does_not_lose_optimum_to_false_symmetry():
    permutation = [0, 2, 4, 1, 3]
    lgp = [_cycle_graph(5), _cycle_graph(5, permutation)]
    kernel = Kernel(
        r_idx=list(range(5)),
        p_idx=list(range(5)),
        r_colors=[6] * 5,
        p_colors=[6] * 5,
        fixed_mapping={},
        lgp=lgp,
        binary=True,
        candidate_images=[list(range(5)) for _ in range(5)],
    )

    assert chemical_distance(lgp, permutation, binary=True) == 0
    assert ExactMapper().solve(*lgp).cost == 0
    assert solve_kernel(kernel, enumerate_all=True).cost == 0

    labeled = enumerate_distance_mappings(
        lgp, CD="minimal", binary=True, max_bijections=None
    )
    quotient = enumerate_distance_mappings(
        lgp,
        CD="minimal",
        binary=True,
        max_bijections=None,
        symmetry_pruning=True,
    )
    expanded = enumerate_distance_mappings(
        lgp,
        CD="minimal",
        binary=True,
        max_bijections=None,
        symmetry_pruning=True,
        expand_symmetry=True,
    )
    assert labeled.minimum_cost == 0
    assert len(labeled.mappings) == 10
    assert quotient.minimum_cost == 0
    assert len(quotient.mappings) == 1
    assert expanded.scope == "complete_atom_compatible_assignment_space"
    assert {tuple(mapping) for mapping in expanded.mappings} == {
        tuple(mapping) for mapping in labeled.mappings
    }
    assert quotient.symmetry_group_order == 10
    assert quotient.selected_labeled_mapping_count == len(labeled.mappings)
    assert expanded.selected_labeled_mapping_count == len(labeled.mappings)

    with pytest.raises(ValueError, match="requires symmetry_pruning"):
        enumerate_distance_mappings(lgp, expand_symmetry=True)
    with pytest.raises(ValueError, match="not supported by certificates"):
        enumerate_distance_mappings(
            lgp,
            symmetry_pruning=True,
            expand_symmetry=True,
            certify=True,
        )


def test_symmetry_is_checked_against_exact_directed_objective():
    graph = LabeledGraph({0: {}, 1: {0: 1}, 2: {}}, [6, 6, 6])
    lgp = [graph, graph.copy()]

    result = enumerate_distance_mappings(
        lgp,
        CD="minimal",
        binary=True,
        max_bijections=None,
        symmetry_pruning=True,
        certify=True,
    )

    assert result.minimum_cost == 0
    assert [0, 1, 2] in result.mappings
    assert result.symmetry_automorphism_count == 1
    assert verify_distance_enumeration_certificate(
        lgp, result.certificate, mappings=result.mappings
    )


def test_mapping_limit_at_last_leaf_has_complete_certificate():
    graph = LabeledGraph({0: {}}, [6])
    result = enumerate_distance_mappings(
        [graph, graph.copy()],
        CD=0,
        binary=True,
        max_mappings=1,
        certify=True,
    )

    assert result.complete is True
    assert result.status == "complete"
    assert result.truncation_reason is None
    assert verify_distance_enumeration_certificate(
        [graph, graph.copy()], result.certificate, mappings=result.mappings
    )


def test_weighted_bounds_match_brute_force_for_every_shell():
    reactant = LabeledGraph(
        {
            0: {1: 1.0, 2: 0.5},
            1: {0: 1.0, 3: 1.5},
            2: {0: 0.5, 3: 1.0},
            3: {1: 1.5, 2: 1.0},
        },
        [6, 6, 6, 6],
    )
    product = LabeledGraph(
        {
            0: {1: 1.5, 3: 0.5},
            1: {0: 1.5, 2: 1.0},
            2: {1: 1.0, 3: 1.0},
            3: {0: 0.5, 2: 1.0},
        },
        [6, 6, 6, 6],
    )
    lgp = [reactant, product]
    brute = {
        permutation: chemical_distance(lgp, permutation, binary=False)
        for permutation in itertools.permutations(range(4))
    }
    minimum = min(brute.values())

    optimal = enumerate_distance_mappings(
        lgp,
        CD="minimal",
        binary=False,
        max_bijections=None,
        tolerance=0,
    )
    assert optimal.minimum_cost == minimum
    assert {tuple(mapping) for mapping in optimal.mappings} == {
        permutation for permutation, cost in brute.items() if cost == minimum
    }

    for shell in sorted(set(brute.values())):
        result = enumerate_distance_mappings(
            lgp,
            CD=shell,
            binary=False,
            max_bijections=None,
            tolerance=0,
            certify=True,
        )
        assert result.minimum_cost == minimum
        assert {tuple(mapping) for mapping in result.mappings} == {
            permutation for permutation, cost in brute.items() if cost == shell
        }
        assert verify_distance_enumeration_certificate(
            lgp, result.certificate, mappings=result.mappings
        )
        profile_pruned = enumerate_distance_mappings(
            lgp,
            CD=shell,
            binary=False,
            max_bijections=None,
            tolerance=0,
            compute_minimum_cost=False,
            atom_profile_pruning=True,
            initial_mapping=list(range(4)),
        )
        assert {tuple(mapping) for mapping in profile_pruned.mappings} == {
            permutation for permutation, cost in brute.items() if cost == shell
        }
        fixed = enumerate_distance_mappings(
            lgp,
            CD=shell,
            binary=False,
            max_bijections=None,
            tolerance=0,
            compute_minimum_cost=False,
            fixed_mapping={0: 0},
        )
        assert {tuple(mapping) for mapping in fixed.mappings} == {
            permutation
            for permutation, cost in brute.items()
            if permutation[0] == 0 and cost == shell
        }


def test_minimal_mappings_can_stream_without_collection():
    permutation = [0, 2, 4, 1, 3]
    lgp = [_cycle_graph(5), _cycle_graph(5, permutation)]
    streamed = []

    result = enumerate_distance_mappings(
        lgp,
        CD="minimal",
        binary=True,
        max_bijections=None,
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: streamed.append((mapping, cost)),
    )

    assert result.complete is True
    assert result.minimum_cost == 0
    assert result.selected_mapping_count == 10
    assert result.mappings == []
    assert result.distances == []
    assert len(streamed) == 10
    assert {cost for _, cost in streamed} == {0}

    with pytest.raises(ValueError, match="requires collect_mappings"):
        enumerate_distance_mappings(
            lgp,
            CD="minimal",
            binary=True,
            max_bijections=None,
            collect_mappings=False,
            certify=True,
        )

    mapper = AAMapper(binary=True)
    streamed_smiles = []
    public_result = mapper.enumerate_smiles(
        "CCC>>CCC",
        CD="minimal",
        add_Hs=False,
        unique=False,
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: streamed_smiles.append((mapping, cost)),
    )
    assert public_result.selected_mapping_count == 2
    assert len(streamed_smiles) == 2
    assert mapper.results == []


def test_numeric_shell_can_skip_separate_minimum_pass(monkeypatch):
    lgp = [_cycle_graph(5), _cycle_graph(5)]
    original = distance_module.enumerate_distance_mappings
    calls = []

    def recording_enumerator(*args, **kwargs):
        calls.append(kwargs.get("CD", "minimal"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        distance_module, "enumerate_distance_mappings", recording_enumerator
    )
    result = recording_enumerator(
        lgp,
        CD=0,
        binary=True,
        max_bijections=None,
        collect_mappings=False,
        compute_minimum_cost=False,
    )

    assert calls == [0]
    assert result.complete is True
    assert result.minimum_cost is None
    assert result.selected_mapping_count == 10


def test_timed_out_stream_does_not_emit_unproven_minima():
    lgp = [_cycle_graph(5), _cycle_graph(5)]
    streamed = []

    result = enumerate_distance_mappings(
        lgp,
        CD="minimal",
        binary=True,
        max_bijections=None,
        time_limit_seconds=0,
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: streamed.append((mapping, cost)),
    )

    assert result.status == "timeout"
    assert result.minimum_cost is None
    assert result.selected_mapping_count == 0
    assert streamed == []
