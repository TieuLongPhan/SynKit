"""Native CRN regression checks against independent Python references."""

from itertools import combinations

import numpy as np
import networkx as nx
from synkit.CRN import (
    SynCRN,
    crnt_summary,
    integer_conservation_laws,
    native_backend,
    PathwayRealizability,
    crn_from_sbml,
    enumerate_structural_sets,
    siphon_persistence_details,
)
from synkit.CRN.backend import python_backend
from synkit.CRN.native_analysis import (
    matrix_analysis,
    graph_symmetry,
    NativeSearchLimit,
    exhaustive_siphons,
)
from synkit.CRN.Props.stoich import _exact_null_space
from synkit.CRN.Props.thermo import is_consistent
from Experiment.CRN.benchmarks.networks import BENCHMARK_NETWORKS
from Experiment.CRN.benchmarks.crosschecks import brute_force_siphons
from Experiment.CRN.studies.formose_case_study import expand


def _assert_agreement(checks):
    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, "Native/reference disagreements: " + ", ".join(failed)


def test_native_fixture_analysis_agrees_with_independent_references(exact_library):
    library = exact_library
    checks = {}
    for fixture in BENCHMARK_NETWORKS:
        c = fixture.build()
        expected = crnt_summary(c, backend="python").to_dict()
        actual = crnt_summary(c, library_path=library).to_dict()
        checks["crnt_" + fixture.name] = actual == expected
        checks["laws_" + fixture.name] = integer_conservation_laws(
            c, library_path=library
        ) == integer_conservation_laws(c, backend="python")
        if c.n_species <= 14:
            native = exhaustive_siphons(c, library_path=library)
            # The reference returns source graph node ids; align by labels.
            label = {sid: s.label for sid, s in c.species.items()}
            source = {s.source_node_id: s.label for s in c.species.values()}
            checks["exhaustive_" + fixture.name] = {
                frozenset(label.get(s, s) for s in group) for group in native
            } == {
                frozenset(source.get(s, s) for s in group)
                for group in brute_force_siphons(c, max_species=14)
            }
    _assert_agreement(checks)


def test_native_matrix_analysis_agrees_with_independent_references(exact_library):
    library = exact_library
    checks = {}
    rng = np.random.default_rng(42)
    for i in range(30):
        s = rng.integers(-3, 4, size=(int(rng.integers(1, 6)), int(rng.integers(1, 6))))
        got = matrix_analysis(s, library_path=library, kernels="both")
        checks[f"algebra_{i}"] = got["left"] == _exact_null_space(
            s.T.tolist(), s.shape[0]
        ) and got["right"] == _exact_null_space(s.tolist(), s.shape[1])
    _assert_agreement(checks)


def test_sbml_persistence_preserves_coordinate_alignment():
    xml = (
        '<sbml><model><listOfSpecies><species id="A"/><species id="B"/>'
        '</listOfSpecies><listOfReactions><reaction id="f"><listOfReactants>'
        '<speciesReference species="A"/></listOfReactants><listOfProducts>'
        '<speciesReference species="B"/></listOfProducts></reaction>'
        '<reaction id="r"><listOfReactants><speciesReference species="B"/>'
        '</listOfReactants><listOfProducts><speciesReference species="A"/>'
        '</listOfProducts></reaction></listOfReactions></model></sbml>'
    )
    assert siphon_persistence_details(crn_from_sbml(xml)).persistence_ok is True


def test_native_symmetry_and_pathways_agree_with_independent_references(exact_library):
    library = exact_library
    checks = {}
    cases = [
        ["A>>B", "B>>A"],
        ["A>>B", "B>>C", "C>>A"],
        ["A+B>>C", "C>>A+B"],
        ["A>>2B", "2B>>A"],
    ]
    for i, reactions in enumerate(cases):
        c = SynCRN.from_reaction_strings(reactions)
        g = c.to_digraph(node_ids="internal")
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            g,
            g,
            node_match=nx.algorithms.isomorphism.categorical_node_match("kind", None),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match("stoich", 1),
        )
        symmetry = graph_symmetry(c, library_path=library)
        checks[f"symmetry_{i}"] = symmetry.complete and symmetry.value[
            "automorphism_count"
        ] == sum(1 for _ in matcher.isomorphisms_iter())
        permuted = nx.relabel_nodes(
            g, {node: "renamed_" + str(j) for j, node in enumerate(reversed(list(g)))}
        )
        checks[f"canonical_{i}"] = (
            graph_symmetry(permuted, library_path=library).value["canonical_key"]
            == symmetry.value["canonical_key"]
        )
        flow = {r: 1 for r in c.reactions}
        for marking in ({}, {sid: 2 for sid in c.species}):
            search = PathwayRealizability().load_syncrn_and_flow(
                c, flow=flow, initial_marking=marking, species="id", reaction="id"
            )
            search.build_petri_net_from_flow()
            a = search.realizability_result(backend="python")
            b = search.realizability_result(backend="cpp", library_path=library)
            checks[f"pathway_{i}_{len(marking)}"] = (
                a.status == b.status and a.exhaustive == b.exhaustive
            )
    _assert_agreement(checks)


def test_native_formose_agrees_with_independent_references(exact_library):
    library = exact_library
    checks = {}

    with python_backend():
        reference = expand(4)
    with native_backend(library):
        formose = expand(4)

    def signatures(model):
        species = sorted(s.smiles for s in model.species.values())
        reactions = sorted(
            (
                tuple(
                    sorted((model.species[sid].smiles, n) for sid, n in r.lhs.items())
                ),
                tuple(
                    sorted((model.species[sid].smiles, n) for sid, n in r.rhs.items())
                ),
            )
            for r in model.reactions.values()
        )
        return species, reactions

    checks["native_input_generation_matches_reference"] = signatures(
        formose
    ) == signatures(reference)
    symmetry = graph_symmetry(formose, library_path=library, max_count=1024)
    checks["formose_group_512"] = (
        symmetry.complete and symmetry.value["automorphism_count"] == 512
    )
    checks["formose_18_paired_orbits"] = (
        sum(len(o) == 2 for o in symmetry.value["orbits"]) == 18
    )

    # Every singleton trap dominates all larger sets containing that species.
    # Exhaust only the remaining species to independently check completeness.
    def is_trap(group):
        return all(
            not (set(reaction.lhs.counts) & group)
            or bool(set(reaction.rhs.counts) & group)
            for reaction in formose.reactions.values()
        )

    singletons = {sid for sid in formose.species if is_trap({sid})}
    remaining = sorted(set(formose.species) - singletons)
    if len(remaining) > 14:
        raise RuntimeError("Formose residual trap oracle exceeds its subset budget")
    oracle = {frozenset({sid}) for sid in singletons}
    for size in range(1, len(remaining) + 1):
        for subset in combinations(remaining, size):
            candidate = frozenset(subset)
            if not any(trap < candidate for trap in oracle) and is_trap(candidate):
                oracle.add(candidate)
    traps = enumerate_structural_sets(formose, kind="trap", library_path=library)
    checks["formose_traps_match_exhaustive_residual_oracle"] = (
        traps.complete and {frozenset(group) for group in traps.value["sets"]} == oracle
    )
    checks["formose_28_minimal_traps"] = len(oracle) == 28
    _assert_agreement(checks)


def test_native_limits_and_positive_flux_bounds_are_sound(exact_library):
    library = exact_library
    checks = {}
    try:
        matrix_analysis(np.eye(10, dtype=int), library_path=library, time_limit=1e-12)
        checks["algebra_limit_explicit"] = False
    except NativeSearchLimit:
        checks["algebra_limit_explicit"] = True
    limited = graph_symmetry(
        SynCRN.from_reaction_strings(["A>>B", "B>>A"]),
        library_path=library,
        max_count=1,
    )
    checks["canonical_limit_no_key"] = (
        not limited.complete and limited.value["canonical_key"] is None
    )
    with native_backend(library):
        checks["positive_flux_bounds_sound"] = (
            is_consistent(SynCRN.from_reaction_strings(["A>>B", "2A>>2B"])) is False
        )
    _assert_agreement(checks)
