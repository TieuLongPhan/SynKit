"""Regression contracts for the biochemical analysis workflow (offline)."""

import itertools
import tempfile
from pathlib import Path
import numpy as np
import pytest
from synkit.CRN import (
    SynCRN,
    CRNAnalysis,
    Scenario,
    PetriNet,
    build_native,
    enumerate_structural_sets,
    siphon_persistence_details,
)
from synkit.CRN.Petrinet.semiflows import left_kernel_basis
from synkit.CRN.Petrinet.structure import find_siphons, find_traps
from synkit.CRN.Props.thermo import is_consistent, has_irreversible_futile_cycles
from synkit.CRN.Petrinet.minimal_semiflows import minimal_semiflows
from synkit.CRN.IO import crn_from_sbml, crn_to_sbml


def test_sound_limits_and_direction():
    c = SynCRN.from_reaction_strings(["A>>B", "B>>2A"])
    assert siphon_persistence_details(c).persistence_ok is False
    limited = siphon_persistence_details(c, max_siphon_size=1)
    assert limited.persistence_ok is None and not limited.complete
    c = SynCRN.from_reaction_strings(["A>>B", "2A>>2B"])
    assert is_consistent(c) is False
    assert has_irreversible_futile_cycles(c) is False


def test_nullspace_representation_invariance():
    c = SynCRN.from_reaction_strings(["A>>400001 A+400001 B", "B>>399999 A+400001 B"])
    assert (
        left_kernel_basis(c).shape == left_kernel_basis(PetriNet.from_syncrn(c)).shape
    )


def test_snapshot_rescue_and_replay():
    c = SynCRN.from_reaction_strings(["S+ATP>>I+ADP", "I+ATP>>P+ADP"])
    a = CRNAnalysis(c)
    flow = {r: 1 for r in a.reaction_order}
    assert a.verify_pathway(flow, {"S": 1, "ATP": 1})["status"] == "unrealizable"
    rescue = a.minimum_supplements(
        flow, initial_marking={"S": 1, "ATP": 1}, allowed=["ATP"]
    )
    assert rescue.value["cost"] == 1
    assert rescue.value["optimal_within_bounds"]
    assert rescue.certificate["replay_verified"]
    c.species.clear()
    assert a.audit().value["n_species"] == 5
    detached = a.model
    detached.species.clear()
    assert a.audit().value["n_species"] == 5


def test_chemostats_are_explicit():
    c = SynCRN.from_reaction_strings(["S+ATP>>P+ADP"])
    a = CRNAnalysis(c, Scenario(chemostats=("ATP",)))
    assert (
        a.verify_pathway({a.reaction_order[0]: 1}, {"S": 1})["status"] == "realizable"
    )
    assert not CRNAnalysis(c).verify_pathway({a.reaction_order[0]: 1}, {"S": 1})[
        "replay_verified"
    ]


def test_flux_and_qualitative_scope():
    c = SynCRN.from_reaction_strings(["A>>B"])
    closed = CRNAnalysis(c)
    assert closed.flux_feasibility(required={"r_1": 1}).value is False
    opened = CRNAnalysis(c, Scenario(chemostats=("A", "B")))
    assert opened.flux_feasibility(required={"r_1": 1}).value is True
    assert closed.blocked_reactions().value["blocked"] == ["r_1"]
    assert len(closed.accessibility(["A"]).value["reachable_species"]) == 2


def test_exact_precheck():
    assert minimal_semiflows(np.array([[2, 1], [1, 2]])) == []
    assert minimal_semiflows(np.array([[-1], [1]])) == [[1, 1]]
    with pytest.raises(ValueError):
        minimal_semiflows(np.array([[100000.1]]))


def test_metadata_roundtrip():
    source = '<sbml><model><listOfSpecies><species id="A" compartment="outside" boundaryCondition="true" initialConcentration="2.5"/><species id="B" compartment="cell"/></listOfSpecies><listOfReactions><reaction id="r"><listOfReactants><speciesReference species="A"/></listOfReactants><listOfProducts><speciesReference species="B"/></listOfProducts></reaction></listOfReactions></model></sbml>'
    c = crn_from_sbml(source)
    out = crn_from_sbml(crn_to_sbml(c))
    attrs = next(s.source_attrs for s in out.species.values() if s.label == "A")
    assert attrs["boundaryCondition"] == "true"
    assert attrs["compartment"] == "outside"
    assert attrs["initialConcentration"] == "2.5"


def test_native_matches_exhaustive_small_networks(tmp_path):
    library = build_native(tmp_path)
    candidates = ["A>>B", "B>>A", "A+B>>C", "C>>A", "C>>B"]
    for subset in itertools.combinations(candidates, 3):
        c = SynCRN.from_reaction_strings(subset)
        for kind, reference in [("siphon", find_siphons), ("trap", find_traps)]:
            expected = {frozenset(s) for s in reference(c, names="id")}
            for backend in ["python", "cpp"]:
                result = enumerate_structural_sets(
                    c, kind=kind, backend=backend, library_path=library
                )
                assert result.complete
                assert {frozenset(s) for s in result.value["sets"]} == expected
    assert not enumerate_structural_sets(c, max_nodes=1).complete


def test_limits_inputs_and_open_conservation():
    c = SynCRN.from_reaction_strings(["A+C>>B+C"])
    a = CRNAnalysis(c)
    flow = {a.reaction_order[0]: 1}
    with pytest.raises(ValueError):
        a.verify_pathway(flow, {"A": 0.5})
    limited = a.minimum_supplements(flow, allowed=["C"], max_added=10, max_candidates=2)
    assert limited.status == "inconclusive"
    result = a.minimum_supplements(
        flow, initial_marking={"A": 1}, allowed=["C"], max_added=1, max_states=1
    )
    assert result.certificate["replay_verified"]
    open_model = CRNAnalysis(c, Scenario(chemostats=("C",)))
    assert "C" not in [
        open_model.model.species[s].label
        for s in open_model.conservation().value["species_order"]
    ]
    assert open_model.structure().value["deficiency_zero"]["applicable"] is None


def test_unannotated_graph_direction_and_cache_lifetime():
    import networkx as nx
    from synkit.CRN import stoichiometric_matrix, crnt_summary

    g = nx.DiGraph()
    g.add_nodes_from(
        [
            ("A", {"kind": "species"}),
            ("B", {"kind": "species"}),
            ("r", {"kind": "reaction"}),
        ]
    )
    g.add_edge("A", "r", stoich=2)
    g.add_edge("r", "B", stoich=1)
    assert sorted(stoichiometric_matrix(g).ravel()) == [-2, 1]
    c = SynCRN.from_reaction_strings(["A>>B"])
    assert crnt_summary(c).rank == 1
    r = next(iter(c.reactions.values()))
    r.rhs.counts = dict(r.lhs.counts)
    assert crnt_summary(c).rank == 0


def test_nonintegral_sbml_rejected():
    source = '<sbml><model><listOfSpecies><species id="A"/><species id="B"/></listOfSpecies><listOfReactions><reaction id="r"><listOfReactants><speciesReference species="A" stoichiometry="1.5"/></listOfReactants><listOfProducts><speciesReference species="B"/></listOfProducts></reaction></listOfReactions></model></sbml>'
    with pytest.raises(ValueError, match="integral"):
        crn_from_sbml(source)


def test_target_pathways_are_verified_but_not_exhaustive():
    a = CRNAnalysis(SynCRN.from_reaction_strings(["A>>B", "B>>C"]))
    result = a.pathways(["A"], ["C"], initial_marking={"A": 1})
    assert result.value and result.value[0]["execution"]["replay_verified"]
    assert not result.complete
    assert not a.pathways(["A"], ["C"], initial_marking={"A": 0}).value[0]["execution"][
        "replay_verified"
    ]


def test_research_imports_redirect_to_single_implementation():
    import importlib
    import warnings

    pairs = [
        ("networks", "benchmarks.networks"),
        ("crosschecks", "benchmarks.crosschecks"),
        ("validate", "benchmarks.validate"),
        ("kegg", "benchmarks.kegg"),
        ("scaling", "benchmarks.scaling"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old, new in pairs:
            assert importlib.import_module(
                "synkit.CRN.Benchmark." + old
            ) is importlib.import_module("Experiment.CRN." + new)


def test_user_kegg_cache_does_not_require_research_fixtures(tmp_path):
    import json
    from synkit.CRN.Query import load_kegg_cache, build_cached_kegg_crn

    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"modules": {"custom": {"equations": {"r": "C00001 => C00002"}}}})
    )
    cache = load_kegg_cache(path)
    assert build_cached_kegg_crn(cache, "custom").n_species == 2
    assert (
        build_cached_kegg_crn(path, "custom", drop_compounds=["C00001"]).n_species == 1
    )
    path.write_text('{"modules": {"bad": {"equations": []}}}')
    with pytest.raises(ValueError, match="equations"):
        load_kegg_cache(path)
