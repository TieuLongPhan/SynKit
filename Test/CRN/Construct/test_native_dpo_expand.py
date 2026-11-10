"""The full C++ DPO controller preserves existing reactor/CRN semantics."""

from collections import Counter
import os
from pathlib import Path

import pytest
from rdkit import Chem

from synkit.CRN.Construct import CRNExpand
from synkit.CRN.Construct.native_dpo import (
    build_native_dpo_expander,
    native_dpo_expander,
    run_native_dpo_expansion,
    dpo_expander_manifest,
)
from Experiment.CRN.studies.formose_case_study import RULES, SEEDS


@pytest.fixture(scope="module")
def controller(tmp_path_factory):
    configured = os.environ.get("SYNKIT_RDKIT_CPP_DIR")
    sdks = [Path(configured)] if configured else []
    sdks += sorted((Path.home() / ".cache/synkit").glob("rdkit-cpp-*"), reverse=True)
    sdk = next((p for p in sdks if (p / "include/rdkit/GraphMol/GraphMol.h").exists()), None)
    if sdk is None:
        pytest.skip("A molecular RDKit C++ SDK is required")
    return build_native_dpo_expander(tmp_path_factory.mktemp("full-dpo-crn"), rdkit_dir=sdk)


def signature(graph):
    labels = {n: d["smiles"] for n, d in graph.nodes(data=True) if d["kind"] == "species"}
    reactions = set()
    for n, d in graph.nodes(data=True):
        if d["kind"] != "rule":
            continue
        left = tuple(sorted((labels[u], int(e["stoich"])) for u, _, e in graph.in_edges(n, data=True)))
        right = tuple(sorted((labels[v], int(e["stoich"])) for _, v, e in graph.out_edges(n, data=True)))
        reactions.add((left, right))
    return set(labels.values()), reactions


def builder(controller, **kwargs):
    options = dict(
        rules=RULES,
        repeats=4,
        keep_aam=False,
        rule_backend="synreactor-native",
        native_dpo_expander=str(controller),
        hydrogen_model="lewis",
        implicit_temp=True,
        dpo_radical_policy="ignore",
    )
    options.update(kwargs)
    return CRNExpand(**options)


@pytest.mark.parametrize("policy", ["union", "distinct"])
def test_formose_matches_existing_dpo_kernel_controller(controller, policy):
    b = builder(controller, dpo_component_policy=policy)
    native = b.build(SEEDS)
    reference = CRNExpand(
        rules=RULES,
        repeats=4,
        keep_aam=False,
        rule_backend="synreactor-cpp",
        reactor_library=dpo_expander_manifest(controller)["kernel_library"],
        dpo_radical_policy="ignore",
        dpo_component_policy=policy,
    ).build(SEEDS)
    assert signature(native) == signature(reference)
    assert tuple(map(len, signature(native))) == (37, 44)
    assert b.dpo_report["controller"] == "cpp"
    assert b.dpo_report["kernel"] == "synkit_dpo_cpp"


def test_full_controller_avoids_python_task_and_rdkit_reaction_paths(controller, monkeypatch):
    from synkit.CRN.Construct import native_expand
    from rdkit.Chem import rdChemReactions

    def forbidden(*args, **kwargs):
        raise AssertionError("Generation must remain in the C++ DPO controller")

    monkeypatch.setattr(CRNExpand, "_run_tasks", forbidden)
    monkeypatch.setattr(native_expand, "run_native_expansion", forbidden)
    monkeypatch.setattr(rdChemReactions, "ReactionFromSmarts", forbidden)
    monkeypatch.setattr(rdChemReactions.ChemicalReaction, "RunReactants", forbidden)
    b = builder(controller)
    assert tuple(map(len, signature(b.build(SEEDS)))) == (37, 44)
    assert b.dpo_report["kernel_calls"] > 0


def test_native_dpo_conserves_elements_and_charge(controller):
    graph = builder(controller).build(SEEDS)
    labels = {n: d["smiles"] for n, d in graph.nodes(data=True) if d["kind"] == "species"}
    inventories = {}
    for n, s in labels.items():
        mol = Chem.AddHs(Chem.MolFromSmiles(s))
        inventory = Counter(a.GetSymbol() for a in mol.GetAtoms())
        inventory["charge"] = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        inventories[n] = inventory
    for n, d in graph.nodes(data=True):
        if d["kind"] != "rule":
            continue
        difference = Counter()
        for u, _, e in graph.in_edges(n, data=True):
            for k, v in inventories[u].items():
                difference[k] -= v * e["stoich"]
        for _, v, e in graph.out_edges(n, data=True):
            for k, c in inventories[v].items():
                difference[k] += c * e["stoich"]
        assert not any(difference.values())


def test_rejected_spectator_event_does_not_register_products(controller):
    rule = "[CH3:1][CH2:2][OH:3].[CH3:4][CH2:5][OH:6]>>[CH3:1][O:3][CH3:2].[CH3:4][CH2:5][OH:6]"
    b = builder(
        controller, rules=[rule], repeats=1, allow_self_mixtures=True, strip_spectators=True, allow_empty_side=False
    )
    graph = b.build(["CCO"])
    assert signature(graph) == ({"CCO"}, set())


@pytest.mark.parametrize("policy,expected", [("union", 1), ("distinct", 0)])
def test_component_policy_retains_intramolecular_union_semantics(controller, policy, expected):
    rule = "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[OH2:3]"
    b = builder(controller, rules=[rule], repeats=1, dpo_component_policy=policy)
    graph = b.build(["OCC(=O)O", "C"])
    assert len(signature(graph)[1]) == expected
    reference = CRNExpand(
        rules=[rule],
        repeats=1,
        keep_aam=False,
        rule_backend="synreactor-cpp",
        reactor_library=dpo_expander_manifest(controller)["kernel_library"],
        dpo_radical_policy="ignore",
        dpo_component_policy=policy,
    ).build(["OCC(=O)O", "C"])
    assert signature(graph) == signature(reference)


def test_reused_native_server_is_deterministic(controller):
    with native_dpo_expander(controller):
        first = builder(controller).build(SEEDS)
        second = builder(controller).build(SEEDS)
    assert signature(first) == signature(second)


@pytest.mark.parametrize(
    "limits,reason", [({"time_limit": 0}, "time_limit"), ({"max_mixtures_per_rule_step": 1}, "mixture_limit")]
)
def test_native_bounds_are_explicit_incomplete_results(controller, limits, reason):
    options = dict(repeats=4, component_policy="union", **limits)
    result = run_native_dpo_expansion(controller, rules=RULES, seeds=SEEDS, options=options, radical_policy="ignore")
    assert result["status"] == "limit"
    assert result["stop_reason"] == reason


def test_unsupported_isotope_is_not_silently_erased(controller):
    with pytest.raises(ValueError, match="Isotope"):
        builder(controller).build(["[13CH2]=O", "OCC=O"])


def test_nonstandard_bond_partition_is_not_erased_by_molecular_identity(controller):
    from synkit.CRN.Construct.native_dpo import compile_dpo_request, _invoke

    payload = compile_dpo_request(rules=[], seeds=["CC"], options={"repeats": 0})
    edge = payload["seeds"][0]["edges"][0]["state"]
    edge.update(sigma2=0, pi2=edge["order2"])
    with pytest.raises(RuntimeError, match="sigma/pi"):
        _invoke(controller, payload)
