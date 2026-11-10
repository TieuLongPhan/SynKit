"""The SynKit DPO CRN backend executes the actual C++ reactor kernel."""

from collections import Counter

import pytest
from rdkit.Chem import rdChemReactions

from synkit.CRN.Construct import CRNExpand
from synkit.Synthesis.Reactor import NativeReactorUnsupportedError

FORMOSE = [
    "[C:1]([C:2]=[O:3])[H:4]>>[C:1]=[C:2][O:3][H:4]",
    "[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]>>[C:1]([C:2]=[O:3])[C:6][O:5][H:4]",
    "[C:1]=[C:2][O:3][H:4]>>[C:1]([C:2]=[O:3])[H:4]",
    "[C:1]([C:2]=[O:3])[C:6][O:5][H:4]>>[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]",
]
SEEDS = ["C=O", "OCC=O"]


@pytest.fixture(scope="module")
def library(reactor_native_library):
    return str(reactor_native_library)


def build(library, *, backend="synreactor-cpp", depth=3, **options):
    settings = dict(
        rules=FORMOSE,
        repeats=depth,
        keep_aam=False,
        explicit_h=False,
        implicit_temp=True,
        rule_backend=backend,
    )
    if backend != "synreactor":
        settings.update(reactor_library=library, dpo_radical_policy="ignore")
    settings.update(options)
    builder = CRNExpand(**settings)
    graph = builder.build(SEEDS)
    return builder, graph


def signature(graph):
    species = {n: d["smiles"] for n, d in graph.nodes(data=True) if d["kind"] == "species"}
    reactions = []
    for node, data in graph.nodes(data=True):
        if data["kind"] != "rule":
            continue
        lhs = tuple(sorted((species[u], e["stoich"]) for u, _, e in graph.in_edges(node, data=True)))
        rhs = tuple(sorted((species[v], e["stoich"]) for _, v, e in graph.out_edges(node, data=True)))
        reactions.append((lhs, rhs))
    return sorted(species.values()), sorted(reactions)


def test_formose_matches_legacy_and_typed_python_dpo(library):
    cpp, graph = build(library, depth=4)
    _, typed_python = build(library, backend="synreactor-python", depth=4)
    _, legacy = build(library, backend="synreactor", depth=4)
    assert signature(graph) == signature(typed_python) == signature(legacy)
    assert Counter(d["kind"] for _, d in graph.nodes(data=True)) == {"species": 37, "rule": 44}
    report = cpp.dpo_report
    assert report["engine"] == "synkit_dpo_cpp"
    assert report["kernel"] == "cpp" and report["controller"] == "python"
    assert report["partial"] is False and report["hydrogen_model"] == "implicit"
    assert report["radical_policy"] == "ignore"
    assert report["native_applications"] > 0 and report["native_elapsed_ns"] > 0
    assert cpp.native_report is None  # Reserved for the distinct RDKit expander.


def test_cpp_never_uses_rdkit_reactions_or_python_match_rewrite(library, monkeypatch):
    from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
    from synkit.Graph.Matcher.partial_matcher import PartialMatcher
    from synkit.Synthesis.Reactor.core import rewrite
    from synkit.CRN.Construct import rdkit_worker

    def forbidden(*args, **kwargs):
        raise AssertionError("A reaction executor outside the SynKit C++ DPO kernel was called")

    monkeypatch.setattr(rdChemReactions, "ReactionFromSmarts", forbidden)
    monkeypatch.setattr(rdChemReactions.ChemicalReaction, "RunReactants", forbidden)
    monkeypatch.setattr(rdkit_worker, "apply_rdkit_rule_worker", forbidden)
    monkeypatch.setattr(SubgraphSearchEngine, "find_subgraph_mappings", forbidden)
    monkeypatch.setattr(PartialMatcher, "get_mappings", forbidden)
    monkeypatch.setattr(rewrite, "_glue_graph", forbidden)
    builder, graph = build(library, depth=3)
    assert len(signature(graph)[0]) == 9
    assert len(signature(graph)[1]) == 10
    assert builder.dpo_report["native_applications"] > 0


def test_rules_and_library_are_prepared_once_per_build(library, monkeypatch):
    from synkit.Rule import SynRule
    from synkit.Synthesis.Reactor._native import runtime

    calls = {"rules": 0, "library": 0}
    compile_rule, load_library = SynRule.from_smart, runtime.load

    def compile_once(*args, **kwargs):
        calls["rules"] += 1
        return compile_rule(*args, **kwargs)

    def load_once(*args, **kwargs):
        calls["library"] += 1
        return load_library(*args, **kwargs)

    monkeypatch.setattr(SynRule, "from_smart", compile_once)
    monkeypatch.setattr(runtime, "load", load_once)
    builder, _ = build(library, depth=3)
    assert calls == {"rules": 4, "library": 1}
    assert builder.dpo_report["rule_calls"] > 4


def test_generic_radical_policy_is_explicit_and_strict_remains_default(library):
    strict = CRNExpand(
        rules=FORMOSE[:1], repeats=1, keep_aam=False, rule_backend="synreactor-cpp", reactor_library=library
    )
    strict_graph = strict.build(SEEDS)
    compatible, compatible_graph = build(library, depth=1, rules=FORMOSE[:1])
    assert strict.dpo_radical_policy == "strict"
    assert len(signature(strict_graph)[1]) == 0
    assert len(signature(compatible_graph)[1]) == 1
    assert compatible.dpo_report["radical_policy"] == "ignore"


def test_cpp_refuses_stereo_instead_of_falling_back(library):
    builder = CRNExpand(
        rules=["[C:1][H:2]>>[C:1][H:2]"],
        repeats=1,
        rule_backend="synreactor-cpp",
        reactor_library=library,
        dpo_radical_policy="ignore",
        keep_aam=False,
    )
    with pytest.raises(NativeReactorUnsupportedError, match="Stereo"):
        builder.build(["N[C@H](C)F"])


def test_strict_radical_recombination_consumes_real_radicals(library):
    settings = dict(
        rules=["[CH3:1].[CH3:2]>>[CH3:1][CH3:2]"],
        repeats=1,
        keep_aam=False,
        allow_self_mixtures=True,
        dpo_radical_policy="strict",
    )
    cpp = CRNExpand(rule_backend="synreactor-cpp", reactor_library=library, **settings)
    python = CRNExpand(rule_backend="synreactor-python", **settings)
    actual = cpp.build(["[CH3]", "C"])
    expected = python.build(["[CH3]", "C"])
    assert signature(actual) == signature(expected)
    assert signature(actual)[1] == [((("[CH3]", 2),), (("CC", 1),))]
    assert cpp.dpo_report["radical_policy"] == "strict"


def test_wire_compiler_preserves_the_native_rule_resource_contract(monkeypatch):
    import json

    from synkit.Synthesis.Reactor import WildcardEngine
    from synkit.Synthesis.Reactor._native import runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Wire compilation must not load or execute a reaction kernel")

    monkeypatch.setattr(runtime, "load", forbidden)
    permissive = WildcardEngine(SEEDS[1], FORMOSE[0], partial=False, backend="cpp", radical_policy="ignore")
    strict = WildcardEngine(SEEDS[1], FORMOSE[0], partial=False, backend="cpp")
    wire = runtime.serialize_native_request(permissive)
    strict_wire = runtime.serialize_native_request(strict)
    assert json.loads(json.dumps(wire)) == wire
    assert not wire["partial"] and wire["candidate_limit"] == -1
    assert wire["radical_policy"] == 2 and strict_wire["radical_policy"] == 0
    assert all(not atom["match_mask"] & 8 for atom in wire["rule_atoms"])
    # Context-decorated unknown attributes may omit a guard even in strict
    # mode, but known radical guards must remain present.
    assert any(atom["match_mask"] & 8 for atom in strict_wire["rule_atoms"])
    assert len(wire["rule_atoms"]) == 3  # The transferred H is a count edit.
    by_map = {atom["atom_map"]: atom for atom in wire["rule_atoms"]}
    assert by_map[1]["right"]["hcount"] - by_map[1]["left"]["hcount"] == -1
    assert by_map[3]["right"]["hcount"] - by_map[3]["left"]["hcount"] == 1
    assert by_map[3]["baseline_lp"] == 2


def test_process_pool_preserves_network_and_native_counters(library):
    serial, graph = build(library, depth=3)
    parallel = CRNExpand(
        rules=FORMOSE,
        repeats=3,
        keep_aam=False,
        rule_backend="synreactor-cpp",
        reactor_library=library,
        dpo_radical_policy="ignore",
    )
    result = parallel.build(SEEDS, parallel=True, max_workers=2)
    assert signature(result) == signature(graph)
    for key in ["rule_calls", "native_candidates", "native_applications"]:
        assert parallel.dpo_report[key] == serial.dpo_report[key]
    assert parallel.dpo_report["native_elapsed_ns"] > 0


def test_disconnected_lhs_union_matching_retains_an_unused_spectator(library):
    rule = "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[OH2:3]"
    settings = dict(rules=[rule], repeats=1, keep_aam=False, implicit_temp=True)
    outputs = []
    for backend in ["synreactor", "synreactor-python", "synreactor-cpp"]:
        builder = CRNExpand(
            rule_backend=backend,
            reactor_library=library,
            dpo_radical_policy="ignore",
            dpo_component_policy="union",
            **settings,
        )
        outputs.append(signature(builder.build(["OCC(=O)O", "C"])))
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][1] == [
        (
            (("C", 1), ("O=C(O)CO", 1)),
            (("C", 1), ("O", 1), ("O=C1CO1", 1)),
        )
    ]
    for backend in ["synreactor-python", "synreactor-cpp"]:
        builder = CRNExpand(
            rule_backend=backend,
            reactor_library=library,
            dpo_radical_policy="ignore",
            dpo_component_policy="distinct",
            **settings,
        )
        assert signature(builder.build(["OCC(=O)O", "C"]))[1] == []
        assert builder.dpo_report["component_policy"] == "distinct"
        assert builder.dpo_report["component_rejected_applications"] > 0


def test_intramolecular_engine_application_and_controller_arity_are_distinct(library):
    from rdkit import Chem
    from synkit.Synthesis.Reactor import WildcardEngine

    rule = "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[OH2:3]"
    products = []
    for backend in ["python", "cpp"]:
        engine = WildcardEngine(
            "OCC(=O)O",
            rule,
            backend=backend,
            library_path=library if backend == "cpp" else None,
            partial=False,
            radical_policy="ignore",
        )
        endpoints = set()
        for text in engine.smiles_list:
            mol = Chem.MolFromSmiles(text)
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            endpoints.add(Chem.MolToSmiles(mol))
        products.append(endpoints)
    assert products == [{"O.O=C1CO1"}] * 2
    # The network controller enumerates arity-two mixtures. A single seed with
    # self mixtures disabled supplies no such task, although the DPO kernel
    # itself permits both LHS components to embed within that one molecule.
    for backend in ["synreactor", "synreactor-cpp"]:
        builder = CRNExpand(
            rules=[rule],
            repeats=1,
            rule_backend=backend,
            reactor_library=library,
            keep_aam=False,
            dpo_radical_policy="ignore",
        )
        assert signature(builder.build(["OCC(=O)O"]))[1] == []


@pytest.mark.parametrize(
    "options,match",
    [
        ({"explicit_h": True}, "explicit_h=False"),
        ({"keep_aam": True}, "keep_aam=False"),
        ({"dpo_radical_policy": "guess"}, "dpo_radical_policy"),
        ({"time_limit": 1.0}, "cancellation"),
    ],
)
def test_dpo_profile_refuses_unsupported_settings(options, match):
    settings = dict(rules=FORMOSE, keep_aam=False, rule_backend="synreactor-cpp")
    settings.update(options)
    with pytest.raises(ValueError, match=match):
        CRNExpand(**settings)
