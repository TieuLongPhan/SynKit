"""Exercise the common adapter against real solvers and independent costs."""

import json
from itertools import permutations

import pytest
from rdkit import Chem

from synkit.Chem.Mapper import LabeledGraph, Mapper, MappingResult, PropagationConfig
from synkit.Chem.Mapper.reaction_search import _graphs, _parse
from synkit.Chem.Mapper.slap.lap import chemical_distance


@pytest.mark.parametrize("method", Mapper.methods())
def test_all_methods_map_weighted_graphs_without_mutating_inputs(
    method, native_library
):
    reactant = LabeledGraph({0: {1: 2}, 1: {0: 2}}, ["C", "O"])
    product = LabeledGraph({0: {1: 1}, 1: {0: 1}}, ["O", "C"])
    options = {"library_path": native_library} if method == "native" else {}
    if method == "edits":
        options.update(CD=0, binary=True)
    result = Mapper(method, **options).map((reactant, product))
    assert isinstance(result, MappingResult)
    assert result.method == method
    assert result.mappings == ((1, 0),)
    assert result.distances == ((0.0,) if method == "edits" else (1.0,))
    assert result.minimum_cost == (None if method in {"slap", "edits"} else 1.0)
    assert result.optimality_proven == (method not in {"slap", "edits"})
    assert result.complete == (method not in {"slap", "milp", "exhaustive"})
    assert result.search_scope == (
        "heuristic"
        if method == "slap"
        else "reduced_kernel" if method in {"kernel", "milp"} else "full_space"
    )
    assert reactant.labels == ["C", "O"] and reactant.graph[0][1] == 2
    assert product.labels == ["O", "C"] and product.props == {}
    assert result.mapped_reactions == ()
    assert json.loads(json.dumps(result.as_dict()))["mappings"] == [[1, 0]]


@pytest.mark.parametrize("method", Mapper.methods())
def test_all_methods_share_chemistry_and_preserve_isotopes(method, native_library):
    reaction = "[13CH3]CO>>[13CH3]C=O"
    options = {"library_path": native_library} if method == "native" else {}
    if method == "edits":
        options.update(CD=0, binary=True)
    result = Mapper(method, **options).map(reaction)
    assert result.mapped_reactions
    assert result.distances == ((0.0,) if method == "edits" else (1.0,))
    sides = result.mapped_reactions[0].split(">>")
    mols = [Chem.MolFromSmiles(side) for side in sides]
    isotope_maps = [
        [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetIsotope() == 13]
        for mol in mols
    ]
    assert isotope_maps[0] == isotope_maps[1]
    assert result.reaction_result.classification_complete
    assert result.reaction_result.objective == "heavy_bond_distance"
    assert json.loads(json.dumps(result.as_dict()))["method"] == method


@pytest.mark.parametrize("method", ["distance", "propagate", "native", "exhaustive"])
def test_full_space_minimum_matches_independent_permutations(method, native_library):
    mols, _ = _parse("CCO>>COC", "heavy")
    graphs = _graphs(mols, "strict", "bond")
    expected = min(
        chemical_distance(graphs, mapping, False)
        for mapping in permutations(range(3))
        if all(
            graphs[0].labels[i] == graphs[1].labels[j] for i, j in enumerate(mapping)
        )
    )
    options = {"library_path": native_library} if method == "native" else {}
    result = Mapper(method, **options).map("CCO>>COC")
    assert result.minimum_cost == expected == 2
    assert result.optimality_proven and result.search_scope == "full_space"


@pytest.mark.parametrize("method", ["distance", "propagate", "native", "edits"])
def test_numeric_shells_have_labeled_graph_maps_and_smiles_classes(
    method, native_library
):
    options = {"library_path": native_library} if method == "native" else {}
    mapper = Mapper(method, CD=2, binary=True, **options)
    mols, _ = _parse("CCC>>CCC", "heavy")
    graphs = _graphs(mols, "strict", "bond")
    graph_result = mapper.map(graphs)
    result = mapper.map("CCC>>CCC")
    assert graph_result.complete and result.complete
    assert (
        len(graph_result.mappings) == result.reaction_result.labeled_mapping_count == 4
    )
    assert len(result.mapped_reactions) == 1
    assert result.distances == (2.0,)
    assert result.minimum_cost is None and not result.optimality_proven


@pytest.mark.parametrize("method", ["distance", "propagate", "native", "edits"])
def test_deadlines_and_output_caps_never_claim_complete(method, native_library):
    options = {"library_path": native_library} if method == "native" else {}
    if method == "edits":
        options.update(CD=0, binary=True)
    mapper = Mapper(method, **options)
    timed = mapper.map("CCC>>CCC", time_limit_seconds=0)
    assert not timed.complete and timed.minimum_cost is None
    assert timed.incomplete_reason
    capped = mapper.map("CCC>>CCC", max_mappings=1)
    assert not capped.complete and not capped.reaction_result.search_complete
    assert capped.incomplete_reason


@pytest.mark.parametrize("method", ["slap", "kernel", "milp", "exhaustive"])
@pytest.mark.parametrize(
    "option", ["time_limit_seconds", "max_mappings", "max_bijections"]
)
def test_unbounded_methods_reject_unsupported_limits(method, option):
    with pytest.raises(ValueError, match=option):
        Mapper(method).map("CO>>CO", **{option: 1})


def test_reusable_defaults_are_overridden_for_one_call():
    mapper = Mapper(binary=True)
    assert mapper.method == "propagate" and len(mapper.methods()) == 8
    assert mapper.map("CCO>>CC=O").minimum_cost == 0
    assert mapper.map("CCO>>CC=O", binary=False).minimum_cost == 1
    assert mapper.map("CCO>>CC=O").minimum_cost == 0
    assert mapper.map("CO>>CO", search_config=PropagationConfig()).complete


def test_classification_failure_preserves_search_proof(monkeypatch):
    import synkit.Chem.Mapper.reaction_search as chemistry

    monkeypatch.setattr(
        chemistry, "_exact_code", lambda *args, **kwargs: (None, "test_limit")
    )
    result = Mapper().map("CO>>CO")
    assert result.minimum_cost == 0 and result.optimality_proven
    assert not result.complete and result.mapped_reactions == ()
    assert result.incomplete_reason == "classification:test_limit"


@pytest.mark.parametrize(
    "method", ["slap", "kernel", "distance", "propagate", "exhaustive"]
)
def test_shared_dummy_lewis_and_compressed_hydrogen_semantics(method):
    result = Mapper(method).map(
        "O>>[OH-]",
        objective="lewis",
        hydrogen="explicit",
        balance="dummy",
    )
    assert result.distances == (2.0,)
    assert len(result.reaction_result.reactions[0].reactant_only_maps) == 1
    compressed = Mapper(method).map("CO>>[CH2-][OH2+]", hydrogen="compressed")
    assert compressed.distances == (2.0,)
    assert compressed.minimum_cost == (None if method == "slap" else 0)
    assert compressed.reaction_result.minimum_cost_scope == "heavy_bond_distance"


@pytest.mark.parametrize(
    "method,options,message",
    [
        ("edits", {}, "numeric CD"),
        ("edits", {"CD": 0}, "binary=True"),
        ("kernel", {"CD": 2}, "CD='minimal'"),
        ("native", {}, "library_path"),
        ("propagate", {"library_path": "unused"}, "method='native'"),
        ("slap", {"search_config": PropagationConfig()}, "method='propagate'"),
        ("distance", {"max_edit_support_pairs": 10}, "method='edits'"),
    ],
)
def test_incompatible_method_options_fail_clearly(method, options, message):
    with pytest.raises(ValueError, match=message):
        Mapper(method).map("CO>>CO", **options)


def test_invalid_methods_options_and_graph_contract():
    with pytest.raises(ValueError, match="choose from"):
        Mapper("unknown")
    with pytest.raises(TypeError, match="Unknown Mapper option"):
        Mapper(backend="cpp")
    graph = LabeledGraph({0: {}}, [6])
    with pytest.raises(ValueError, match="require reaction SMILES"):
        Mapper().map((graph, graph), hydrogen="explicit")
    with pytest.raises(ValueError, match="loop-free"):
        Mapper().map((LabeledGraph({0: {0: 1}}, [6]), graph))
    with pytest.raises(ValueError, match="symmetric"):
        Mapper().map((LabeledGraph({0: {1: 1}, 1: {}}, [6, 6]),) * 2)
    with pytest.raises(ValueError, match="atom-type"):
        Mapper().map((graph, LabeledGraph({0: {}}, [8])))
