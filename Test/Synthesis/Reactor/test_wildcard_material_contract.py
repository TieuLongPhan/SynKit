"""Material typing and explicit admission limits of the wildcard profile."""

import networkx as nx
import pytest

from synkit.IO import smiles_to_graph
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return str(build_native_reactor(tmp_path_factory.mktemp("material-contract")))


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_concrete_atomic_valence_label_is_preserved_by_binding(backend, library):
    host = smiles_to_graph("C")
    # Keep the local charge equation true. A charge guard cannot detect this
    # conflicting atomic label because the extra lone pair cancels it.
    host.nodes[1].update(valence_electrons=6, lone_pairs=1)
    engine = WildcardEngine.from_lewis_graph(
        host,
        "[CH4:1]>>[CH4:1]",
        partial=False,
        backend=backend,
        library_path=library,
    )
    assert engine.mappings == []
    assert engine.its_list == []


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "host,template",
    [
        ("C", "[13CH4:1]>>[13CH4:1]"),
        ("[13CH4]", "[CH4:1]>>[CH4:1]"),
        ("C", "[CH4:1]>>[13CH4:1]"),
    ],
)
def test_isotope_labels_cannot_be_silently_dropped(backend, host, template):
    # Admission precedes native loading, including target-only labels.
    engine = WildcardEngine(host, template, partial=False, backend=backend)
    with pytest.raises(ValueError, match="[Ii]sotope"):
        _ = engine.its_list
    assert engine._its is None and engine._native_cache is None


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_preconstructed_tuple_rule_retains_isotope_admission_information(backend):
    rule = SynRule.from_smart("[13CH4:1]>>[13CH4:1]", format="tuple", canon=False)
    assert rule.rc.raw.nodes[1]["isotope"] == (13, 13)
    engine = WildcardEngine("C", rule, partial=False, backend=backend)
    with pytest.raises(ValueError, match="[Ii]sotope"):
        _ = engine.mappings
    assert engine._rule is None


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_explicit_isotope_identity_on_graph_hosts_is_refused(backend):
    graph = smiles_to_graph("C")
    graph.nodes[1]["isotope"] = 13
    engine = WildcardEngine(graph, "[CH4:1]>>[CH4:1]", partial=False, backend=backend)
    with pytest.raises(ValueError, match="[Ii]sotope"):
        _ = engine.its_list


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("location", ["host", "rule"])
def test_looped_carriers_are_outside_the_simple_graph_profile(backend, location):
    host = smiles_to_graph("C")
    template = SynRule.from_smart("[CH4:1]>>[CH4:1]", format="tuple", canon=False)
    if location == "host":
        host.add_edge(1, 1, order=1, sigma_order=1, pi_order=0)
    else:
        for endpoint in (template.left.raw, template.right.raw):
            endpoint.add_edge(1, 1, order=1, sigma_order=1, pi_order=0)
    before = host.copy()
    engine = WildcardEngine(host, template, partial=False, backend=backend)
    with pytest.raises(ValueError, match="simple|loop"):
        _ = engine.its_list
    assert nx.utils.graphs_equal(host, before)
    assert engine._its is None and engine._native_cache is None
