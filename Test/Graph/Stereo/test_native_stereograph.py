"""The native graph convention preserves configured identity and mirror semantics."""

from itertools import permutations

import networkx as nx
import pytest

from synkit.Graph.Canon._native import build_native_canon, bundled_native_canon
from synkit.Graph.Stereo import (
    AtropBondStereo, CumuleneAxisStereo, ExtendedCisTransStereo,
    FrameworkFrame, FrameworkStereo, HelicalStereo, OctahedralStereo,
    PlanarBondStereo, PlanarChiralityStereo, SquarePlanarStereo,
    StereographMirrorStatus, TetrahedralStereo, TrigonalBipyramidalStereo,
    canonicalize_configured_stereograph, classify_configured_stereograph_mirror,
    descriptor_id, mirror_configured_descriptor,
)
from synkit.Graph.Stereo.native import canonicalize_native_stereograph, classify_native_stereograph_mirror


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    try:
        return bundled_native_canon()
    except FileNotFoundError:
        pass
    return build_native_canon(tmp_path_factory.mktemp("stereo-native"))


def cases():
    for shape, arity, parity in (
        (TetrahedralStereo, 4, 1), (SquarePlanarStereo, 4, 0),
        (TrigonalBipyramidalStereo, 5, 1), (OctahedralStereo, 6, 1),
    ):
        graph = nx.star_graph(arity)
        yield graph, shape(tuple(range(arity + 1)), parity)
    graph = nx.Graph([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)])
    yield graph, PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)
    yield graph, AtropBondStereo((0, 1, 2, 3, 4, 5), 1)
    for path, shape in (((2, 3, 4), CumuleneAxisStereo), ((2, 3, 4, 5), ExtendedCisTransStereo)):
        graph = nx.path_graph(path)
        graph.add_edges_from([(0, 2), (1, 2), (path[-1], 6), (path[-1], 7)])
        yield graph, shape(path, ((0, 1), (6, 7)), 1 if shape is CumuleneAxisStereo else 0)
    yield nx.path_graph(5), HelicalStereo(tuple(range(5)), 1)
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 5)
    yield graph, PlanarChiralityStereo(tuple(range(5)), 5, 1)
    graph = nx.star_graph(4)
    yield graph, FrameworkStereo(frozenset(graph), (FrameworkFrame(0, (1, 2, 3, 4), 1),), 1)


@pytest.mark.parametrize("graph,descriptor", list(cases()))
def test_all_configured_families_match_reference_relations(library, graph, descriptor):
    nx.set_node_attributes(graph, {v: f"atom:{v}" for v in graph}, "color")
    nx.set_edge_attributes(graph, "bond", "color")
    options = dict(atom_color="color", bond_color="color")
    reference = classify_configured_stereograph_mirror(
        graph, {descriptor_id(descriptor): descriptor}, **options,
    )
    native = classify_native_stereograph_mirror(graph, (descriptor,), library_path=library, **options)
    assert native.is_definitive and native.status == reference.status
    mapping = {v: 17 + 3 * v for v in graph}
    renamed = canonicalize_native_stereograph(
        nx.relabel_nodes(graph, mapping), (descriptor.relabel(mapping),), library_path=library, **options,
    )
    assert native.original.same_stereograph(renamed)
    before = canonicalize_configured_stereograph(graph, (descriptor,), enumerate_automorphism_group=False, **options)
    mirrored = canonicalize_configured_stereograph(
        graph, (mirror_configured_descriptor(descriptor),), enumerate_automorphism_group=False, **options,
    )
    assert native.original.same_stereograph(native.mirror) == before.same_stereograph(mirrored)
    assert native.original.canonical_code != before.canonical_code


def test_complete_local_partitions_not_only_mirror_verdicts(library):
    graph = nx.star_graph(4)
    nx.set_node_attributes(graph, {v: str(v) for v in graph}, "color")
    native_partition, reverse = {}, {}
    for refs in permutations((1, 2, 3, 4)):
        descriptor = TetrahedralStereo((0, *refs), 1)
        native = canonicalize_native_stereograph(graph, (descriptor,), library_path=library, atom_color="color")
        reference = canonicalize_configured_stereograph(
            graph, (descriptor,), atom_color="color", enumerate_automorphism_group=False,
        )
        assert native_partition.setdefault(native.canonical_code, reference.canonical_code) == reference.canonical_code
        assert reverse.setdefault(reference.canonical_code, native.canonical_code) == native.canonical_code
    assert len(native_partition) == 2


def test_mirror_witness_replays_on_a_symmetric_meso_registry(library):
    from rdkit import Chem
    from synkit.Graph.Stereo.canonical import _rdkit_graph_and_registry
    from synkit.Graph.Stereo.matching import mapped_stereo_registries_match

    graph, registry = _rdkit_graph_and_registry(Chem.MolFromSmiles("C[C@H](Br)[C@H](Br)C"))
    result = classify_native_stereograph_mirror(graph, registry, library_path=library)
    assert result.status is StereographMirrorStatus.ACHIRAL
    mapping = dict(result.atom_mirror_isomorphism)
    assert set(mapping) == set(graph) and len(set(mapping.values())) == len(graph)
    assert all(graph.has_edge(mapping[u], mapping[v]) for u, v in graph.edges)
    mirrored = {k: mirror_configured_descriptor(d) for k, d in registry.items()}
    graph.graph["stereo_descriptors"] = registry
    other = graph.copy()
    other.graph["stereo_descriptors"] = mirrored
    assert mapped_stereo_registries_match(graph, other, mapping)


def test_incomplete_and_unsupported_never_get_promoted(library):
    graph = nx.star_graph(4)
    missing = TetrahedralStereo((0, 1, 2, 3, 4), None)
    result = classify_native_stereograph_mirror(graph, (missing,), library_path=library)
    assert not result.is_definitive and result.status is StereographMirrorStatus.INCOMPLETE
    assert result.original.canonical_code is None
    result = classify_native_stereograph_mirror(graph, (object(),), library_path=library)
    assert result.status is StereographMirrorStatus.UNSUPPORTED
    stopped = classify_native_stereograph_mirror(graph, (), library_path=library, timeout_seconds=0)
    assert not stopped.is_definitive and stopped.reason == "time_limit"


def test_profile_and_convention_mismatch_are_not_inequality(library):
    graph = nx.path_graph(3)
    one = canonicalize_native_stereograph(graph, (), library_path=library)
    two = canonicalize_native_stereograph(graph, (), library_path=library, identity_profile="different")
    with pytest.raises(ValueError, match="profiles differ"):
        one.same_stereograph(two)
    reference = canonicalize_configured_stereograph(graph, ())
    with pytest.raises(TypeError, match="conventions"):
        one.same_stereograph(reference)


def test_mirror_deadline_is_shared_and_first_key_is_not_a_verdict(library, monkeypatch):
    import time
    import synkit.Graph.Stereo.native as implementation

    actual = implementation.canonicalize_native_stereograph
    budgets = []

    def delayed(*args, **kwargs):
        budgets.append(kwargs["timeout_seconds"])
        result = actual(*args, **kwargs)
        if len(budgets) == 1:
            time.sleep(0.03)
        return result

    monkeypatch.setattr(implementation, "canonicalize_native_stereograph", delayed)
    result = implementation.classify_native_stereograph_mirror(
        nx.path_graph(3), (), library_path=library, timeout_seconds=0.02,
    )
    assert result.original.complete and not result.mirror.complete
    assert not result.is_definitive and result.reason == "time_limit"
    assert budgets[1] == 0
