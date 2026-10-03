"""Independent RotA truth covers every frozen category and records chemistry limits."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import pytest

from Experiment.Stereo.Canonicalization.independent_model import State
from Experiment.Stereo.ChemicalValidation.rota_annotation import (
    _reference, annotate, source_inventory, terminal_exchange_truth,
)

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def records():
    path = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/rota_adjudication_selection.json"
    return json.loads(path.read_text())["records"]


def test_all_sixty_frozen_loci_have_independently_validated_inventories(records):
    families, aromatic = Counter(), 0
    for record in records:
        _graph, seeds, family, facts = source_inventory(record)
        families[family] += 1
        assert len(seeds) == (2 if family == "tetrahedral_pair" else 1)
        if family == "axis":
            assert not facts["geometric_axis_certified"]
            aromatic += not facts["ordinary_single_bond_topology"]
    assert families == {"axis": 52, "tetrahedral_pair": 4, "tetrahedral_spiro": 4}
    assert aromatic == 7


@pytest.mark.parametrize("mutation", ["atom_count", "locus", "ports", "duplicate_port"])
def test_corrupt_frozen_axis_information_is_rejected(records, mutation):
    record = deepcopy(records[0])
    if mutation == "atom_count":
        record["atom_count"] += 1
    elif mutation == "locus":
        record["locus"] = "10000-10001"
    elif mutation == "ports":
        record["input_supports"][0]["support"]["terminal_frames"][0][0] = 10
    else:
        frames = record["input_supports"][0]["support"]["terminal_frames"]
        frames[0][1] = frames[0][0]
    with pytest.raises(ValueError, match="frozen_rota"):
        source_inventory(record)


def test_an_acyclic_four_port_carbon_does_not_certify_a_spiro_locus():
    record = {"smiles": "CC(C)(C)C", "atom_count": 5, "locus": "1", "input_supports": [],
              "source_category": "Spiral atom and chain"}
    with pytest.raises(ValueError, match="spiro_cycle_intersection"):
        source_inventory(record)


def test_nitrogen_atom_pair_preserves_both_owner_scoped_electronic_ports(records):
    record = next(value for value in records if value["source_id"] == "RotA-0256")
    graph, seeds, family, _facts = source_inventory(record)
    nitrogen = next(state for state in seeds if state.atoms[0] == 8)
    assert family == "tetrahedral_pair"
    assert nitrogen.atoms[-2:] == ("@H:8", "@LP:8")
    assert graph.nodes[8]["hcount"] == graph.nodes[8]["lone_pairs"] == 1
    reference = _reference(graph, seeds)
    assert len(reference["assignments"]) == len(reference["partition"]) == 4


def test_a_terminal_exchange_can_preserve_global_axial_orientation():
    """A coupled swap cannot be advertised as a whole-configuration mirror map."""
    graph = nx.Graph()
    graph.add_edges_from(((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (3, 5)), color="bond")
    nx.set_node_attributes(graph, {0: "left", 1: "right", 2: "A", 3: "A", 4: "B", 5: "B"}, "color")
    seed = State("atrop_bond", (2, 3, 0, 1, 4, 5), 1)
    terminal = terminal_exchange_truth(graph, seed)
    assert terminal["carrier_status"] == "symmetry_related"
    mapping = dict(terminal["terminal_exchange_witnesses"][0])
    assert mapping[2] == 3 and mapping[4] == 5
    assert len(_reference(graph, (seed,))["partition"]) == 2
    from Experiment.Stereo.Canonicalization.independent_model import to_descriptor
    from synkit.Graph.Stereo import canonicalize_configured_stereograph, mirror_configured_descriptor

    descriptor = to_descriptor(seed)
    options = {"atom_color": "color", "bond_color": "color", "enumerate_automorphism_group": False}
    source = canonicalize_configured_stereograph(graph, (descriptor,), **options)
    reflected = canonicalize_configured_stereograph(graph, (mirror_configured_descriptor(descriptor),), **options)
    assert source.complete and reflected.complete
    assert source.canonical_code != reflected.canonical_code


def test_spiro_symmetry_truth_keeps_experimental_persistence_unknown(records):
    record = next(value for value in records if value["source_category"] == "Spiral atom and chain")
    result = annotate(record)
    assert result["status"] == "complete"
    assert result["configured_class_count"] == 1
    assert result["constitutional_carrier"]["center_statuses"] == ["symmetry_related"]
    assert not result["algorithm_confirmed_rejected_status_read"]
    assert result["experimental_persistence"].startswith("unknown")
