from rdkit import Chem

from synkit.Graph.FG.ring_system import AromaticRingSystemDetector
from synkit.IO.mol_to_graph import MolToGraph


def _systems(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    graph = MolToGraph().transform(mol)
    return AromaticRingSystemDetector.detect(graph)


def test_detects_single_heteroaromatic_ring():
    system = _systems("n1ccccc1")[0]

    assert system.nodes == (1, 2, 3, 4, 5, 6)
    assert system.hetero_nodes == (1,)
    assert system.element_counts == {"C": 5, "N": 1}
    assert system.ring_sizes == (6,)
    assert system.is_fused is False
    assert system.hetero_pattern == "1N-6ring"


def test_detects_multi_hetero_ring():
    system = _systems("n1ccncc1")[0]

    assert system.element_counts == {"C": 4, "N": 2}
    assert system.ring_sizes == (6,)
    assert system.hetero_pattern == "2N-6ring"


def test_detects_fused_aromatic_system():
    system = _systems("c1ccc2ncccc2c1")[0]

    assert system.element_counts == {"C": 9, "N": 1}
    assert system.ring_sizes == (6, 6)
    assert system.is_fused is True
    assert system.hetero_sequence is None
    assert tuple(ring.nodes for ring in system.subrings) == (
        (1, 2, 3, 4, 9, 10),
        (4, 5, 6, 7, 8, 9),
    )
