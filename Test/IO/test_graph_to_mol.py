import unittest
from rdkit import Chem
import networkx as nx
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph


class TestGraphToMol(unittest.TestCase):
    def setUp(self):
        # Define node and edge attributes mappings
        self.node_attributes = {"element": "element", "charge": "charge"}
        self.edge_attributes = {"order": "order"}
        self.converter = GraphToMol(self.node_attributes, self.edge_attributes)

    def test_simple_molecule_conversion(self):
        # Create a simple water molecule graph
        graph = nx.Graph()
        graph.add_node(0, element="O", charge=0)
        graph.add_node(1, element="H", charge=0)
        graph.add_node(2, element="H", charge=0)
        graph.add_edges_from([(0, 1), (0, 2)], order=1)

        mol = self.converter.graph_to_mol(graph)
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        self.assertEqual(smiles, "O")

    def test_bond_order_handling(self):
        # Create a graph representing ethene (C=C)
        graph = nx.Graph()
        graph.add_node(0, element="C", charge=0)
        graph.add_node(1, element="C", charge=0)
        graph.add_edge(0, 1, order=2)

        mol = self.converter.graph_to_mol(graph)
        self.assertEqual(Chem.MolToSmiles(mol), "C=C")

    def test_ignore_bond_order(self):
        # Create a graph representing ethene (C=C) but ignore bond order
        graph = nx.Graph()
        graph.add_node(0, element="C", charge=0)
        graph.add_node(1, element="C", charge=0)
        graph.add_edge(0, 1, order=2)

        mol = self.converter.graph_to_mol(graph, ignore_bond_order=True)
        self.assertEqual(Chem.MolToSmiles(mol), "CC")

    def test_molecule_with_charges(self):
        # Create a graph representing a charged molecule [NH4+]
        graph = nx.Graph()
        graph.add_node(0, element="N", charge=1)
        for i in range(1, 5):
            graph.add_node(i, element="H", charge=0)
            graph.add_edge(0, i, order=1)

        mol = self.converter.graph_to_mol(graph)
        self.assertEqual(Chem.CanonSmiles(Chem.MolToSmiles(mol)), "[NH4+]")

    def test_molecule_with_radical(self):
        graph = nx.Graph()
        graph.add_node(0, element="C", charge=0, radical=1)

        mol = self.converter.graph_to_mol(graph, sanitize=False)

        self.assertEqual(mol.GetAtomWithIdx(0).GetNumRadicalElectrons(), 1)

    def test_aromatic_order_is_reperceived_on_sanitized_output(self):
        graph = nx.cycle_graph(6)
        nx.set_node_attributes(graph, "C", "element")
        nx.set_node_attributes(graph, 0, "charge")
        nx.set_edge_attributes(graph, 1.5, "order")

        mol = self.converter.graph_to_mol(graph)

        self.assertEqual(Chem.MolToSmiles(mol), "c1ccccc1")
        self.assertTrue(all(bond.GetIsAromatic() for bond in mol.GetBonds()))

    def test_retained_kekule_order_rebuilds_charged_heteroaromatic_system(self):
        smiles = "C[N+]1=NOC(=C1SCC(=O)NC2=CC3=C(C=C2)NC=C3)[O-]"
        source = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(source)

        graph = MolToGraph().transform(source)
        rebuilt = GraphToMol().graph_to_mol(graph)

        self.assertEqual(
            Chem.MolToSmiles(rebuilt, canonical=True),
            Chem.MolToSmiles(source, canonical=True),
        )

    def test_sigma_pi_order_is_fallback_for_aromatic_presentation_order(self):
        graph = nx.cycle_graph(6)
        nx.set_node_attributes(graph, "C", "element")
        nx.set_node_attributes(graph, 0, "charge")
        double_edges = {frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))}
        for left, right, data in graph.edges(data=True):
            data["order"] = 1.5
            data["sigma_order"] = 1.0
            data["pi_order"] = float(frozenset((left, right)) in double_edges)

        mol = self.converter.graph_to_mol(graph)

        self.assertEqual(Chem.MolToSmiles(mol), "c1ccccc1")

    def test_empty_electron_fields_do_not_erase_aromatic_bonds(self):
        graph = nx.cycle_graph(6)
        nx.set_node_attributes(graph, "C", "element")
        nx.set_node_attributes(graph, 0, "charge")
        nx.set_edge_attributes(graph, 1.5, "order")
        nx.set_edge_attributes(graph, 0.0, "kekule_order")
        nx.set_edge_attributes(graph, 0.0, "sigma_order")
        nx.set_edge_attributes(graph, 0.0, "pi_order")

        mol = self.converter.graph_to_mol(graph)

        self.assertEqual(Chem.MolToSmiles(mol), "c1ccccc1")


if __name__ == "__main__":
    unittest.main()
