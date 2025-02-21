import unittest
from rdkit import Chem
import networkx as nx
from synkit.IO.mol_to_graph import MolToGraph


class TestMolToGraph(unittest.TestCase):
    def setUp(self):
        self.converter = MolToGraph()
        # Example molecule: Ethanol
        self.ethanol_smiles = "CCO"
        self.mol = Chem.MolFromSmiles(self.ethanol_smiles)

    def test_add_partial_charges(self):
        MolToGraph.add_partial_charges(self.mol)
        for atom in self.mol.GetAtoms():
            self.assertTrue(atom.HasProp("_GasteigerCharge"))

    def test_get_stereochemistry(self):
        # Test with chiral molecule
        chiral_smiles = "CC[C@@H](C)O"
        chiral_mol = Chem.MolFromSmiles(chiral_smiles)
        chiral_atom = chiral_mol.GetAtomWithIdx(2)  # The chiral carbon
        stereo = MolToGraph.get_stereochemistry(chiral_atom)
        self.assertIn(stereo, ["R", "S"])

    def test_get_bond_stereochemistry(self):
        # Test with E-stilbene
        e_stilbene_smiles = "C/C=C/C"
        e_stilbene_mol = Chem.MolFromSmiles(e_stilbene_smiles)
        double_bond = e_stilbene_mol.GetBondWithIdx(1)  # The double bond
        stereo = MolToGraph.get_bond_stereochemistry(double_bond)
        self.assertIn(stereo, ["E", "Z", "N"])

    def test_mol_to_graph(self):
        graph = self.converter.mol_to_graph(self.mol)
        self.assertIsInstance(graph, nx.Graph)
        # Check for expected number of nodes and edges
        self.assertEqual(len(graph.nodes), self.mol.GetNumAtoms())
        self.assertEqual(len(graph.edges), self.mol.GetNumBonds())
        # Check attributes of an arbitrary atom and bond
        some_atom = list(graph.nodes(data=True))[0]
        some_edge = list(graph.edges(data=True))[0]
        self.assertIn("element", some_atom[1])
        self.assertIn("order", some_edge[2])


if __name__ == "__main__":
    unittest.main()
