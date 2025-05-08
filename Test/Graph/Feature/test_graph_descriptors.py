import unittest
import networkx as nx
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.Feature.graph_descriptors import GraphDescriptor


class TestGraphDescriptor(unittest.TestCase):

    def setUp(self):
        # Creating different types of graphs
        self.acyclic_graph = nx.balanced_tree(
            r=2, h=3
        )  # Creates a balanced binary tree, which is acyclic
        self.single_cyclic_graph = nx.cycle_graph(5)  # Creates a cycle with 5 nodes
        self.complex_cyclic_graph = (
            nx.house_x_graph()
        )  # Known small graph with multiple cycles
        self.empty_graph = nx.Graph()  # Empty graph for testing

        # Set up the graph
        self.graph = nx.Graph()
        self.graph.add_node(
            11,
            element="N",
            charge=0,
            hcount=1,
            aromatic=False,
            atom_map=11,
            isomer="N",
            partial_charge=-0.313,
            hybridization="SP3",
            in_ring=True,
            explicit_valence=3,
            implicit_hcount=0,
            neighbors=["C", "C"],
        )
        self.graph.add_node(
            35,
            element="H",
            charge=0,
            hcount=0,
            aromatic=False,
            atom_map=11,
            isomer="N",
            partial_charge=0,
            hybridization="0",
            in_ring=False,
            explicit_valence=0,
            implicit_hcount=0,
        )
        self.graph.add_node(
            28,
            element="C",
            charge=0,
            hcount=0,
            aromatic=True,
            atom_map=28,
            isomer="N",
            partial_charge=0.063,
            hybridization="SP2",
            in_ring=True,
            explicit_valence=4,
            implicit_hcount=0,
            neighbors=["Br", "C", "C"],
        )
        self.graph.add_node(
            29,
            element="Br",
            charge=0,
            hcount=0,
            aromatic=False,
            atom_map=29,
            isomer="N",
            partial_charge=-0.047,
            hybridization="SP3",
            in_ring=False,
            explicit_valence=1,
            implicit_hcount=0,
            neighbors=["C"],
        )

        # Adding edges with their attributes
        self.graph.add_edge(11, 35, order=(1.0, 0), standard_order=1.0)
        self.graph.add_edge(11, 28, order=(0, 1.0), standard_order=-1.0)
        self.graph.add_edge(35, 29, order=(0, 1.0), standard_order=-1.0)
        self.graph.add_edge(28, 29, order=(1.0, 0), standard_order=1.0)
        # Prepare the data dictionary
        self.data = {"RC": self.graph, "ITS": self.graph}

        self.data_parallel = load_from_pickle("Data/Testcase/test.pkl.gz")

    def test_is_acyclic_graph(self):
        self.assertTrue(GraphDescriptor.is_acyclic_graph(self.acyclic_graph))
        self.assertFalse(GraphDescriptor.is_acyclic_graph(self.single_cyclic_graph))
        self.assertFalse(GraphDescriptor.is_acyclic_graph(self.complex_cyclic_graph))
        self.assertFalse(GraphDescriptor.is_acyclic_graph(self.empty_graph))

    def test_is_single_cyclic_graph(self):
        self.assertFalse(GraphDescriptor.is_single_cyclic_graph(self.acyclic_graph))
        self.assertTrue(
            GraphDescriptor.is_single_cyclic_graph(self.single_cyclic_graph)
        )
        self.assertFalse(
            GraphDescriptor.is_single_cyclic_graph(self.complex_cyclic_graph)
        )
        self.assertFalse(GraphDescriptor.is_single_cyclic_graph(self.empty_graph))

    def test_is_complex_cyclic_graph(self):
        self.assertFalse(GraphDescriptor.is_complex_cyclic_graph(self.acyclic_graph))
        self.assertFalse(
            GraphDescriptor.is_complex_cyclic_graph(self.single_cyclic_graph)
        )
        self.assertTrue(
            GraphDescriptor.is_complex_cyclic_graph(self.complex_cyclic_graph)
        )
        self.assertFalse(GraphDescriptor.is_complex_cyclic_graph(self.empty_graph))

    def test_check_graph_type(self):
        self.assertEqual(
            GraphDescriptor.check_graph_type(self.acyclic_graph), "Acyclic"
        )
        self.assertEqual(
            GraphDescriptor.check_graph_type(self.single_cyclic_graph), "Single Cyclic"
        )
        self.assertEqual(
            GraphDescriptor.check_graph_type(self.complex_cyclic_graph),
            "Combinatorial Cyclic",
        )
        self.assertEqual(
            GraphDescriptor.check_graph_type(self.empty_graph), "Empty Graph"
        )

    def test_get_cycle_member_rings(self):
        self.assertEqual(GraphDescriptor.get_cycle_member_rings(self.acyclic_graph), [])
        self.assertEqual(
            GraphDescriptor.get_cycle_member_rings(self.single_cyclic_graph), [5]
        )
        self.assertEqual(
            GraphDescriptor.get_cycle_member_rings(self.complex_cyclic_graph),
            [3, 3, 3, 3],
        )
        self.assertEqual(GraphDescriptor.get_cycle_member_rings(self.empty_graph), [])

    def test_get_element_count(self):
        # Expected results
        expected_element_count = {"N": 1, "H": 1, "C": 1, "Br": 1}

        # Test get_element_count
        self.assertEqual(
            GraphDescriptor.get_element_count(self.graph), expected_element_count
        )

    def test_get_descriptors(self):
        # Expected output after processing
        expected_output = {
            "RC": self.graph,
            "topo": "Single Cyclic",  # Adjust based on expected graph type analysis
            "cycle": [
                4
            ],  # Expected cycle results, to be filled after actual function implementation
            "atom_count": {"N": 1, "H": 1, "C": 1, "Br": 1},
            "rtype": "Elementary",  # Expected reaction type
            "rstep": 1,  # This should be based on the actual cycles count
        }

        # Run the descriptor function
        results = GraphDescriptor.get_descriptors(self.data, "RC")
        self.assertEqual(results["topo"], expected_output["topo"])
        self.assertEqual(results["cycle"], expected_output["cycle"])
        self.assertEqual(results["rstep"], expected_output["rstep"])
        self.assertEqual(results["atom_count"], expected_output["atom_count"])

    def test_get_descriptors_parallel(self):
        # Expected output after processing
        expected_output = {
            "RC": self.graph,
            "topo": "Single Cyclic",
            "cycle": [4],
            "atom_count": {"N": 1, "H": 1, "C": 1, "Br": 1},
            "rtype": "Elementary",
            "rstep": 1,
        }

        # Run the descriptor function
        results = GraphDescriptor.process_entries_in_parallel(
            self.data_parallel, "GraphRules", "ITSGraph", n_jobs=1
        )
        self.assertEqual(results[0]["topo"], expected_output["topo"])
        self.assertEqual(results[0]["cycle"], expected_output["cycle"])
        self.assertEqual(results[0]["rstep"], expected_output["rstep"])
        self.assertEqual(results[0]["atom_count"], expected_output["atom_count"])


if __name__ == "__main__":
    unittest.main()
