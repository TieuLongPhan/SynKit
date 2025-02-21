import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.Feature.graph_signature import GraphSignature


class TestGraphSignature(unittest.TestCase):

    def setUp(self):
        # Create a sample graph for testing
        data = load_from_pickle("Data/Testcase/test.pkl.gz")
        self.rc = data[0]["GraphRules"][2]
        self.its = data[0]["ITSGraph"][2]
        self.graph_signature = GraphSignature(self.its)

    def test_validate_graph(self):
        """Test the validation of graph structure"""
        # Test should pass if graph is valid (no exceptions)
        try:
            self.graph_signature._validate_graph()
        except ValueError as e:
            self.fail(f"Graph validation failed: {str(e)}")

    def test_edge_signature(self):
        """Test edge signature creation"""
        edge_signature = self.graph_signature.create_edge_signature(
            include_neighbors=False, max_hop=1
        )

        # Check that the edge signature is a non-empty string
        self.assertIsInstance(edge_signature, str)
        self.assertGreater(len(edge_signature), 0)
        self.assertIn("Br0", edge_signature)
        self.assertIn("{0.0,1.0}", edge_signature)
        self.assertIn("H0", edge_signature)

    def test_edge_signature_with_neighbors(self):
        """Test edge signature creation including neighbors"""
        edge_signature_with_neighbors = self.graph_signature.create_edge_signature(
            include_neighbors=True, max_hop=1
        )

        # Check that the edge signature with neighbors includes node degrees and neighbors
        self.assertIsInstance(edge_signature_with_neighbors, str)
        self.assertGreater(len(edge_signature_with_neighbors), 0)
        self.assertIn("d1", edge_signature_with_neighbors)  # node degree for neighbor

    def test_wl_hash(self):
        """Test the Weisfeiler-Lehman hash generation"""
        wl_hash = self.graph_signature.create_wl_hash(iterations=3)

        # Check that the WL hash is a valid hexadecimal string
        self.assertIsInstance(wl_hash, str)
        self.assertRegex(wl_hash, r"^[a-f0-9]{64}$")  # SHA-256 hash format

    def test_graph_signature(self):
        """Test the complete graph signature creation"""
        complete_graph_signature = self.graph_signature.create_graph_signature(
            include_wl_hash=True, include_neighbors=True, max_hop=1
        )

        # Check that the graph signature is a non-empty string
        self.assertIsInstance(complete_graph_signature, str)
        self.assertGreater(len(complete_graph_signature), 0)

    def test_invalid_node_attributes(self):
        """Test for missing node attributes"""
        self.rc.add_node(4)  # Missing 'element' and 'charge'

        with self.assertRaises(ValueError) as context:
            invalid_graph_signature = GraphSignature(self.rc)
            invalid_graph_signature._validate_graph()

        self.assertIn(
            "Node 4 is missing the 'element' attribute", str(context.exception)
        )

    def test_invalid_edge_order(self):
        """Test for invalid edge 'order' attribute"""
        self.its.add_edge(
            3, 4, order="invalid_order", state="steady"
        )  # Invalid 'order' type

        with self.assertRaises(ValueError) as context:
            invalid_graph_signature = GraphSignature(self.its)
            invalid_graph_signature._validate_graph()

        self.assertIn("Edge (3, 4) has an invalid 'order'", str(context.exception))

    def test_invalid_edge_state(self):
        """Test for invalid edge 'state' attribute"""
        self.its.add_edge(2, 4, order=1.0, state="invalid_state")  # Invalid 'state'

        with self.assertRaises(ValueError) as context:
            invalid_graph_signature = GraphSignature(self.its)
            invalid_graph_signature._validate_graph()

        self.assertIn("Edge (2, 4) has an invalid 'state'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
