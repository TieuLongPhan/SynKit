import unittest
import networkx as nx
from copy import deepcopy
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.Hyrogen.hcomplete import HComplete


class TestHComplete(unittest.TestCase):

    def setUp(self):
        """Setup before each test."""
        # Create sample graphs
        self.data = load_from_pickle("./Data/Testcase/hydro/hydrogen_test.pkl.gz")

    def test_process_single_graph_data_success(self):
        """Test the process_single_graph_data method."""
        processed_data = HComplete.process_single_graph_data(self.data[0], "ITS", "RC")
        self.assertTrue(isinstance(processed_data["ITS"], nx.Graph))
        self.assertTrue(isinstance(processed_data["RC"], nx.Graph))

    def test_process_single_graph_data_fail(self):
        """Test the process_single_graph_data method."""
        processed_data = HComplete.process_single_graph_data(self.data[16], "ITS", "RC")
        self.assertIsNone(processed_data["ITS"])
        self.assertIsNone(processed_data["RC"])

    def test_process_single_graph_data_empty_graph(self):
        """Test that an empty graph results in empty ITSGraph and GraphRules."""
        empty_graph_data = {
            "ITS": None,
            "RC": None,
        }

        processed_data = HComplete.process_single_graph_data(
            empty_graph_data, "ITSGraph"
        )

        # Ensure the result is None or empty as expected for an empty graph
        self.assertIsNone(processed_data["ITS"])
        self.assertIsNone(processed_data["RC"])

    def test_process_graph_data_parallel(self):
        """Test the process_graph_data_parallel method."""
        result = HComplete().process_graph_data_parallel(
            self.data,
            "ITS",
            "RC",
            n_jobs=1,
            verbose=0,
        )
        result = [value for value in result if value["ITS"]]
        # Check if the result matches the input data structure
        self.assertEqual(len(result), 45)  # 45 valid graphs

    def test_process_multiple_hydrogens(self):
        """Test the process_multiple_hydrogens method."""
        graphs = deepcopy(self.data[0])
        its = graphs["ITS"]
        react_graph, prod_graph = its_decompose(its)

        result = HComplete.process_multiple_hydrogens(
            graphs,
            "ITS",
            "RC",
            react_graph,
            prod_graph,
            ignore_aromaticity=False,
            balance_its=True,
        )

        self.assertTrue(isinstance(result["ITS"], nx.Graph))
        self.assertTrue(isinstance(result["RC"], nx.Graph))


if __name__ == "__main__":
    unittest.main()
