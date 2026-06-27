import unittest
import networkx as nx
from copy import deepcopy
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import its_to_rsmi, rsmi_to_its
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.Hyrogen.hcomplete import HComplete
from synkit.Graph.Hyrogen.hextend import HExtend, cluster


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

    def test_complete_its_accepts_graph_input(self):
        """Test graph-first hydrogen completion without a dictionary wrapper."""
        result = HComplete.complete_its(self.data[0]["ITS"])
        self.assertTrue(result.ok)
        self.assertTrue(isinstance(result.its, nx.Graph))
        self.assertTrue(isinstance(result.rc, nx.Graph))

    def test_process_single_graph_data_reference_valid_case(self):
        """Test a fixture that is valid under reference-style H expansion."""
        processed_data = HComplete.process_single_graph_data(self.data[16], "ITS", "RC")
        self.assertTrue(isinstance(processed_data["ITS"], nx.Graph))
        self.assertTrue(isinstance(processed_data["RC"], nx.Graph))

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
        self.assertEqual(len(result), 50)  # all fixtures match reference expansion

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

    def test_complete_its_tuple_format(self):
        """Test graph-first completion for tuple ITS graphs."""
        rsmi = "[CH3:1][O:2]>>[CH2:1]=[O:2].[H:3]"
        its = rsmi_to_its(rsmi, format="tuple")

        result = HComplete.complete_its(its)

        self.assertTrue(result.ok)
        self.assertEqual(result.format, "tuple")
        node_attrs = next(iter(result.its.nodes(data=True)))[1]
        self.assertTrue(isinstance(node_attrs["element"], tuple))

    def test_complete_its_roundtrip_inferred_hydrogen_map(self):
        """Test inferred hydrogens get their own map in both ITS formats."""
        rsmi = (
            "[CH3:1][O:2][C:3](=[O:4])[CH2:5][c:6]1[cH:7][cH:8]"
            "[c:9]([OH:10])[cH:11][cH:12]1.[Cl:13][CH2:14][c:15]1"
            "[cH:16][cH:17][cH:18][cH:19][cH:20]1>>"
            "[CH3:1][O:2][C:3](=[O:4])[CH2:5][c:6]1[cH:7][cH:8]"
            "[c:9]([O:10][CH2:14][c:15]2[cH:16][cH:17][cH:18]"
            "[cH:19][cH:20]2)[cH:11][cH:12]1.[ClH:13]"
        )
        expected = (
            "[CH3:1][O:2][C:3](=[O:4])[CH2:5][c:6]1[cH:7][cH:8]"
            "[c:9]([O:10][H:21])[cH:11][cH:12]1.[Cl:13][CH2:14]"
            "[c:15]1[cH:16][cH:17][cH:18][cH:19][cH:20]1>>"
            "[CH3:1][O:2][C:3](=[O:4])[CH2:5][c:6]1[cH:7][cH:8]"
            "[c:9]([O:10][CH2:14][c:15]2[cH:16][cH:17][cH:18]"
            "[cH:19][cH:20]2)[cH:11][cH:12]1.[Cl:13][H:21]"
        )

        for fmt in ("typesGH", "tuple"):
            with self.subTest(format=fmt):
                its = rsmi_to_its(rsmi, format=fmt)
                result = HComplete.complete_its(its, format=fmt)
                expanded = its_to_rsmi(result.its, format=fmt)

                self.assertTrue(result.ok)
                self.assertEqual(expanded, expected)
                self.assertNotIn("[H:10]", expanded)

    def test_complete_its_reuses_product_hydrogen_id(self):
        """Test product-side explicit H IDs are reused for broken hydrogens."""
        rsmi = "[CH3:1][O:2]>>[CH2:1]=[O:2].[H:3]"
        expected = "[CH2:1]([O:2])[H:3]>>[CH2:1]=[O:2].[H:3]"

        for fmt in ("typesGH", "tuple"):
            with self.subTest(format=fmt):
                result = HComplete.complete_its(rsmi_to_its(rsmi, format=fmt))
                expanded = its_to_rsmi(result.its, format=fmt)

                self.assertTrue(result.ok)
                self.assertEqual(expanded, expected)
                self.assertNotIn("[*:", expanded)

    def test_extend_its_accepts_tuple_graph_input(self):
        """Test HExtend can enumerate tuple ITS completions directly."""
        rsmi = "[C:1]=[C:2].[H:3][H:4]>>[CH:1][CH:2]"
        its = rsmi_to_its(rsmi, format="tuple")

        rc_list, its_list, sigs = HExtend.extend_its(its, max_candidates=1)

        self.assertEqual(len(rc_list), 1)
        self.assertEqual(len(its_list), 1)
        self.assertEqual(len(sigs), 1)

    def test_extend_unique_matches_full_cluster_count(self):
        """Test HExtend fast path keeps full enumeration clustering semantics."""
        rsmi = (
            "[CH3:1][O:2][C:3](=[O:4])[CH2:5][c:6]1[cH:7][cH:8]"
            "[c:9]([OH:10])[cH:11][cH:12]1.[Cl:13][CH2:14][c:15]1"
            "[cH:16][cH:17][cH:18][cH:19][cH:20]1>>"
            "[CH3:1][O:2][C:3](=[O:4])[CH2:5][c:6]1[cH:7][cH:8]"
            "[c:9]([O:10][CH2:14][c:15]2[cH:16][cH:17][cH:18]"
            "[cH:19][cH:20]2)[cH:11][cH:12]1.[ClH:13]"
        )

        for fmt in ("typesGH", "tuple"):
            with self.subTest(format=fmt):
                its = rsmi_to_its(rsmi, format=fmt)
                full_rc, _, full_sig = HExtend.extend_its(its, format=fmt)
                full_cmp = [HComplete._comparison_graph(rc, fmt) for rc in full_rc]
                full_clusters, _ = cluster.iterative_cluster(full_cmp, full_sig)
                unique_rc, unique_its, _ = HExtend._extend_unique(its, format=fmt)

                self.assertEqual(len(unique_rc), len(full_clusters))
                self.assertEqual(len(unique_its), len(full_clusters))

    def test_get_unique_graphs_for_clusters_is_deterministic(self):
        """Test cluster representatives use the smallest index."""
        graphs = []
        for idx in range(3):
            graph = nx.Graph()
            graph.graph["idx"] = idx
            graphs.append(graph)

        selected = HExtend.get_unique_graphs_for_clusters(graphs, [{2, 0}, {1}])

        self.assertEqual([graph.graph["idx"] for graph in selected], [0, 1])


if __name__ == "__main__":
    unittest.main()
