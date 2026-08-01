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

    def test_process_single_graph_data_rejects_ambiguous_transfer(self):
        """Different heavy-atom H transfers must not collapse to H relabeling."""
        processed_data = HComplete.process_single_graph_data(self.data[16], "ITS", "RC")
        completion = HComplete.complete_its(self.data[16]["ITS"])
        unique_rc, _, _ = HExtend._extend_unique(self.data[16]["ITS"])

        self.assertIsNone(processed_data["ITS"])
        self.assertIsNone(processed_data["RC"])
        self.assertFalse(completion.ok)
        self.assertEqual(completion.reason, "non_equivariant_rc")
        self.assertEqual(len(unique_rc), 2)

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
        self.assertEqual(len(result), 45)

    def test_candidate_limit_does_not_claim_exhaustive_completion(self):
        """A truncated unique-looking prefix must report an indeterminate result."""
        result = HComplete.complete_its(self.data[16]["ITS"], max_candidates=1)

        self.assertFalse(result.ok)
        self.assertFalse(result.exhaustive)
        self.assertEqual(result.reason, "max_candidates_reached")

    def test_tuple_completion_respects_complete_lewis_state(self):
        """Ignored electron labels must not make transfer plans equivalent."""

        def side(hcounts):
            graph = nx.Graph()
            for node in range(1, 5):
                graph.add_node(
                    node,
                    element="C",
                    aromatic=False,
                    hcount=hcounts[node - 1],
                    charge=0,
                    lone_pairs=node - 1,
                    radical=0,
                    valence_electrons=4,
                    atom_map=node,
                )
            return graph

        result = HComplete._complete_from_side_graphs(
            side([1, 1, 0, 0]),
            side([0, 0, 1, 1]),
            ignore_aromaticity=False,
            balance_its=True,
            get_priority_graph=False,
            format="tuple",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "non_equivariant_rc")
        self.assertEqual(result.candidates, 2)

    def test_explicit_h2_provenance_is_preserved(self):
        """Explicit H2 atoms remain distinct from anonymous implicit-H slots."""
        data = load_from_pickle("./Experiment/Lewis/Data/hydrogen.pkl.gz")
        for fmt, its in (
            ("typesGH", data[26]["ITS"]),
            ("tuple", rsmi_to_its(data[26]["aam"], format="tuple")),
        ):
            with self.subTest(format=fmt):
                result = HComplete.complete_its(its, format=fmt)
                unique_rc, _, _ = HExtend._extend_unique(its, format=fmt)

                self.assertFalse(result.ok)
                self.assertEqual(result.reason, "non_equivariant_rc")
                self.assertEqual(len(unique_rc), 2)

    def test_typesgh_plan_projection_equals_materialized_rc_projection(self):
        """Plan-native RC projection is definitionally equal after construction."""
        data = load_from_pickle("./Experiment/Lewis/Data/hydrogen.pkl.gz")
        for fixture in (self.data[16], data[26]):
            react_graph, prod_graph = its_decompose(fixture["ITS"])
            for plan in HComplete._iter_hydrogen_transfer_plans(
                react_graph, prod_graph
            ):
                direct = HComplete._typesgh_plan_comparison_graph(
                    react_graph,
                    prod_graph,
                    plan,
                    ignore_aromaticity=False,
                )
                _, _, _, rc, _ = HComplete._materialize_transfer_candidate(
                    react_graph,
                    prod_graph,
                    plan,
                    ignore_aromaticity=False,
                    balance_its=True,
                    format="typesGH",
                )
                materialized = HComplete._comparison_graph(rc, "typesGH")
                self.assertTrue(nx.utils.graphs_equal(direct, materialized))

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

    def test_iter_unique_completions_streams_exact_classes(self):
        """The streaming API yields the same exact representatives as the wrapper."""
        its = self.data[16]["ITS"]
        streamed = list(HExtend.iter_unique_completions(its))
        unique_rc, unique_its, unique_sig = HExtend._extend_unique(its)

        self.assertEqual(len(streamed), 2)
        self.assertEqual([item[2] for item in streamed], unique_sig)
        self.assertTrue(
            all(
                nx.is_isomorphic(item[0], rc)
                and nx.is_isomorphic(item[1], completed_its)
                for item, rc, completed_its in zip(streamed, unique_rc, unique_its)
            )
        )

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
