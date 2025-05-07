import time
import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.Feature.graph_signature import GraphSignature
from synkit.Graph.Matcher.batch_cluster import BatchCluster


class TestBatchCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")
        cls.templates = None
        for value in cls.graphs:
            value["rc_sig"] = GraphSignature(value["RC"]).create_graph_signature()
            value["its_sig"] = GraphSignature(value["ITS"]).create_graph_signature()

    def test_initialization(self):
        """Test initialization and verify if the attributes are set correctly."""
        cluster = BatchCluster(["element", "charge"], ["*", 0], "bond_order")
        self.assertEqual(cluster.nodeLabelNames, ["element", "charge"])
        self.assertEqual(cluster.nodeLabelDefault, ["*", 0])
        self.assertEqual(cluster.edgeAttribute, "bond_order")

    def test_initialization_failure(self):
        """Test initialization failure when lengths of node labels and defaults do not match."""
        with self.assertRaises(ValueError):
            BatchCluster(["element"], ["*", 0, 1], "bond_order")

    def test_batch_dicts(self):
        """Test the batching function to split data correctly."""
        batch_cluster = BatchCluster(["element", "charge"], ["*", 0], "bond_order")
        input_list = [{"id": i} for i in range(10)]
        batches = batch_cluster.batch_dicts(input_list, 3)
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[-1]), 1)

    def test_lib_check_functionality(self):
        """Test the lib_check method using directly comparable results."""
        cluster = BatchCluster()
        batch_1 = self.graphs[:50]
        batch_2 = self.graphs[50:]
        _, templates = cluster.fit(batch_1, None, "RC", "rc_sig")
        for entry in batch_2:
            _, templates = cluster.lib_check(entry, templates, "RC", "rc_sig")
        self.assertEqual(len(templates), 30)

    def test_cluster_integration(self):
        """Test the cluster method to ensure it processes data entries correctly."""
        cluster = BatchCluster()
        expected_template_count = 30
        _, updated_templates = cluster.cluster(self.graphs, [], "RC", "rc_sig")

        self.assertEqual(
            len(updated_templates),
            expected_template_count,
            f"Failed: expected {expected_template_count} templates, got {len(updated_templates)}",
        )

    def test_fit(self):
        cluster = BatchCluster()
        batch_sizes = [None, 10]
        expected_template_count = 30

        for batch_size in batch_sizes:
            start_time = time.time()
            _, updated_templates = cluster.fit(
                self.graphs, self.templates, "RC", "rc_sig", batch_size=batch_size
            )
            elapsed_time = time.time() - start_time

            self.assertEqual(
                len(updated_templates),
                expected_template_count,
                f"Failed for batch_size={batch_size}: expected "
                + f"{expected_template_count} templates, got {len(updated_templates)}",
            )
            print(
                f"Test for batch_size={batch_size} completed in {elapsed_time:.2f} seconds."
            )

    def test_fit_gml(self):
        cluster = BatchCluster()
        batch_sizes = [None, 10]
        expected_template_count = (
            30  # Assuming this is the expected number of templates after processing
        )

        for batch_size in batch_sizes:
            start_time = time.time()
            _, updated_templates = cluster.fit(
                self.graphs, self.templates, "RC", "rc_sig", batch_size=batch_size
            )
            elapsed_time = time.time() - start_time

            self.assertEqual(
                len(updated_templates),
                expected_template_count,
                f"Failed for batch_size={batch_size}: expected"
                + f" {expected_template_count} templates, got {len(updated_templates)}",
            )
            print(
                f"Test for batch_size={batch_size} completed in {elapsed_time:.2f} seconds."
            )


# To run the tests
if __name__ == "__main__":
    unittest.main()
