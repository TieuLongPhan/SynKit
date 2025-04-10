import unittest
from copy import deepcopy
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.Context.hier_context import HierContext


class TestRuleCluster(unittest.TestCase):
    def setUp(self) -> None:
        # Create an instance of HierContext with max_radius=3.
        self.cluster = HierContext(max_radius=3)
        # Load test data from a pickle file.
        # Ensure this file exists at the specified location: "Data/Testcase/graph.pkl.gz"
        self.data = load_from_pickle("Data/Testcase/graph.pkl.gz")
        # Verify that the loaded data is a non-empty list.
        self.assertIsInstance(self.data, list)
        self.assertGreater(len(self.data), 0, "Loaded test data should not be empty.")

    def test_group_class(self):
        """
        Tests the _group_class function by grouping dictionaries based on a key.
        """
        data = [
            {"a": 1, "value": "x"},
            {"a": 2, "value": "y"},
            {"a": 1, "value": "z"},
            {"a": 3, "value": "w"},
        ]
        grouped = HierContext._group_class(data, "a")
        expected = {
            1: [{"a": 1, "value": "x"}, {"a": 1, "value": "z"}],
            2: [{"a": 2, "value": "y"}],
            3: [{"a": 3, "value": "w"}],
        }
        self.assertEqual(
            grouped,
            expected,
            "The grouped output does not match the expected grouping.",
        )

    def test_update_child_idx(self):
        """
        Tests the _update_child_idx function to ensure that child IDs are correctly added
        based on parent–cluster relationships.
        """
        # Create a two-layer hierarchy:
        # Layer 0: One node with 'class' = 1.
        # Layer 1: Two nodes with 'class' = 2 and 3, each having Parent set to 1.
        layer0 = [{"class": 1}]
        layer1 = [{"class": 2, "Parent": 1}, {"class": 3, "Parent": 1}]
        data = [layer0, layer1]
        updated = HierContext._update_child_idx(data, cls_id="class")

        # The node from layer 0 should have its "Child" field updated with the class IDs from layer 1.
        self.assertIn("Child", updated[0][0])
        self.assertEqual(
            set(updated[0][0]["Child"]),
            {2, 3},
            "The 'Child' list of the parent node does not contain the expected child class IDs.",
        )

    def test_process(self):
        """
        Tests the _process function which extracts a context, computes a hash,
        and clusters the data.
        """
        # Use the loaded test data.
        data = deepcopy(self.data)
        cluster_results, templates = HierContext._process(
            data,
            k=1,
            its_key="ITS",
            context_key="K",
            cls_func=self.cluster.cluster,
        )

        # Validate that the first processed data entry has updated keys.
        self.assertIn(
            "R_1", cluster_results[0], "Cluster result should include the key 'R_1'."
        )
        self.assertIn(
            "K",
            cluster_results[0],
            "Cluster result should include the context key 'K'.",
        )
        self.assertIsInstance(
            cluster_results[0]["R_1"],
            int,
            "Cluster identifier 'R_1' should be an integer.",
        )

        # Check that templates are processed correctly.
        # Depending on your dummy clustering behavior, update these expected values.
        self.assertGreater(len(templates), 0, "Templates list should not be empty.")
        self.assertEqual(
            templates[0]["class"],
            0,
            "Expected the first template's 'class' to be 0 after processing.",
        )
        # If you expect a specific number of templates from your test data, verify it.
        # Here, we expect 61 templates. Change if necessary.
        self.assertEqual(len(templates), 61, "Unexpected number of templates produced.")
        self.assertIn("K", templates[0], "Template should contain the context key 'K'.")

    def test_process_level(self):
        """
        Tests the _process_level function by grouping data based on parent cluster IDs,
        processing each group, and then verifying that cluster keys and templates are updated.
        """
        # Use the loaded test data for processing at a child level.
        data = deepcopy(self.data)
        cluster_indices, templates = self.cluster._process_level(
            data, "ITS", "K", self.cluster.cluster, 1
        )

        # Check that each clustered result has the "R_1" field.
        self.assertIn(
            "R_1",
            cluster_indices[0],
            "Each cluster result should have the 'R_1' field for level 1 clustering.",
        )
        self.assertIn(
            "K", cluster_indices[0], "Each cluster result should contain the key 'K'."
        )
        # Validate that templates include the necessary keys.
        self.assertIn("K", templates[0], "Each template should contain the key 'K'.")
        self.assertIn(
            "Parent",
            templates[0],
            "Each template should contain the key 'Parent' to link to a parent cluster.",
        )

    def test_fit(self):
        """
        Tests the fit method which processes a list of graph data entries, classifying each based on
        hierarchical clustering, and verifies that cluster indices and templates are updated at multiple levels.
        """
        data, templates = self.cluster.fit(self.data, its_key="ITS", context_key="K")

        # Check that each data entry has keys for the expected levels.
        for level in range(4):  # Levels 0, 1, 2, 3 (since max_radius=3)
            self.assertIn(
                f"R_{level}",
                data[0],
                f"Data entry should include the key 'R_{level}' after processing.",
            )

        # Check that a template list exists for each hierarchical level.
        self.assertEqual(
            len(templates), 4, "There should be 4 levels of templates (0, 1, 2, 3)."
        )

        # Validate that the first template of the parent level contains required keys.
        self.assertIn(
            "K",
            templates[0][0],
            "The first template in level 0 should contain the key 'K'.",
        )
        self.assertIn(
            "Child",
            templates[0][0],
            "The first template in level 0 should contain the key 'Child' if child relationships were established.",
        )
        # Validate that a template in level 1 contains a Parent key.
        if len(templates) > 1 and templates[1]:
            self.assertIn(
                "Parent",
                templates[1][0],
                "Templates at level 1 should contain the key 'Parent' linking to the parent cluster.",
            )


if __name__ == "__main__":
    unittest.main()
