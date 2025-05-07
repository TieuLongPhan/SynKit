import unittest
from unittest.mock import MagicMock
from synkit.Rule.Compose.valence_constrain import ValenceConstrain

import importlib.util

if importlib.util.find_spec("mod"):
    from mod import BondType

    MOD_AVAILABLE = True
else:
    MOD_AVAILABLE = importlib.util.find_spec("mod") is not None
    print("Optional 'mod' package not found")


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestValenceConstrain(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment. This will be run before each test.
        """
        # Create a mock for load_database
        self.mock_load_database = MagicMock()
        self.mock_load_database.return_value = [{"C": 4, "H": 1}]
        self.valence_constrain = ValenceConstrain()

    def test_initialization(self):
        """
        Test if ValenceConstrain is initialized correctly.
        """
        # Check if bond type orders are correctly initialized
        self.assertEqual(self.valence_constrain.btToOrder[BondType.Single], 1)
        self.assertEqual(self.valence_constrain.btToOrder[BondType.Double], 2)
        self.assertEqual(self.valence_constrain.btToOrder[BondType.Triple], 3)
        self.assertEqual(self.valence_constrain.btToOrder[BondType.Aromatic], 0)

    def test_valence(self):
        """
        Test the valence calculation for a vertex.
        """
        # Create mock vertex and edge objects
        mock_edge = MagicMock()
        mock_edge.bondType = BondType.Single
        mock_vertex = MagicMock()
        mock_vertex.incidentEdges = [mock_edge, mock_edge]  # Two edges
        mock_vertex.stringLabel = "C"  # Carbon atom

        # Check valence calculation
        valence = self.valence_constrain.valence(mock_vertex)
        self.assertEqual(
            valence, 2
        )  # Since Single bond has order 1, and there are two edges

    def test_check_rule_valid(self):
        """
        Test checking a valid rule.
        """
        # Mocking rule and vertices
        mock_rule = MagicMock()
        mock_vertex_pair = MagicMock()
        mock_vertex_pair.left.stringLabel = "C"
        mock_vertex_pair.right.stringLabel = "C"
        mock_vertex_pair.left.incidentEdges = [MagicMock(bondType=BondType.Single)]
        mock_vertex_pair.right.incidentEdges = [MagicMock(bondType=BondType.Single)]
        mock_rule.vertices = [mock_vertex_pair]

        # Check rule validity
        is_valid = self.valence_constrain.check_rule(mock_rule)
        self.assertTrue(is_valid)

    def test_check_rule_invalid_valence(self):
        """
        Test checking an invalid rule due to valence mismatch.
        """
        # Mocking rule and vertices
        mock_rule = MagicMock()
        mock_vertex_pair = MagicMock()
        mock_vertex_pair.left.stringLabel = "C"
        mock_vertex_pair.right.stringLabel = "H"
        mock_vertex_pair.left.incidentEdges = [MagicMock(bondType=BondType.Single)]
        mock_vertex_pair.right.incidentEdges = [
            MagicMock(bondType=BondType.Single),
            MagicMock(bondType=BondType.Double),
        ]  # Two edges for the right vertex

        mock_rule.vertices = [mock_vertex_pair]

        # Check rule validity
        is_valid = self.valence_constrain.check_rule(mock_rule)
        self.assertFalse(is_valid)

    def test_split(self):
        """
        Test splitting good and bad rules.
        """
        # Mocking valid and invalid rules
        valid_rule = MagicMock()
        invalid_rule = MagicMock()

        self.valence_constrain.check_rule = MagicMock(side_effect=[True, False])

        rules = [valid_rule, invalid_rule]

        good, bad = self.valence_constrain.split(rules)

        # Check that valid_rule is in good and invalid_rule is in bad
        self.assertIn(valid_rule, good)
        self.assertIn(invalid_rule, bad)


if __name__ == "__main__":
    unittest.main()
