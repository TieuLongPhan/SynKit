import unittest
from synkit.Rule.Modify.rule_utils import (
    find_block,
    get_nodes_from_edges,
    parse_context,
    filter_context,
    strip_context,
    _increment_gml_ids,
)
from synkit.Graph.Matcher import GraphMatcherEngine, SubgraphMatch
import importlib

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


class TestGMLFunctions(unittest.TestCase):
    def setUp(self):
        # Example GML data used for various tests
        self.gml_h = (
            "rule [\n"
            '   ruleID "rule"\n'
            "   left [\n"
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 1 target 3 label "-" ]\n'
            '      edge [ source 3 target 4 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "C" ]\n'
            '      node [ id 2 label "H" ]\n'
            '      node [ id 3 label "C" ]\n'
            '      node [ id 4 label "O" ]\n'
            '      node [ id 5 label "H" ]\n'
            '      node [ id 6 label "H" ]\n'
            '      node [ id 7 label "H" ]\n'
            '      node [ id 8 label "H" ]\n'
            '      node [ id 9 label "H" ]\n'
            '      edge [ source 1 target 5 label "-" ]\n'
            '      edge [ source 1 target 6 label "-" ]\n'
            '      edge [ source 3 target 7 label "-" ]\n'
            '      edge [ source 3 target 8 label "-" ]\n'
            '      edge [ source 4 target 9 label "-" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 3 label "=" ]\n'
            '      edge [ source 2 target 4 label "-" ]\n'
            "   ]\n"
            "]"
        )

        # Expected GML after filtering context
        self.gml_expected = (
            "rule [\n"
            '   ruleID "rule"\n'
            "   left [\n"
            '      edge [ source 1 target 4 label "-" ]\n'
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 2 target 3 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "C" ]\n'
            '      node [ id 4 label "H" ]\n'
            '      node [ id 2 label "C" ]\n'
            '      node [ id 3 label "O" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 2 label "=" ]\n'
            '      edge [ source 4 target 3 label "-" ]\n'
            "   ]\n"
            "]"
        )

        self.lines = self.gml_h.split("\n")

    def test_find_block(self):
        rule_start, rule_end = find_block(self.lines, "rule [")
        self.assertIsNotNone(rule_start)
        self.assertIsNotNone(rule_end)

        left_start, left_end = find_block(self.lines, "left [")
        self.assertIsNotNone(left_start)
        self.assertIsNotNone(left_end)

        context_start, context_end = find_block(self.lines, "context [")
        self.assertIsNotNone(context_start)
        self.assertIsNotNone(context_end)

        right_start, right_end = find_block(self.lines, "right [")
        self.assertIsNotNone(right_start)
        self.assertIsNotNone(right_end)

    def test_get_nodes_from_edges(self):
        # Extract left and right blocks
        left_start, left_end = find_block(self.lines, "left [")
        # fmt: off
        left_lines = self.lines[left_start: left_end + 1]
        # fmt: on

        right_start, right_end = find_block(self.lines, "right [")
        # fmt: off
        right_lines = self.lines[right_start: right_end + 1]
        # fmt: on

        left_nodes = get_nodes_from_edges(left_lines)
        right_nodes = get_nodes_from_edges(right_lines)

        # Check some expected nodes in left section
        # left edges: 1-2, 1-3, 3-4 => left_nodes should contain {1,2,3,4}
        self.assertSetEqual(left_nodes, {"1", "2", "3", "4"})

        # right edges: 1-3 (=), 2-4 (-) => right_nodes should contain {1,2,3,4}
        self.assertSetEqual(right_nodes, {"1", "2", "3", "4"})

    def test_parse_context(self):
        # Extract context block
        context_start, context_end = find_block(self.lines, "context [")
        # fmt: off
        context_lines = self.lines[context_start: context_end + 1]
        # fmt: on

        context_nodes, context_edges = parse_context(context_lines)

        # Check that we have all nodes 1 through 9 in context_nodes
        # Node 1: C, Node 2: H, Node 3: C, Node 4: O, Node 5-9: H
        self.assertEqual(len(context_nodes), 9)
        self.assertEqual(context_nodes["1"], "C")
        self.assertEqual(context_nodes["2"], "H")
        self.assertEqual(context_nodes["9"], "H")

        # Edges in context:
        # 1-5, 1-6, 3-7, 3-8, 4-9 all with label "-"
        self.assertEqual(len(context_edges), 5)
        edge_sources_targets = [(e[0], e[1]) for e in context_edges]
        self.assertIn(("1", "5"), edge_sources_targets)
        self.assertIn(("4", "9"), edge_sources_targets)

    def test_filter_context(self):
        # Extract sections
        left_start, left_end = find_block(self.lines, "left [")
        # fmt: off
        left_lines = self.lines[left_start: left_end + 1]
        right_start, right_end = find_block(self.lines, "right [")
        right_lines = self.lines[right_start: right_end + 1]
        context_start, context_end = find_block(self.lines, "context [")
        context_lines = self.lines[context_start: context_end + 1]
        # fmt: on

        # Get relevant nodes
        left_nodes = get_nodes_from_edges(left_lines)
        right_nodes = get_nodes_from_edges(right_lines)
        relevant_nodes = left_nodes.intersection(right_nodes)  # Should be {1,2,3,4}

        filtered_context = filter_context(context_lines, relevant_nodes)

        # Check that hydrogen nodes not in relevant_nodes are removed
        # relevant_nodes = {1,2,3,4}, nodes 5-9 are H and should be removed
        filtered_nodes, filtered_edges = parse_context(filtered_context)
        # After filtering, we should have only nodes 1,2,3,4 left, and these should match:
        # node 1: C, node 2: H (since it's relevant), node 3: C, node 4: O
        self.assertEqual(set(filtered_nodes.keys()), {"1", "2", "3", "4"})
        self.assertNotIn("5", filtered_nodes)  # previously H removed
        self.assertNotIn("9", filtered_nodes)  # previously H removed

        # Edges in filtered context should not contain edges to 5-9
        for e in filtered_edges:
            self.assertNotIn(e[0], {"5", "6", "7", "8", "9"})
            self.assertNotIn(e[1], {"5", "6", "7", "8", "9"})

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_strip_context(self):
        self.assertFalse(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(
                self.gml_expected, self.gml_h
            )
        )
        self.assertTrue(
            SubgraphMatch().rule_subgraph_morphism(self.gml_expected, self.gml_h)
        )
        output = strip_context(self.gml_h)
        self.assertTrue(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(
                self.gml_expected, output
            )
        )

    def test_increment_gml_ids_without_id_zero(self):
        """Test increment_gml_ids method with no id 0 in input."""
        gml_content = "node [ id 1 ]"
        result = _increment_gml_ids(gml_content)
        self.assertEqual(result, gml_content)

    def test_increment_gml_ids_with_id_zero(self):
        """Test increment_gml_ids method with id 0 in input."""
        gml_content = "node [ id 0 ] node [ id 1 ]"
        expected_result = "node [ id 1 ] node [ id 2 ]"
        result = _increment_gml_ids(gml_content)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
