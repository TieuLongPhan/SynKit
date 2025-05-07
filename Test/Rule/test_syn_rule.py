import unittest
import networkx as nx
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule.syn_rule import SynRule
from synkit.Graph.canon_graph import GraphCanonicaliser


class TestSynRuleImplicitAndCanon(unittest.TestCase):
    """Test SynRule with implicit H‐stripping and canonicalisation."""

    def setUp(self):
        # SMARTS and GML from your example
        self.smart = "[Br:1][CH3:2].[OH:3][H:4]>>[Br:1][H:4].[CH3:2][OH:3]"
        self.gml = (
            "rule [\n"
            '   ruleID "rule"\n'
            "   left [\n"
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 3 target 4 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "Br" ]\n'
            '      node [ id 2 label "C" ]\n'
            '      node [ id 3 label "H" ]\n'
            '      node [ id 4 label "O" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 3 label "-" ]\n'
            '      edge [ source 2 target 4 label "-" ]\n'
            "   ]\n"
            "]"
        )

        self.explicit = rsmi_to_its(self.smart)

        self.explicit = SynRule(self.explicit, implicit_h=True)

    def _canonical_graph(self, G: nx.Graph) -> nx.Graph:
        """Helper to strip implicit H, then canonicalize a raw ITS graph."""
        # make copies for left/right (we only care about shared H in rc)
        left = G.copy()
        right = G.copy()
        # strip explicit H-nodes into hcount/h_pairs
        SynRule._strip_explicit_h(G, left, right)
        # canonicalise
        canon = GraphCanonicaliser()
        return canon.make_canonical_graph(G)

    def test_smart_and_gml_agree(self):
        # Build rules with implicit_h & canon enabled
        rule_s = SynRule.from_smart(
            self.smart, name="r", canonicaliser=None, canon=True, implicit_h=True
        )
        rule_g = SynRule.from_gml(
            self.gml, name="r", canonicaliser=None, canon=True, implicit_h=True
        )

        # Their canonical ITS graphs should be isomorphic
        Cr_s = rule_s.rc.canonical  # type: ignore[attr-defined]
        Cr_g = rule_g.rc.canonical

        def node_match(n1_attrs, n2_attrs):
            """
            Match if element and hcount agree.
            """
            return n1_attrs.get("element") == n2_attrs.get("element") and n1_attrs.get(
                "hcount"
            ) == n2_attrs.get("hcount")

        def edge_match(e1_attrs, e2_attrs):
            """
            Match if bond order and standard_order agree.
            """
            return e1_attrs.get("order") == e2_attrs.get("order") and e1_attrs.get(
                "standard_order"
            ) == e2_attrs.get("standard_order")

        iso = nx.is_isomorphic(Cr_s, Cr_g, node_match=node_match, edge_match=edge_match)
        self.assertTrue(
            iso,
            "Canonical ITS graphs from SMART and GML should be isomorphic "
            "(mismatch in node or edge attributes)",
        )

    def test_str_repr(self):
        rule = SynRule.from_smart(self.smart, name="foo", canon=True, implicit_h=True)
        s = str(rule)
        self.assertIn("foo", s)
        # signatures truncated to 8 chars
        self.assertRegex(s, r"left=[0-9a-f]{8}… right=[0-9a-f]{8}…")
        r = repr(rule)
        # repr mentions node/edge counts for canonical rc
        self.assertIn("rc=(|V|=", r)
        self.assertIn("left=(|V|=", r)
        self.assertIn("right=(|V|=", r)


if __name__ == "__main__":
    unittest.main()
