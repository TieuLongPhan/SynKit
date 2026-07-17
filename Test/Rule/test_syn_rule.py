import unittest
import networkx as nx
import pytest
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.Stereo import StereoChange, TetrahedralStereo
from synkit.Rule import NonInvertibleStereoEffectError, SynRule
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
        self.assertIn("stereo=none", r)

    def test_repr_summarizes_stereo_semantics(self):
        sn2 = SynRule.from_smart(
            "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>"
            "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]",
            name="sn2-inversion",
            canon=False,
            implicit_h=False,
            format="tuple",
        )
        self.assertIn(
            "stereo=(guards={atom:1},effects={INVERTED:1})",
            repr(sn2),
        )
        self.assertEqual(
            sn2.stereo_summary(),
            {
                "atom:2": {
                    "descriptor": "tetrahedral",
                    "change": "INVERTED",
                    "reactant": {
                        "atoms": [2, 1, 3, 4, "@H:2"],
                        "parity": -1,
                    },
                    "product": {
                        "atoms": [2, 1, 3, 5, "@H:2"],
                        "parity": 1,
                    },
                    "reference_delta": {"removed": [4], "added": [5]},
                }
            },
        )

        racemic = SynRule.from_smart(
            "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]",
            canon=False,
            implicit_h=False,
            format="tuple",
            stereo_outcomes={"atom:2": "RACEMIC"},
        )
        self.assertIn(
            "stereo=(guards=0,effects={FORMED:1},outcomes={RACEMIC:1})",
            repr(racemic),
        )
        self.assertEqual(
            racemic.stereo_summary()["atom:2"]["outcome"],
            {"kind": "RACEMIC", "weights": [0.5, 0.5]},
        )
        self.assertIn("queries={either:1}", repr(racemic.reversed()))

    def test_syn_anti_coupling_replaces_encoded_endpoint_stereo(self):
        semihydrogenation = (
            "[CH3:1][C:2]#[C:3][CH3:4].[H:5][H:6]>>"
            "[CH3:1][C:2]([H:5])=[C:3]([H:6])[CH3:4]"
        )

        def coupling(relation):
            return {"bond:2-3": relation}

        syn = SynRule.from_smart(
            semihydrogenation,
            canon=False,
            implicit_h=False,
            format="tuple",
            stereo_couplings=coupling("syn"),
        )
        anti = SynRule.from_smart(
            semihydrogenation,
            canon=False,
            implicit_h=False,
            format="tuple",
            stereo_couplings=coupling("anti"),
        )

        self.assertEqual(
            syn.stereo_summary()["bond:2-3"]["coupling"],
            {
                "kind": "VICINAL_ADDITION",
                "relation": "SYN",
                "centers": [2, 3],
                "ligands": [5, 6],
            },
        )
        self.assertEqual(
            anti.stereo_summary()["bond:2-3"]["coupling"]["relation"],
            "ANTI",
        )
        self.assertEqual(syn.stereo_effects, {})
        self.assertEqual(anti.stereo_effects, {})
        self.assertEqual(
            anti.rc.raw.graph["stereo_couplings"],
            {"bond:2-3": "ANTI"},
        )
        self.assertNotEqual(syn, anti)
        self.assertIn("couplings={SYN:1}", repr(syn))
        self.assertEqual(
            syn.reversed().stereo_couplings["bond:2-3"].kind,
            "VICINAL_ELIMINATION",
        )

        encoded_endpoint = (
            "[CH3:1][C:2]#[C:3][CH3:4].[H:5][H:6]>>"
            "[CH3:1]/[C:2]([H:5])=[C:3](/[H:6])[CH3:4]"
        )
        with self.assertRaisesRegex(ValueError, "derives endpoint stereo"):
            SynRule.from_smart(
                encoded_endpoint,
                canon=False,
                implicit_h=False,
                format="tuple",
                stereo_couplings=coupling("syn"),
            )

    def test_tuple_rule_preserves_tuple_representation(self):
        smart = "[CH3:1][CH3:2]>>[CH2:1]=[CH2:2]"

        rule = SynRule.from_smart(
            smart,
            canon=False,
            implicit_h=False,
            format="tuple",
        )

        self.assertEqual(rule._format, "tuple")
        self.assertEqual(rule.rc.raw.nodes[1]["element"], ("C", "C"))
        self.assertEqual(rule.rc.raw.edges[1, 2]["pi_order"], (0.0, 1.0))
        self.assertEqual(rule.left.raw.edges[1, 2]["pi_order"], 0.0)
        self.assertEqual(rule.right.raw.edges[1, 2]["pi_order"], 1.0)

    def test_tuple_rule_implicit_h_strips_removable_explicit_hydrogens(self):
        smart = "[CH3:1][Cl:2].[O:3]([H:4])[H:5]>>[CH3:1][O:3][H:4].[Cl:2][H:5]"
        rule = SynRule.from_smart(
            smart,
            canon=False,
            implicit_h=True,
            format="tuple",
        )

        self.assertFalse(
            any(data["element"] == "H" for _, data in rule.left.raw.nodes(data=True))
        )
        self.assertFalse(
            any(data["element"] == "H" for _, data in rule.right.raw.nodes(data=True))
        )
        self.assertEqual(rule.rc.raw.nodes[1]["hcount"], (0, 0))
        self.assertEqual(rule.rc.raw.nodes[2]["hcount"], (0, 1))
        self.assertEqual(rule.rc.raw.nodes[3]["hcount"], (2, 1))
        self.assertTrue(rule.rc.raw.nodes[2]["h_pairs"])
        self.assertTrue(rule.rc.raw.nodes[3]["h_pairs"])


def test_unspecified_rule_refuses_reverse_before_mutation():
    its = rsmi_to_its("[CH3:1][Cl:2]>>[CH3:1][Cl:2]", format="tuple")
    before = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    after = TetrahedralStereo((1, 2, 3, 4, 5), None)
    its.graph["stereo_descriptors"] = {
        "reactant": {"atom:1": before},
        "product": {"atom:1": after},
    }
    its.graph["stereo_changes"] = {"atom:1": StereoChange("UNSPECIFIED", before, after)}
    rule = SynRule(its, canon=False, implicit_h=False, format="tuple")
    original_summary = rule.stereo_summary()

    assert rule.is_stereo_reversible is False
    assert rule.non_invertible_stereo_targets() == ("atom:1",)
    for _ in range(2):
        with pytest.raises(NonInvertibleStereoEffectError) as excinfo:
            rule.reversed()
        assert excinfo.value.reason == "non_reversible_unspecified_descriptor"
        assert excinfo.value.targets == ("atom:1",)
    assert rule.stereo_summary() == original_summary


if __name__ == "__main__":
    unittest.main()
