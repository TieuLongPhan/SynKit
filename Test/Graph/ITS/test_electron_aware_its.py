import unittest

import networkx as nx

from synkit.Graph.Hyrogen._misc import h_to_explicit, normalize_h_pair_graph
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.rc_extractor import RCExtractor


class TestElectronAwareITS(unittest.TestCase):
    def setUp(self):
        self.reactant = nx.Graph()
        self.reactant.add_node(
            1,
            element="N",
            aromatic=False,
            hcount=2,
            charge=0,
            neighbors=["C"],
            lone_pairs=1,
            radical=0,
            valence_electrons=5,
        )
        self.reactant.add_node(
            2,
            element="C",
            aromatic=False,
            hcount=3,
            charge=0,
            neighbors=["N"],
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )
        self.reactant.add_edge(
            1,
            2,
            order=1.0,
            kekule_order=1.0,
            sigma_order=1.0,
            pi_order=0.0,
        )

        self.product = nx.Graph()
        self.product.add_node(
            1,
            element="N",
            aromatic=False,
            hcount=1,
            charge=0,
            neighbors=["C"],
            lone_pairs=1,
            radical=1,
            valence_electrons=5,
        )
        self.product.add_node(
            2,
            element="C",
            aromatic=False,
            hcount=3,
            charge=0,
            neighbors=["N"],
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )
        self.product.add_edge(
            1,
            2,
            order=2.0,
            kekule_order=2.0,
            sigma_order=1.0,
            pi_order=1.0,
        )

    def test_construct_stores_default_electron_pairs(self):
        its = ITSConstruction.construct(self.reactant, self.product)

        self.assertEqual(its.nodes[1]["lone_pairs"], (1, 1))
        self.assertEqual(its.nodes[1]["radical"], (0, 1))
        self.assertEqual(its.nodes[1]["valence_electrons"], (5, 5))
        self.assertEqual(its.edges[1, 2]["sigma_order"], (1.0, 1.0))
        self.assertEqual(its.edges[1, 2]["pi_order"], (0.0, 1.0))

    def test_rc_extractor_marks_radical_change(self):
        its = ITSConstruction.construct(self.reactant, self.product)

        rc = RCExtractor().extract(its)

        self.assertIn(1, rc)
        self.assertIn("radical", rc.graph["rc"]["node_reasons"][1])
        self.assertEqual(rc.edges[1, 2]["standard_order"], -1.0)

    def test_reverter_restores_electron_fields(self):
        its = ITSConstruction.construct(self.reactant, self.product)
        reactant, product = (
            ITSReverter(its).to_reactant_graph(),
            ITSReverter(its).to_product_graph(),
        )

        self.assertEqual(reactant.nodes[1]["radical"], 0)
        self.assertEqual(product.nodes[1]["radical"], 1)
        self.assertEqual(reactant.edges[1, 2]["pi_order"], 0.0)
        self.assertEqual(product.edges[1, 2]["pi_order"], 1.0)

    def test_normalize_h_pair_graph_supports_named_pair_storage(self):
        its = ITSConstruction.construct(self.reactant, self.product)

        normalized = normalize_h_pair_graph(its)

        self.assertEqual(normalized.nodes[1]["hcount"], (1, 0))

    def test_rc_extraction_survives_named_hcount_normalization(self):
        its = ITSConstruction.construct(self.reactant, self.product)
        normalized = normalize_h_pair_graph(its)

        before = RCExtractor().extract(its)
        after = RCExtractor().extract(normalized)

        self.assertEqual(before.graph["rc"]["nodes"], after.graph["rc"]["nodes"])
        self.assertEqual(
            before.graph["rc"]["node_reasons"],
            after.graph["rc"]["node_reasons"],
        )

    def test_preserve_full_attrs_exports_unfiltered_rc_snapshots(self):
        its = ITSConstruction.construct(self.reactant, self.product)
        its.nodes[1]["custom_marker"] = "kept"

        rc = RCExtractor(preserve_full_attrs=True).extract(its)

        self.assertEqual(rc.graph["rc"]["node_attrs"][1]["custom_marker"], "kept")

    def test_reverter_drops_nodes_absent_on_one_side(self):
        self.reactant.add_node(
            3,
            element="H",
            aromatic=False,
            hcount=0,
            charge=0,
            neighbors=["N"],
            lone_pairs=0,
            radical=0,
            valence_electrons=1,
        )
        self.reactant.add_edge(
            1,
            3,
            order=1.0,
            kekule_order=1.0,
            sigma_order=1.0,
            pi_order=0.0,
        )

        its = ITSConstruction.construct(self.reactant, self.product)
        reactant, product = (
            ITSReverter(its).to_reactant_graph(),
            ITSReverter(its).to_product_graph(),
        )

        self.assertIn(3, reactant)
        self.assertNotIn(3, product)

    def test_h_to_explicit_expands_named_pair_hcounts_by_side(self):
        its = ITSConstruction.construct(self.reactant, self.product)

        expanded = h_to_explicit(its, [1], its=True)
        hydrogens = [
            node
            for node, attrs in expanded.nodes(data=True)
            if attrs.get("element") == ("H", "H")
        ]

        self.assertEqual(expanded.nodes[1]["hcount"], (0, 0))
        self.assertEqual(len(hydrogens), 2)
        self.assertEqual(
            {expanded.nodes[node]["present"] for node in hydrogens},
            {(True, True), (True, False)},
        )
        self.assertEqual(
            {expanded.edges[1, node]["order"] for node in hydrogens},
            {(1.0, 1.0), (1.0, 0.0)},
        )

    def test_explicit_tuple_hydrogens_revert_to_correct_side_graphs(self):
        its = ITSConstruction.construct(self.reactant, self.product)
        expanded = h_to_explicit(its, [1], its=True)
        reactant, product = (
            ITSReverter(expanded).to_reactant_graph(),
            ITSReverter(expanded).to_product_graph(),
        )

        reactant_hydrogens = [
            node for node, attrs in reactant.nodes(data=True) if attrs["element"] == "H"
        ]
        product_hydrogens = [
            node for node, attrs in product.nodes(data=True) if attrs["element"] == "H"
        ]

        self.assertEqual(len(reactant_hydrogens), 2)
        self.assertEqual(len(product_hydrogens), 1)


if __name__ == "__main__":
    unittest.main()
