import unittest
import networkx as nx
from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
from synkit.IO.chem_converter import rsmi_to_graph


class TestCanonRSMI(unittest.TestCase):
    def setUp(self):
        # Use generic backend for deterministic behavior
        self.canon = CanonRSMI(backend="wl", wl_iterations=5)
        # Example reaction SMILES with atom-map labels
        self.input_rsmi = "[CH3:3][CH2:5][OH:10]>>[CH2:3]=[CH2:5].[OH2:10]"
        # After expand_aam and canonicalisation, since backend is generic,
        # we expect the SMILES to remain unchanged (identity)
        self.expected_canonical = "[CH2:1]([CH3:2])[OH:3]>>[CH2:1]=[CH2:2].[OH2:3]"

    def test_raw_and_canonical_rsmi(self):
        result = self.canon.canonicalise(self.input_rsmi)
        self.assertEqual(result.raw_rsmi, self.input_rsmi)
        self.assertEqual(result.canonical_rsmi, self.expected_canonical)

    def test_mapping_pairs(self):
        result = self.canon(self.input_rsmi)
        # Mapping pairs should preserve identity mapping
        # mapping_pairs will be matching (map_new, map_old)
        self.assertEqual(sorted(result.mapping_pairs), [(1, 5), (2, 3), (3, 10)])

    def test_graphs_and_hashes(self):
        result = self.canon(self.input_rsmi)
        # raw graphs should parse correctly
        r_raw, p_raw = rsmi_to_graph(self.input_rsmi)
        self.assertIsInstance(result.raw_reactant_graph, nx.Graph)
        self.assertIsInstance(result.raw_product_graph, nx.Graph)
        # canonical graphs should also be Graph
        self.assertIsInstance(result.canonical_reactant_graph, nx.Graph)
        self.assertIsInstance(result.canonical_product_graph, nx.Graph)
        # reactant_hash and product_hash should match underlying canonicaliser
        h_reac = self.canon._canon.canonical_signature(result.canonical_reactant_graph)
        h_prod = self.canon._canon.canonical_signature(result.canonical_product_graph)
        # canonical_hash is combined
        self.assertEqual(result.canonical_hash, f"{h_reac}>>{h_prod}")

    def test_idempotence(self):
        # Applying canonicalise twice yields same result
        result1 = self.canon(self.input_rsmi)
        result2 = self.canon(result1.canonical_rsmi)
        self.assertEqual(result1.canonical_rsmi, result2.canonical_rsmi)
        self.assertEqual(result1.mapping_pairs, result2.mapping_pairs)


if __name__ == "__main__":
    unittest.main()
