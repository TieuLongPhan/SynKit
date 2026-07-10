import unittest

from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_destruction import ITSDestruction
from synkit.IO.chem_converter import rsmi_to_graph


class TestITSDestruction(unittest.TestCase):
    def test_direct_tuple_mode_preserves_electron_fields(self):
        reactant, product = rsmi_to_graph(
            "[NH2:1][CH3:2]>>[NH:1]=[CH2:2]",
        )
        reactant.nodes[1]["lone_pairs"] = 1
        reactant.nodes[1]["radical"] = 0
        reactant.nodes[1]["valence_electrons"] = 5
        product.nodes[1]["lone_pairs"] = 1
        product.nodes[1]["radical"] = 1
        product.nodes[1]["valence_electrons"] = 5

        its = ITSConstruction.construct(reactant, product)
        left, right = ITSDestruction(its).decompose()

        self.assertEqual(left.nodes[1]["lone_pairs"], 1)
        self.assertEqual(right.nodes[1]["radical"], 1)
        self.assertEqual(right.nodes[1]["valence_electrons"], 5)
        self.assertEqual(left.edges[1, 2]["sigma_order"], 1.0)
        self.assertEqual(right.edges[1, 2]["pi_order"], 1.0)


if __name__ == "__main__":
    unittest.main()
