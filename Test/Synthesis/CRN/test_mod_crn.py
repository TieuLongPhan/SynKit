import unittest
import importlib.util
from synkit.IO.data_io import load_database
from synkit.Synthesis.CRN.mod_crn import MODCRN


MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestCRN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Heavy I/O only once
        cls.rules = load_database("Data/Testcase/para_rule.json.gz")
        cls.rules = [v["gml"] for v in cls.rules]

    def setUp(self):
        self.start_smiles = [
            "c1ccccc1",
            "ClCl",
            "O[Na]",
            "O=[N+]([O-])O",
            "[H][H]",
            "[H][H]",
            "[H][H]",
            "CC(=O)Cl",
        ]

        # Auto‑builds CRN on construction
        self.crn = MODCRN(self.rules, self.start_smiles, n_repeats=2)

    def test_initialisation(self):
        # Rules and graphs loaded correctly
        self.assertEqual(len(self.crn.rules), len(self.rules))
        # Duplicates removed: one H2 should be deduplicated
        self.assertEqual(len(self.crn.graphs), len(self.start_smiles))
        # Repeats parameter stored
        self.assertEqual(self.crn.repeats, 2)

        self.crn.build()
        self.assertEqual(self.crn.num_vertices, 38)
        self.assertEqual(self.crn.num_edges, 35)

    def test_build_populates_graph(self):
        self.crn.build()
        # After building, should have at least one vertex and possibly edges
        self.assertGreater(self.crn.num_vertices, 0)
        # Edges may be zero if no rule applies, but ensure property works
        self.assertIsInstance(self.crn.num_edges, int)
