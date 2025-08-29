import unittest
import importlib

try:
    from rdkit import Chem  # type: ignore

    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

VALENCE = None
if RDKit_AVAILABLE:
    try:
        VALENCE = importlib.import_module("synkit.Chem.Molecule.valence")
    except Exception:
        VALENCE = None


@unittest.skipIf(
    not RDKit_AVAILABLE or VALENCE is None, "RDKit or valence module not available"
)
class TestValenceResolver(unittest.TestCase):
    def test_explicit_implicit_and_total_types_and_relation(self):
        """explicit and implicit are non-negative ints; total == explicit + implicit"""
        m = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(m)
        atom = m.GetAtomWithIdx(1)
        ev = VALENCE.ValenceResolver.explicit(atom)
        iv = VALENCE.ValenceResolver.implicit(atom)
        tot = VALENCE.ValenceResolver.total(atom)

        self.assertIsInstance(ev, int)
        self.assertGreaterEqual(ev, 0)
        self.assertIsInstance(iv, int)
        self.assertGreaterEqual(iv, 0)
        self.assertIsInstance(tot, int)
        self.assertGreaterEqual(tot, 0)
        self.assertEqual(tot, ev + iv)

    def test_explicit_after_addhs_non_decreasing(self):
        """explicit valence after AddHs should not decrease (best-effort)"""
        m = Chem.MolFromSmiles("C")
        self.assertIsNotNone(m)
        a0 = m.GetAtomWithIdx(0)
        ev_before = VALENCE.ValenceResolver.explicit(a0)
        m2 = Chem.AddHs(Chem.Mol(m))
        a1 = m2.GetAtomWithIdx(0)
        ev_after = VALENCE.ValenceResolver.explicit(a1)
        self.assertIsInstance(ev_after, int)
        self.assertGreaterEqual(ev_after, ev_before)


if __name__ == "__main__":
    unittest.main()
