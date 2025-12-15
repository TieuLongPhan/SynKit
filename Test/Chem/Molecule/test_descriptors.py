import unittest
import importlib

try:
    from rdkit import Chem  # type: ignore

    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

DESCRIPTORS = None
if RDKit_AVAILABLE:
    try:
        DESCRIPTORS = importlib.import_module("synkit.Chem.Molecule.descriptors")
    except Exception:
        DESCRIPTORS = None


@unittest.skipIf(
    not RDKit_AVAILABLE or DESCRIPTORS is None,
    "RDKit or descriptors module not available",
)
class TestPerMolDescriptors(unittest.TestCase):
    def test_compute_lengths_and_types(self):
        m = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(m)
        # best-effort Gasteiger
        try:
            DESCRIPTORS.compute_gasteiger_inplace(m)
        except Exception:
            pass
        desc = DESCRIPTORS.PerMolDescriptors.compute(m)
        n = m.GetNumAtoms()

        self.assertIsInstance(desc, DESCRIPTORS.PerMolDescriptors)
        self.assertEqual(len(desc.gasteiger), n)
        self.assertEqual(len(desc.estate), n)
        self.assertEqual(len(desc.crippen_logp), n)
        self.assertEqual(len(desc.crippen_mr), n)

        for v in desc.gasteiger + desc.estate + desc.crippen_logp + desc.crippen_mr:
            self.assertIsInstance(v, float)

    def test_from_smiles_and_minmax_normalize(self):
        desc = DESCRIPTORS.PerMolDescriptors.from_smiles("CC(=O)O", normalize=None)
        n = desc.num_atoms
        self.assertEqual(n, len(desc.gasteiger))

        desc_mm = DESCRIPTORS.PerMolDescriptors.compute(
            Chem.MolFromSmiles("CC(=O)O"), normalize="minmax"
        )
        for v in (
            desc_mm.gasteiger
            + desc_mm.estate
            + desc_mm.crippen_logp
            + desc_mm.crippen_mr
        ):
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, -1e-12)
            self.assertLessEqual(v, 1.0 + 1e-12)

    def test_zscore_normalization_statistics(self):
        desc = DESCRIPTORS.PerMolDescriptors.compute(
            Chem.MolFromSmiles("CCO"), normalize="zscore"
        )

        def approx_zero(vec):
            if len(vec) <= 1:
                return True
            mean = sum(vec) / len(vec)
            return abs(mean) < 1e-6 or all(abs(x) < 1e-6 for x in vec)

        self.assertTrue(approx_zero(desc.gasteiger))
        self.assertIsInstance(desc.estate, list)


if __name__ == "__main__":
    unittest.main()
