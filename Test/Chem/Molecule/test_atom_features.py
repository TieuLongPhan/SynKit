import unittest
import importlib

try:
    from rdkit import Chem  # type: ignore

    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

ATOM_FEATURES = None
DESCRIPTORS = None
if RDKit_AVAILABLE:
    try:
        ATOM_FEATURES = importlib.import_module("synkit.Chem.Molecule.atom_features")
    except Exception:
        ATOM_FEATURES = None
    try:
        DESCRIPTORS = importlib.import_module("synkit.Chem.Molecule.descriptors")
    except Exception:
        DESCRIPTORS = None


@unittest.skipIf(
    not RDKit_AVAILABLE or ATOM_FEATURES is None,
    "RDKit or atom_features module not available",
)
class TestAtomFeatureExtractor(unittest.TestCase):
    def test_minimal_profile_contains_expected_keys(self):
        m = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(m)
        afe = ATOM_FEATURES.AtomFeatureExtractor(m, per=None, profile="minimal")
        atom = m.GetAtomWithIdx(0)
        feat = afe.build_dict(atom)
        expected = {
            "element",
            "aromatic",
            "hcount",
            "charge",
            "radical",
            "isomer",
            "partial_charge",
            "hybridization",
            "in_ring",
            "implicit_hcount",
            "neighbors",
            "atom_map",
        }
        self.assertTrue(expected.issubset(set(feat.keys())))

    @unittest.skipIf(DESCRIPTORS is None, "PerMolDescriptors not available")
    def test_full_profile_includes_distances_and_permol(self):
        m = Chem.MolFromSmiles("CC(=O)O")
        self.assertIsNotNone(m)
        per = DESCRIPTORS.PerMolDescriptors.compute(m)
        afe = ATOM_FEATURES.AtomFeatureExtractor(m, per=per, profile="full")
        afe.build_all()
        feats = afe.all_features
        self.assertIsInstance(feats, list)
        self.assertEqual(len(feats), m.GetNumAtoms())

        # identify carbonyl carbon and assert dist_to_carbonyl == 0 for it
        carbonyl_idx = None
        for atom in m.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetSymbol() == "C":
                for nb in atom.GetNeighbors():
                    if nb.GetSymbol() == "O":
                        b = m.GetBondBetweenAtoms(idx, nb.GetIdx())
                        if b and b.GetBondTypeAsDouble() >= 2.0:
                            carbonyl_idx = idx
                            break
            if carbonyl_idx is not None:
                break
        self.assertIsNotNone(carbonyl_idx)
        feat_carbonyl = feats[carbonyl_idx]
        self.assertEqual(feat_carbonyl.get("dist_to_carbonyl"), 0)
        self.assertTrue("estate" in feat_carbonyl or "crippen_logp" in feat_carbonyl)


if __name__ == "__main__":
    unittest.main()
