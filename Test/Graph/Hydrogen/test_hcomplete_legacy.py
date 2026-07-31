import unittest

from synkit.Graph.Hyrogen.hcomplete import HComplete
from synkit.Graph.Hyrogen.hcomplete_legacy import LegacyHComplete
from synkit.Graph.Hyrogen.hextend import HExtend
from synkit.Graph.Hyrogen.hextend_legacy import LegacyHExtend
from synkit.IO.chem_converter import its_to_rsmi
from synkit.IO.data_io import load_from_pickle


class TestLegacyHComplete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_from_pickle("./Data/Testcase/hydro/hydrogen_test.pkl.gz")

    def test_unambiguous_completion_remains_compatible(self):
        enhanced = HComplete.complete_its(self.data[0]["ITS"])
        legacy = LegacyHComplete.complete_its(self.data[0]["ITS"])

        self.assertTrue(enhanced.ok)
        self.assertTrue(legacy.ok)
        self.assertEqual(
            its_to_rsmi(enhanced.its),
            its_to_rsmi(legacy.its),
        )

    def test_legacy_behavior_is_isolated_from_enhanced_completion(self):
        enhanced = HComplete.complete_its(self.data[16]["ITS"])
        legacy = LegacyHComplete.complete_its(self.data[16]["ITS"])

        self.assertFalse(enhanced.ok)
        self.assertEqual(enhanced.reason, "non_equivariant_rc")
        self.assertTrue(legacy.ok)

    def test_extension_class_counts_show_legacy_collapse(self):
        enhanced_rc, _, _ = HExtend._extend_unique(self.data[16]["ITS"])
        legacy_rc, _, _ = LegacyHExtend._extend_unique(self.data[16]["ITS"])

        self.assertEqual(len(enhanced_rc), 2)
        self.assertEqual(len(legacy_rc), 1)


if __name__ == "__main__":
    unittest.main()
