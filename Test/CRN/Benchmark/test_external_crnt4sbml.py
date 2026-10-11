"""Unit contracts for the optional CRNT4SBML comparison study."""

import unittest

from Experiment.CRN.external_crnt4sbml import SHARED_FIELDS, compare_payload


class TestExternalComparison(unittest.TestCase):
    def setUp(self):
        self.row = {
            "n_species": 2,
            "n_reactions": 2,
            "n_complexes": 2,
            "n_linkage_classes": 1,
            "rank": 1,
            "deficiency": 0,
            "weakly_reversible": True,
            "linkage_class_deficiencies": [0],
        }

    def test_equal_payload_passes(self):
        external = {"networks": [{"name": "example", **self.row}]}
        result = compare_payload({"example": self.row}, external)
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(result["networks"][0]["matches"])

    def test_difference_is_named(self):
        changed = dict(self.row, deficiency=1)
        external = {"networks": [{"name": "example", **changed}]}
        result = compare_payload({"example": self.row}, external)
        self.assertFalse(result["checks"]["all_shared_structural_quantities_agree"])
        difference = result["networks"][0]["differences"]["deficiency"]
        self.assertEqual(difference, {"synkit": 0, "crnt4sbml": 1})

    def test_missing_and_error_rows_fail(self):
        for rows in ([], [{"name": "example", "error": "parse failed"}]):
            with self.subTest(rows=rows):
                result = compare_payload({"example": self.row}, {"networks": rows})
                self.assertFalse(all(result["checks"].values()))

    def test_declared_fields_are_present(self):
        self.assertTrue(set(SHARED_FIELDS).issubset(self.row))


if __name__ == "__main__":
    unittest.main()
