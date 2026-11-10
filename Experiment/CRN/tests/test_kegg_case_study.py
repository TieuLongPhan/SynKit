"""Cached KEGG modules, conservation laws, and prescribed glycolysis flux."""

from collections import Counter
import unittest

from Experiment.CRN.studies.kegg import (
    CASE_STUDY_MODULES,
    GLYCOLYSIS_FLUX,
    KeggModuleAnalysis,
    analyze_kegg_module,
    build_kegg_crn,
    check_glycolysis_flux,
    kegg_case_study_report,
    load_kegg_cache,
    module_ids,
)
from synkit.CRN.Structure.syncrn import SynCRN


class TestKeggCache(unittest.TestCase):
    def test_cached_modules_have_reviewed_inputs(self):
        cache = load_kegg_cache()
        self.assertIn("modules", cache)
        self.assertIn("source", cache)
        available = set(module_ids())
        for module_id in CASE_STUDY_MODULES:
            with self.subTest(module=module_id):
                self.assertIn(module_id, available)
        for module_id, entry in cache["modules"].items():
            with self.subTest(module=module_id):
                self.assertTrue(entry["equations"])
                self.assertTrue(entry["compound_names"])
                self.assertTrue(entry["name"])

    def test_cache_is_memoized(self):
        self.assertIs(load_kegg_cache(), load_kegg_cache())


class TestBuildKeggCrn(unittest.TestCase):

    def test_glycolysis_drops_currency_metabolites_by_default(self):
        crn = build_kegg_crn("M00001")
        self.assertIsInstance(crn, SynCRN)
        self.assertGreater(crn.n_species, 5)
        self.assertGreater(crn.n_reactions, 5)
        labels = {sp.label for sp in crn.species.values()}
        self.assertNotIn("ATP", labels)
        self.assertNotIn("ADP", labels)
        self.assertIn("Pyruvate", labels)

    def test_keeping_currency_metabolites_grows_the_network(self):
        without = build_kegg_crn("M00001")
        with_currency = build_kegg_crn("M00001", drop_currency=False)
        self.assertGreater(with_currency.n_species, without.n_species)
        self.assertIn("ATP", {sp.label for sp in with_currency.species.values()})

    def test_unknown_module_raises(self):
        with self.assertRaises(KeyError):
            build_kegg_crn("M99999")


class TestAnalyzeKeggModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_kegg_module("M00001")

    def test_report_retains_module_identity_and_moieties(self):
        self.assertIsInstance(self.result, KeggModuleAnalysis)
        self.assertEqual(self.result.module_id, "M00001")
        self.assertIn("Glycolysis", self.result.module_name)
        payload = self.result.to_dict()
        self.assertEqual(payload["module_id"], "M00001")
        self.assertIn("moieties", payload)
        text = str(self.result)
        self.assertIn("M00001", text)
        self.assertIn("conserved moieties", text)

    def test_deficiency_identity_holds(self):
        self.assertEqual(
            self.result.deficiency,
            self.result.n_complexes - self.result.n_linkage_classes - self.result.rank,
        )

    def test_finds_the_sugar_phosphate_backbone_moiety(self):
        # The largest conserved moiety of glycolysis is its carbon skeleton:
        # every phosphorylated intermediate from glucose through to pyruvate.
        largest = max(self.result.moieties, key=len)
        self.assertIn("alpha-D-Glucose", largest)
        self.assertIn("Pyruvate", largest)
        self.assertIn("D-Fructose 1,6-bisphosphate", largest)

    def test_finds_the_ferredoxin_couple(self):
        pools = {frozenset(m) for m in self.result.moieties}
        self.assertIn(frozenset({"Oxidized ferredoxin", "Reduced ferredoxin"}), pools)

    def test_glucose_is_a_siphon(self):
        # Glucose is consumed and never produced: the module depends on it.
        self.assertIn(["alpha-D-Glucose"], self.result.siphons)

    def test_a_catabolic_module_is_not_persistent(self):
        self.assertFalse(self.result.persistent)

    def test_uncovered_siphons_are_a_subset_of_siphons(self):
        for siphon in self.result.uncovered_siphons:
            self.assertIn(siphon, self.result.siphons)

    def test_every_case_study_module_analyses(self):
        for module_id in CASE_STUDY_MODULES:
            with self.subTest(module=module_id):
                result = analyze_kegg_module(module_id)
                self.assertGreater(result.n_species, 0)
                self.assertGreaterEqual(result.deficiency, 0)


class TestGlycolysisFluxRealizability(unittest.TestCase):
    def test_one_glucose_yields_a_complete_prescribed_flux_certificate(self):
        result = check_glycolysis_flux()
        self.assertTrue(result["realizable"])
        self.assertEqual(result["status"], "realizable")
        self.assertEqual(result["termination_reason"], "certificate")
        self.assertGreater(result["explored_states"], 0)
        self.assertIsNotNone(result["certificate"])
        self.assertEqual(Counter(result["certificate"]), GLYCOLYSIS_FLUX)
        self.assertEqual(result["certificate"][0], "R01786")

    def test_a_flux_needing_absent_substrate_is_not_realizable(self):
        # Firing the lower half without ever making a triose cannot work.
        result = check_glycolysis_flux(flux={"R00200": 1}, initial_glucose=1)
        self.assertFalse(result["realizable"])
        self.assertEqual(result["status"], "unrealizable")
        self.assertTrue(result["exhaustive"])


class TestCaseStudyReport(unittest.TestCase):
    def test_supported_report_formats_cover_the_case_studies(self):
        text = kegg_case_study_report()
        for module_id in CASE_STUDY_MODULES:
            self.assertIn(module_id, text)
        table = kegg_case_study_report(fmt="markdown")
        self.assertIn("| module", table)
        self.assertIn("delta", table)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            kegg_case_study_report(fmt="latex")


if __name__ == "__main__":
    unittest.main()
