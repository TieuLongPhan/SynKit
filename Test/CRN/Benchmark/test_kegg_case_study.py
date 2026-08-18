"""The KEGG case study, asserted as a test.

The cached modules make this reproducible offline: no KEGG request is made, so
the numbers here are the same ones the paper reports.
"""

import unittest

from synkit.CRN.Benchmark.kegg import (
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
from synkit.CRN.Query.to_syncrn import (
    CURRENCY_COMPOUNDS,
    syncrn_from_kegg_equations,
)
from synkit.CRN.Structure.syncrn import SynCRN


class TestKeggCache(unittest.TestCase):
    def test_cache_loads(self):
        cache = load_kegg_cache()
        self.assertIn("modules", cache)
        self.assertIn("source", cache)

    def test_every_case_study_module_is_cached(self):
        available = set(module_ids())
        for module_id in CASE_STUDY_MODULES:
            with self.subTest(module=module_id):
                self.assertIn(module_id, available)

    def test_modules_carry_equations_and_names(self):
        for module_id, entry in load_kegg_cache()["modules"].items():
            with self.subTest(module=module_id):
                self.assertTrue(entry["equations"])
                self.assertTrue(entry["compound_names"])
                self.assertTrue(entry["name"])

    def test_cache_is_memoized(self):
        self.assertIs(load_kegg_cache(), load_kegg_cache())


class TestKeggToSynCRN(unittest.TestCase):
    def test_irreversible_equation(self):
        crn = syncrn_from_kegg_equations({"R1": "C00001 => C00002"})
        self.assertEqual(crn.n_species, 2)
        self.assertEqual(crn.n_reactions, 1)

    def test_reversible_equation_is_split(self):
        crn = syncrn_from_kegg_equations({"R1": "C00001 <=> C00002"})
        self.assertEqual(crn.n_reactions, 2)

    def test_reversible_expansion_can_be_disabled(self):
        crn = syncrn_from_kegg_equations(
            {"R1": "C00001 <=> C00002"}, expand_reversible=False
        )
        self.assertEqual(crn.n_reactions, 1)

    def test_stoichiometry_is_parsed(self):
        crn = syncrn_from_kegg_equations({"R1": "2 C00001 => C00002"})
        reaction = next(iter(crn.reactions.values()))
        self.assertEqual(sorted(reaction.lhs.to_dict().values()), [2])

    def test_names_become_labels(self):
        crn = syncrn_from_kegg_equations(
            {"R1": "C00111 => C00118"},
            names={"C00111": "DHAP", "C00118": "GAP"},
        )
        self.assertEqual(
            crn.to_equations(species="label", include_id=False), ["DHAP >> GAP"]
        )

    def test_dropping_compounds_removes_them(self):
        crn = syncrn_from_kegg_equations(
            {"R1": "C00002 + C00267 => C00008 + C00668"},
            drop_compounds=CURRENCY_COMPOUNDS,
        )
        self.assertEqual(crn.n_species, 2)

    def test_empty_equations_are_skipped(self):
        crn = syncrn_from_kegg_equations({"R1": None, "R2": "C00001 => C00002"})
        self.assertEqual(crn.n_reactions, 1)

    def test_reaction_labels_are_kegg_ids(self):
        crn = syncrn_from_kegg_equations({"R01015": "C00111 => C00118"})
        self.assertEqual(
            [r.label for r in crn.reactions.values()], ["R01015"]
        )


class TestBuildKeggCrn(unittest.TestCase):
    def test_glycolysis_builds(self):
        crn = build_kegg_crn("M00001")
        self.assertIsInstance(crn, SynCRN)
        self.assertGreater(crn.n_species, 5)
        self.assertGreater(crn.n_reactions, 5)

    def test_currency_metabolites_are_dropped_by_default(self):
        labels = {sp.label for sp in build_kegg_crn("M00001").species.values()}
        self.assertNotIn("ATP", labels)
        self.assertNotIn("ADP", labels)
        self.assertIn("Pyruvate", labels)

    def test_keeping_currency_metabolites_grows_the_network(self):
        without = build_kegg_crn("M00001")
        with_currency = build_kegg_crn("M00001", drop_currency=False)
        self.assertGreater(with_currency.n_species, without.n_species)
        self.assertIn(
            "ATP", {sp.label for sp in with_currency.species.values()}
        )

    def test_unknown_module_raises(self):
        with self.assertRaises(KeyError):
            build_kegg_crn("M99999")


class TestAnalyzeKeggModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_kegg_module("M00001")

    def test_returns_a_report(self):
        self.assertIsInstance(self.result, KeggModuleAnalysis)
        self.assertEqual(self.result.module_id, "M00001")
        self.assertIn("Glycolysis", self.result.module_name)

    def test_deficiency_identity_holds(self):
        self.assertEqual(
            self.result.deficiency,
            self.result.n_complexes
            - self.result.n_linkage_classes
            - self.result.rank,
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
        self.assertIn(
            frozenset({"Oxidized ferredoxin", "Reduced ferredoxin"}), pools
        )

    def test_glucose_is_a_siphon(self):
        # Glucose is consumed and never produced: the module depends on it.
        self.assertIn(["alpha-D-Glucose"], self.result.siphons)

    def test_a_catabolic_module_is_not_persistent(self):
        self.assertFalse(self.result.persistent)

    def test_uncovered_siphons_are_a_subset_of_siphons(self):
        for siphon in self.result.uncovered_siphons:
            self.assertIn(siphon, self.result.siphons)

    def test_report_serializes(self):
        payload = self.result.to_dict()
        self.assertEqual(payload["module_id"], "M00001")
        self.assertIn("moieties", payload)

    def test_report_is_readable(self):
        text = str(self.result)
        self.assertIn("M00001", text)
        self.assertIn("conserved moieties", text)

    def test_every_case_study_module_analyses(self):
        for module_id in CASE_STUDY_MODULES:
            with self.subTest(module=module_id):
                result = analyze_kegg_module(module_id)
                self.assertGreater(result.n_species, 0)
                self.assertGreaterEqual(result.deficiency, 0)


class TestGlycolysisFluxRealizability(unittest.TestCase):
    def test_the_canonical_flux_is_realizable_from_one_glucose(self):
        result = check_glycolysis_flux()
        self.assertTrue(result["realizable"])
        self.assertIsNotNone(result["certificate"])

    def test_the_certificate_fires_each_reaction_the_requested_number_of_times(self):
        result = check_glycolysis_flux()
        fired = {}
        for reaction in result["certificate"]:
            fired[reaction] = fired.get(reaction, 0) + 1
        self.assertEqual(fired, GLYCOLYSIS_FLUX)

    def test_the_certificate_starts_with_hexokinase(self):
        # Nothing else can fire from glucose alone.
        self.assertEqual(check_glycolysis_flux()["certificate"][0], "R01786")

    def test_a_flux_needing_absent_substrate_is_not_realizable(self):
        # Firing the lower half without ever making a triose cannot work.
        result = check_glycolysis_flux(flux={"R00200": 1}, initial_glucose=1)
        self.assertFalse(result["realizable"])


class TestCaseStudyReport(unittest.TestCase):
    def test_text_report_covers_every_module(self):
        text = kegg_case_study_report()
        for module_id in CASE_STUDY_MODULES:
            self.assertIn(module_id, text)

    def test_markdown_report_is_a_table(self):
        table = kegg_case_study_report(fmt="markdown")
        self.assertIn("| module", table)
        self.assertIn("delta", table)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            kegg_case_study_report(fmt="latex")


if __name__ == "__main__":
    unittest.main()
