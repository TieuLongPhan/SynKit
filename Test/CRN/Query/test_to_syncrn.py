"""KEGG equation conversion through the public CRN API."""

import unittest

from synkit.CRN.Query.to_syncrn import (
    CURRENCY_COMPOUNDS,
    syncrn_from_kegg_equations,
)


class TestKeggToSynCRN(unittest.TestCase):
    def test_irreversible_equation(self):
        crn = syncrn_from_kegg_equations({"R1": "C00001 => C00002"})
        self.assertEqual(crn.n_species, 2)
        self.assertEqual(crn.n_reactions, 1)

    def test_reversible_equation_is_split(self):
        crn = syncrn_from_kegg_equations({"R1": "C00001 <=> C00002"})
        self.assertEqual(crn.n_reactions, 2)

    def test_reversible_expansion_can_be_disabled(self):
        crn = syncrn_from_kegg_equations({"R1": "C00001 <=> C00002"}, expand_reversible=False)
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
        self.assertEqual(crn.to_equations(species="label", include_id=False), ["DHAP >> GAP"])

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
        self.assertEqual([r.label for r in crn.reactions.values()], ["R01015"])
