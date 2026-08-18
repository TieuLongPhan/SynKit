from __future__ import annotations

import unittest

from synkit.CRN.Construct.builder import CRNExpand


class TestReactionDelta(unittest.TestCase):
    """Multiset semantics for the reaction delta and the recorded sides."""

    def setUp(self) -> None:
        self.crn = CRNExpand(rules=[], keep_aam=False)
        for smi in ("CCO", "CC=O", "[Pd]"):
            self.crn._add_species_node(smi)
        self.ids = {
            self.crn.graph.nodes[n]["smiles"]: n for n in self.crn.graph.nodes
        }

    def test_catalyst_is_kept_on_both_sides(self) -> None:
        """
        A catalyst must retain its incidence edges.

        Regression test: set-based cancellation recorded
        ``EtOH + Pd >> AcH + Pd`` as ``EtOH >> AcH``, so the catalyst never
        received an edge and was invisible to every downstream analysis.
        """
        reactants = [self.ids["CCO"], self.ids["[Pd]"]]
        products = ["CC=O", "[Pd]"]

        r_side, p_side = self.crn._reaction_sides(reactants, products)

        self.assertIn(self.ids["[Pd]"], r_side)
        self.assertIn("[Pd]", p_side)

    def test_catalyst_cancels_in_the_delta(self) -> None:
        """The delta still reports only the net transformation."""
        reactants = [self.ids["CCO"], self.ids["[Pd]"]]
        products = ["CC=O", "[Pd]"]

        self.assertEqual(
            self.crn._reaction_delta(reactants, products),
            (("CCO",), ("CC=O",)),
        )

    def test_repeated_reactant_keeps_multiplicity(self) -> None:
        """
        ``2 EtOH >> EtOH + AcH`` must keep both reactant copies.

        Regression test: set-based cancellation emptied the reactant side, and
        with ``allow_empty_side=False`` the reaction was silently discarded.
        """
        reactants = [self.ids["CCO"], self.ids["CCO"]]
        products = ["CCO", "CC=O"]

        r_side, p_side = self.crn._reaction_sides(reactants, products)

        self.assertEqual(len(r_side), 2)
        self.assertEqual(sorted(p_side), ["CC=O", "CCO"])

    def test_repeated_reactant_delta_is_not_empty(self) -> None:
        """The net change of ``2A >> A + B`` is ``A >> B``, not nothing."""
        reactants = [self.ids["CCO"], self.ids["CCO"]]
        products = ["CCO", "CC=O"]

        self.assertEqual(
            self.crn._reaction_delta(reactants, products),
            (("CCO",), ("CC=O",)),
        )

    def test_true_no_change_has_empty_delta(self) -> None:
        """An identity transformation still cancels completely."""
        self.assertEqual(
            self.crn._reaction_delta([self.ids["CCO"]], ["CCO"]),
            ((), ()),
        )

    def test_strip_spectators_restores_legacy_behaviour(self) -> None:
        """The opt-in flag reproduces the pre-1.6.3 cancellation."""
        legacy = CRNExpand(rules=[], keep_aam=False, strip_spectators=True)
        for smi in ("CCO", "CC=O", "[Pd]"):
            legacy._add_species_node(smi)
        ids = {legacy.graph.nodes[n]["smiles"]: n for n in legacy.graph.nodes}

        r_side, p_side = legacy._reaction_sides(
            [ids["CCO"], ids["[Pd]"]], ["CC=O", "[Pd]"]
        )

        self.assertEqual(r_side, [ids["CCO"]])
        self.assertEqual(p_side, ["CC=O"])


if __name__ == "__main__":
    unittest.main()
