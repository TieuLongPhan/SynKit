from __future__ import annotations

import unittest

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Petrinet.net import PetriNet
from synkit.CRN.Petrinet.persistence import (
    PersistenceCheckResult,
    siphon_persistence_condition,
    siphon_persistence_details,
)


class TestPersistenceCheckResult(unittest.TestCase):
    """Unit tests for the structured persistence result."""

    def test_dataclass_fields(self) -> None:
        res = PersistenceCheckResult(
            persistence_ok=True,
            siphons=[{"A"}],
            semiflow_supports=[{"A", "B"}],
            uncovered_siphons=[],
        )

        self.assertTrue(res.persistence_ok)
        self.assertEqual(res.siphons, [{"A"}])
        self.assertEqual(res.semiflow_supports, [{"A", "B"}])
        self.assertEqual(res.uncovered_siphons, [])


class TestSiphonPersistence(unittest.TestCase):
    """Unit tests for siphon/P-semiflow persistence checks."""

    def test_condition_matches_details(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B"])

        cond = siphon_persistence_condition(syn)
        details = siphon_persistence_details(syn)

        self.assertEqual(cond, details.persistence_ok)

    def test_simple_conversion_is_persistent_by_condition(self) -> None:
        """
        A >> B

        Minimal siphon is {A}. The P-semiflow support is {A, B}, which is not
        contained in {A}, so the sufficient condition should fail.
        """
        syn = SynCRN.from_reaction_strings(["A>>B"])

        details = siphon_persistence_details(syn)

        self.assertFalse(details.persistence_ok)
        self.assertTrue(details.siphons)
        self.assertTrue(details.uncovered_siphons)

    def test_reversible_pair_is_persistent_by_condition(self) -> None:
        """
        A <-> B

        Minimal siphon is the full species set, and the P-semiflow support is also
        the full species set, so the sufficient condition should hold.
        """
        syn = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>A",
            ]
        )

        details = siphon_persistence_details(syn)

        self.assertTrue(details.persistence_ok)
        self.assertGreaterEqual(len(details.siphons), 1)
        self.assertGreaterEqual(len(details.semiflow_supports), 1)
        self.assertEqual(details.uncovered_siphons, [])

        net = PetriNet.from_syncrn(syn)

        siphon_sets = {frozenset(net.place_name(p) for p in s) for s in details.siphons}
        support_sets = {
            frozenset(net.place_name(p) for p in s) for s in details.semiflow_supports
        }

        self.assertIn(frozenset({"A", "B"}), siphon_sets)
        self.assertIn(frozenset({"A", "B"}), support_sets)

    def test_uncovered_siphons_reported(self) -> None:
        """
        A >> B should produce at least one uncovered siphon.
        """
        syn = SynCRN.from_reaction_strings(["A>>B"])

        details = siphon_persistence_details(syn)

        self.assertFalse(details.persistence_ok)
        self.assertGreaterEqual(len(details.uncovered_siphons), 1)
        for s in details.uncovered_siphons:
            self.assertIsInstance(s, set)
            self.assertTrue(s)

    def test_details_returns_sets(self) -> None:
        syn = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>A",
            ]
        )
        details = siphon_persistence_details(syn)

        for s in details.siphons:
            self.assertIsInstance(s, set)
        for s in details.semiflow_supports:
            self.assertIsInstance(s, set)
        for s in details.uncovered_siphons:
            self.assertIsInstance(s, set)

    def test_max_siphon_size_can_change_result(self) -> None:
        """
        For A <-> B, the minimal siphon has size 2. Limiting max_siphon_size=1
        suppresses it, so the function should vacuously return True.
        """
        syn = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>A",
            ]
        )

        details = siphon_persistence_details(syn, max_siphon_size=1)

        self.assertTrue(details.persistence_ok)
        self.assertEqual(details.siphons, [])
        self.assertEqual(details.semiflow_supports, [])
        self.assertEqual(details.uncovered_siphons, [])

    def test_accepts_petrinet_input(self) -> None:
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        details = siphon_persistence_details(net)

        self.assertTrue(details.persistence_ok)
        self.assertEqual(details.uncovered_siphons, [])

    def test_no_p_semiflow_case_returns_false_when_siphons_exist(self) -> None:
        """
        A >> 2B has no nontrivial P-semiflow in the usual sense, while siphons
        can still exist. The implementation should then return False with all
        siphons marked uncovered.
        """
        syn = SynCRN.from_reaction_strings(["A>>2B"])

        details = siphon_persistence_details(syn)

        if details.siphons:
            self.assertFalse(details.persistence_ok)
            if not details.semiflow_supports:
                self.assertEqual(
                    {frozenset(s) for s in details.uncovered_siphons},
                    {frozenset(s) for s in details.siphons},
                )

    def test_support_tol_parameter_is_used(self) -> None:
        """
        Smoke test that support_tol is accepted and computation succeeds.
        """
        syn = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>A",
            ]
        )

        details = siphon_persistence_details(syn, support_tol=1e-6)

        self.assertIsInstance(details, PersistenceCheckResult)

    def test_boolean_wrapper(self) -> None:
        syn_good = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        syn_bad = SynCRN.from_reaction_strings(["A>>B"])

        self.assertTrue(siphon_persistence_condition(syn_good))
        self.assertFalse(siphon_persistence_condition(syn_bad))


if __name__ == "__main__":
    unittest.main()
