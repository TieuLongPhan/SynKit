from __future__ import annotations

import unittest

import numpy as np

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Petrinet.analyzer import PetriAnalyzer, PetriSummary
from synkit.CRN.Petrinet.net import PetriNet
from synkit.CRN.Petrinet.persistence import (
    PersistenceCheckResult,
    siphon_persistence_condition,
    siphon_persistence_details,
)
from synkit.CRN.Petrinet.semiflows import find_p_semiflows, find_t_semiflows
from synkit.CRN.Petrinet.structure import find_siphons, find_traps


class TestPetriSummary(unittest.TestCase):
    """Unit tests for the PetriSummary dataclass."""

    def test_dataclass_fields(self) -> None:
        p = np.array([[1.0], [1.0]])
        t = np.array([[1.0], [1.0]])

        summary = PetriSummary(
            p_semiflows=p,
            t_semiflows=t,
            siphons=[{"A", "B"}],
            traps=[{"A", "B"}],
            persistence_ok=True,
            place_order=["A", "B"],
            transition_order=["r1", "r2"],
        )

        self.assertTrue(np.allclose(summary.p_semiflows, p))
        self.assertTrue(np.allclose(summary.t_semiflows, t))
        self.assertEqual(summary.siphons, [{"A", "B"}])
        self.assertEqual(summary.traps, [{"A", "B"}])
        self.assertTrue(summary.persistence_ok)
        self.assertEqual(summary.place_order, ["A", "B"])
        self.assertEqual(summary.transition_order, ["r1", "r2"])


class TestPetriAnalyzer(unittest.TestCase):
    """Unit tests for the PetriAnalyzer OOP wrapper."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.reversible = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>A",
            ]
        )
        cls.irreversible = SynCRN.from_reaction_strings(["A>>B"])

    def test_init_from_syncrn(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)

        self.assertIs(analyzer._crn, self.reversible)
        self.assertIsInstance(analyzer.petri, PetriNet)
        self.assertEqual(analyzer.p_semiflows, None)
        self.assertEqual(analyzer.t_semiflows, None)
        self.assertEqual(analyzer.siphons, None)
        self.assertEqual(analyzer.traps, None)
        self.assertEqual(analyzer.persistence_ok, None)
        self.assertEqual(analyzer.persistence_details, None)

    def test_init_from_petrinet(self) -> None:
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        analyzer = PetriAnalyzer(net)

        self.assertIs(analyzer.petri, net)
        self.assertIs(analyzer._crn, net)

    def test_orders(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        places, transitions = analyzer._orders()

        self.assertIsInstance(places, list)
        self.assertIsInstance(transitions, list)
        self.assertGreaterEqual(len(places), 1)
        self.assertGreaterEqual(len(transitions), 1)

    def test_ensure_persistence_computed_false_initially(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        self.assertFalse(analyzer._ensure_persistence_computed())

    def test_ensure_persistence_computed_true_after_check(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).check_persistence()
        self.assertTrue(analyzer._ensure_persistence_computed())

    def test_persistence_details_as_dict_none_initially(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        self.assertIsNone(analyzer._persistence_details_as_dict())

    def test_compute_semiflows(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        out = analyzer.compute_semiflows()

        self.assertIs(out, analyzer)
        self.assertIsInstance(analyzer.p_semiflows, np.ndarray)
        self.assertIsInstance(analyzer.t_semiflows, np.ndarray)
        self.assertEqual(analyzer.p_semiflows.shape[0], 2)
        self.assertEqual(analyzer.t_semiflows.shape[0], 2)

    def test_compute_siphons_traps(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        out = analyzer.compute_siphons_traps()

        self.assertIs(out, analyzer)
        self.assertIsInstance(analyzer.siphons, list)
        self.assertIsInstance(analyzer.traps, list)
        self.assertGreaterEqual(len(analyzer.siphons), 1)
        self.assertGreaterEqual(len(analyzer.traps), 1)

    def test_check_persistence(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        out = analyzer.check_persistence()

        self.assertIs(out, analyzer)
        self.assertIsInstance(analyzer.persistence_ok, bool)
        self.assertIsInstance(analyzer.persistence_details, PersistenceCheckResult)

    def test_compute_all(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        out = analyzer.compute_all()

        self.assertIs(out, analyzer)
        self.assertIsNotNone(analyzer.p_semiflows)
        self.assertIsNotNone(analyzer.t_semiflows)
        self.assertIsNotNone(analyzer.siphons)
        self.assertIsNotNone(analyzer.traps)
        self.assertIsNotNone(analyzer.persistence_ok)
        self.assertIsNotNone(analyzer.persistence_details)

    def test_summary_none_before_all_components_ready(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)

        self.assertIsNone(analyzer.summary)

        analyzer.compute_semiflows()
        self.assertIsNone(analyzer.summary)

        analyzer.compute_siphons_traps()
        self.assertIsNone(analyzer.summary)

    def test_summary_after_compute_all(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_all()
        summary = analyzer.summary

        self.assertIsInstance(summary, PetriSummary)
        self.assertIsInstance(summary.p_semiflows, np.ndarray)
        self.assertIsInstance(summary.t_semiflows, np.ndarray)
        self.assertIsInstance(summary.siphons, list)
        self.assertIsInstance(summary.traps, list)
        self.assertIsInstance(summary.persistence_ok, bool)
        self.assertIsInstance(summary.place_order, list)
        self.assertIsInstance(summary.transition_order, list)

    def test_summary_place_and_transition_orders_match_orders_method(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_all()
        summary = analyzer.summary
        assert summary is not None

        places, transitions = analyzer._orders()
        self.assertEqual(summary.place_order, places)
        self.assertEqual(summary.transition_order, transitions)

    def test_as_dict_before_computation(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        payload = analyzer.as_dict()

        self.assertIn("place_order", payload)
        self.assertIn("transition_order", payload)
        self.assertIn("p_semiflows", payload)
        self.assertIn("t_semiflows", payload)
        self.assertIn("siphons", payload)
        self.assertIn("traps", payload)
        self.assertIn("persistence_ok", payload)
        self.assertIn("persistence_details", payload)

        self.assertIsNone(payload["p_semiflows"])
        self.assertIsNone(payload["t_semiflows"])
        self.assertIsNone(payload["siphons"])
        self.assertIsNone(payload["traps"])
        self.assertIsNone(payload["persistence_ok"])
        self.assertIsNone(payload["persistence_details"])

    def test_as_dict_after_compute_all(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_all()
        payload = analyzer.as_dict()

        self.assertIsInstance(payload["place_order"], list)
        self.assertIsInstance(payload["transition_order"], list)
        self.assertIsInstance(payload["p_semiflows"], list)
        self.assertIsInstance(payload["t_semiflows"], list)
        self.assertIsInstance(payload["siphons"], list)
        self.assertIsInstance(payload["traps"], list)
        self.assertIsInstance(payload["persistence_ok"], bool)
        self.assertIsInstance(payload["persistence_details"], dict)

    def test_persistence_details_as_dict_after_check(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).check_persistence()
        d = analyzer._persistence_details_as_dict()

        self.assertIsInstance(d, dict)
        self.assertIn("persistence_ok", d)
        self.assertIn("siphons", d)
        self.assertIn("semiflow_supports", d)
        self.assertIn("uncovered_siphons", d)

    def test_explain_before_computation(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        msg = analyzer.explain()

        self.assertIn("No Petri computations performed yet", msg)

    def test_explain_after_compute_all(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_all()
        msg = analyzer.explain()

        self.assertIn("persistence_ok=", msg)
        self.assertIn("p_semiflows=", msg)
        self.assertIn("t_semiflows=", msg)
        self.assertIn("siphons=", msg)
        self.assertIn("traps=", msg)

    def test_repr_before_check(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        self.assertEqual(repr(analyzer), "<PetriAnalyzer persistence_ok=NA>")

    def test_repr_after_true_check(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).check_persistence()
        self.assertEqual(repr(analyzer), "<PetriAnalyzer persistence_ok=True>")

    def test_repr_after_false_check(self) -> None:
        analyzer = PetriAnalyzer(self.irreversible).check_persistence()
        self.assertEqual(repr(analyzer), "<PetriAnalyzer persistence_ok=False>")

    def test_reversible_pair_values(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_all()

        self.assertTrue(analyzer.persistence_ok)
        self.assertIsNotNone(analyzer.siphons)
        self.assertIsNotNone(analyzer.traps)
        self.assertIsNotNone(analyzer.p_semiflows)
        self.assertIsNotNone(analyzer.t_semiflows)

        self.assertGreaterEqual(analyzer.p_semiflows.shape[1], 1)
        self.assertGreaterEqual(analyzer.t_semiflows.shape[1], 1)

    def test_irreversible_pair_values(self) -> None:
        analyzer = PetriAnalyzer(self.irreversible).compute_all()

        self.assertFalse(analyzer.persistence_ok)
        self.assertIsNotNone(analyzer.p_semiflows)
        self.assertIsNotNone(analyzer.t_semiflows)

        self.assertGreaterEqual(analyzer.p_semiflows.shape[1], 1)
        self.assertEqual(analyzer.t_semiflows.shape[1], 0)

    def test_method_chaining(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        out = analyzer.compute_semiflows().compute_siphons_traps().check_persistence()

        self.assertIs(out, analyzer)
        self.assertIsNotNone(analyzer.p_semiflows)
        self.assertIsNotNone(analyzer.siphons)
        self.assertIsNotNone(analyzer.persistence_ok)

    def test_compute_semiflows_matches_module_functions(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_semiflows()

        expected_p = find_p_semiflows(self.reversible, rtol=1e-12)
        expected_t = find_t_semiflows(self.reversible, rtol=1e-12)

        self.assertTrue(np.allclose(analyzer.p_semiflows, expected_p))
        self.assertTrue(np.allclose(analyzer.t_semiflows, expected_t))

    def test_compute_siphons_traps_matches_module_functions(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).compute_siphons_traps()

        expected_siphons = find_siphons(
            analyzer.petri,
            max_size=None,
            names="label",
        )
        expected_traps = find_traps(
            analyzer.petri,
            max_size=None,
            names="label",
        )

        self.assertEqual(
            {frozenset(x) for x in analyzer.siphons or []},
            {frozenset(x) for x in expected_siphons},
        )
        self.assertEqual(
            {frozenset(x) for x in analyzer.traps or []},
            {frozenset(x) for x in expected_traps},
        )

    def test_check_persistence_matches_module_functions(self) -> None:
        analyzer = PetriAnalyzer(self.reversible).check_persistence()

        expected_ok = siphon_persistence_condition(self.reversible, rtol=1e-12)
        expected_details = siphon_persistence_details(self.reversible, rtol=1e-12)

        self.assertEqual(analyzer.persistence_ok, expected_ok)
        self.assertEqual(
            analyzer.persistence_details.persistence_ok,
            expected_details.persistence_ok,
        )
        self.assertEqual(
            {frozenset(x) for x in analyzer.persistence_details.siphons},
            {frozenset(x) for x in expected_details.siphons},
        )

    def test_max_siphon_size_affects_siphon_trap_enumeration(self) -> None:
        analyzer = PetriAnalyzer(
            self.reversible,
            max_siphon_size=1,
        ).compute_siphons_traps()

        self.assertEqual(analyzer.siphons, [])
        self.assertEqual(analyzer.traps, [])

    def test_petri_property_returns_internal_net(self) -> None:
        analyzer = PetriAnalyzer(self.reversible)
        self.assertIs(analyzer.petri, analyzer._petri)

    def test_orders_work_for_petrinet_input(self) -> None:
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        analyzer = PetriAnalyzer(net)
        places, transitions = analyzer._orders()

        self.assertEqual(set(places), {"A", "B"})
        self.assertEqual(transitions, ["t1", "t2"])

    def test_compute_all_on_petrinet_input(self) -> None:
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        analyzer = PetriAnalyzer(net).compute_all()

        self.assertTrue(analyzer.persistence_ok)
        self.assertIsNotNone(analyzer.summary)
        self.assertEqual(set(analyzer.summary.place_order), {"A", "B"})
        self.assertEqual(analyzer.summary.transition_order, ["t1", "t2"])


if __name__ == "__main__":
    unittest.main()
