from __future__ import annotations

import unittest

import numpy as np

from synkit.CRN.Petrinet.minimal_semiflows import (
    is_semiflow,
    minimal_semiflow_supports,
    minimal_semiflows,
)
from synkit.CRN.Petrinet.persistence import siphon_persistence_details
from synkit.CRN.Structure import SynCRN


class TestMinimalSemiflows(unittest.TestCase):
    """Unit tests for exact minimal non-negative integer semiflows."""

    def test_reversible_pair(self) -> None:
        """A <-> B has the single conservation law A + B."""
        s = np.array([[-1.0, 1.0], [1.0, -1.0]])

        self.assertEqual(minimal_semiflows(s, kind="p"), [[1, 1]])
        self.assertEqual(minimal_semiflows(s, kind="t"), [[1, 1]])

    def test_disjoint_blocks_give_disjoint_supports(self) -> None:
        """
        Two independent conservation laws must come back separated.

        This is the case an SVD kernel basis gets wrong: it returns rotated,
        mixed-sign vectors whose supports are the union of both blocks.
        """
        # rows A, B, C, D ; cols (A>>B), (C>>D)
        s = np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ]
        )
        flows = minimal_semiflows(s, kind="p")

        self.assertEqual(sorted(flows), [[0, 0, 1, 1], [1, 1, 0, 0]])

    def test_stoichiometric_weights(self) -> None:
        """A >> 2B conserves 2A + B, not A + B."""
        s = np.array([[-1.0], [2.0]])

        self.assertEqual(minimal_semiflows(s, kind="p"), [[2, 1]])

    def test_cycle_has_t_semiflow(self) -> None:
        """A >> B >> C >> A returns to its start after one firing each."""
        s = np.array(
            [
                [-1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],
            ]
        )

        self.assertEqual(minimal_semiflows(s, kind="t"), [[1, 1, 1]])

    def test_irreversible_step_has_no_t_semiflow(self) -> None:
        """A single irreversible step cannot return to its initial marking."""
        s = np.array([[-1.0], [1.0]])

        self.assertEqual(minimal_semiflows(s, kind="t"), [])

    def test_all_results_are_valid_semiflows(self) -> None:
        """Every returned vector must be non-negative and in the right kernel."""
        s = np.array(
            [
                [-2.0, 0.0, 0.0],
                [1.0, -2.0, 0.0],
                [3.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )

        for kind in ("p", "t"):
            for flow in minimal_semiflows(s, kind=kind):
                self.assertTrue(all(v >= 0 for v in flow))
                self.assertTrue(is_semiflow(s, flow, kind=kind))

    def test_supports_are_inclusion_minimal(self) -> None:
        """No returned support may strictly contain another."""
        s = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, -1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        )
        supports = [
            {i for i, v in enumerate(flow) if v != 0}
            for flow in minimal_semiflows(s, kind="p")
        ]

        for a in supports:
            for b in supports:
                if a is not b:
                    self.assertFalse(b < a, f"{b} strictly inside {a}")

    def test_empty_matrix(self) -> None:
        """An empty network has no semiflows and must not raise."""
        self.assertEqual(minimal_semiflows(np.zeros((0, 0)), kind="p"), [])

    def test_rejects_non_integer_matrix(self) -> None:
        """Fractional stoichiometry is rejected rather than silently rounded."""
        with self.assertRaises(ValueError):
            minimal_semiflows(np.array([[-0.5], [1.0]]), kind="p")

    def test_rejects_bad_kind(self) -> None:
        """Only 'p' and 't' are accepted."""
        with self.assertRaises(ValueError):
            minimal_semiflows(np.array([[-1.0], [1.0]]), kind="x")

    def test_supports_with_labels(self) -> None:
        """Labelled supports map names onto coefficients."""
        s = np.array([[-1.0, 1.0], [1.0, -1.0]])

        self.assertEqual(
            minimal_semiflow_supports(s, kind="p", labels=["A", "B"]),
            [{"A": 1, "B": 1}],
        )

    def test_supports_reject_mismatched_labels(self) -> None:
        """A label list of the wrong length is an error, not a silent slice."""
        s = np.array([[-1.0, 1.0], [1.0, -1.0]])

        with self.assertRaises(ValueError):
            minimal_semiflow_supports(s, kind="p", labels=["A"])

    def test_is_semiflow_rejects_negative_vector(self) -> None:
        """A mixed-sign kernel vector is not a semiflow."""
        s = np.array([[-1.0, 1.0], [1.0, -1.0]])

        self.assertTrue(is_semiflow(s, [1, 1], kind="p"))
        self.assertFalse(is_semiflow(s, [-1, -1], kind="p"))
        self.assertFalse(is_semiflow(s, [0, 0], kind="p"))


class TestPersistenceRegression(unittest.TestCase):
    """Networks that the kernel-basis implementation wrongly called transient."""

    def _ok(self, rxns: list[str]) -> bool:
        return siphon_persistence_details(
            SynCRN.from_reaction_strings(rxns)
        ).persistence_ok

    def test_disjoint_reversible_pairs_are_persistent(self) -> None:
        """
        Each minimal siphon {A,B}, {C,D}, {E,F} contains a P-semiflow support.
        """
        self.assertTrue(
            self._ok(["A>>B", "B>>A", "C>>D", "D>>C", "E>>F", "F>>E"])
        )

    def test_stoichiometric_reversible_block_is_persistent(self) -> None:
        """Non-unit stoichiometry must not change the verdict."""
        self.assertTrue(
            self._ok(["A>>B", "B>>A", "C>>D", "D>>C", "E>>2F", "2F>>E"])
        )

    def test_reversible_chain_plus_pair_is_persistent(self) -> None:
        """A three-species reversible chain alongside a reversible pair."""
        self.assertTrue(
            self._ok(["A>>B", "B>>A", "B>>C", "C>>B", "D>>E", "E>>D"])
        )

    def test_open_network_is_not_persistent(self) -> None:
        """A >> B with no way back leaves siphon {A} uncovered."""
        self.assertFalse(self._ok(["A>>B"]))


if __name__ == "__main__":
    unittest.main()
