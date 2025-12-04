import unittest
import numpy as np

from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph
from synkit.CRN.Props.thermo import compute_thermo_summary


# -------------------------------------------------------------------
# Additional Complex CRN Examples
# -------------------------------------------------------------------
def crn_cycle_with_branch():
    # Reversible cycle A <-> B <-> C plus branching C -> D
    rxns = ["A>>B", "B>>A", "B>>C", "C>>B", "C>>D"]
    return rxns_to_hypergraph(rxns)


def crn_partial_conservation():
    # conserved subsystem A<->B, but X degrades
    rxns = ["A>>B", "B>>A", "X>>"]  # sink destroys X
    return rxns_to_hypergraph(rxns)


def crn_two_independent_cycles():
    # Cycle 1: A <-> B
    # Cycle 2: C <-> D
    rxns = ["A>>B", "B>>A", "C>>D", "D>>C"]
    return rxns_to_hypergraph(rxns)


def crn_large_network():
    # Synthetic slightly larger CRN
    rxns = [
        "A>>B",
        "B>>C",
        "C>>A",  # cycle 1
        "D>>E",
        "E>>F",
        "F>>D",  # cycle 2
        "X>>Y",  # irreversible edge
        "C+E>>Z",  # cross coupling
        "Z>>A+F",  # feeds cycles
    ]
    return rxns_to_hypergraph(rxns)


# -------------------------------------------------------------------
# Complex Test Suite
# -------------------------------------------------------------------
class TestThermoComplex(unittest.TestCase):

    # ---------------------------------------------------------------
    # 1. Reversible cycle with branching
    # ---------------------------------------------------------------
    def test_cycle_with_branch(self):
        H = crn_cycle_with_branch()
        summary = compute_thermo_summary(H)

        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertTrue(summary.conservative)

    # ---------------------------------------------------------------
    # 2. Partially conservative system
    # ---------------------------------------------------------------
    def test_partial_conservation(self):
        H = crn_partial_conservation()
        summary = compute_thermo_summary(H)

        # A <-> B is conservative
        # X is not conserved
        self.assertFalse(summary.conservative)
        self.assertTrue(summary.irreversible_futile_cycles)

    # ---------------------------------------------------------------
    # 3. Two independent cycles (kernel dimension >= 2)
    # ---------------------------------------------------------------
    def test_two_independent_cycles(self):
        H = crn_two_independent_cycles()
        summary = compute_thermo_summary(H)

        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertFalse(summary.conservative)
        self.assertIsNone(summary.example_conservation_law)

    # ---------------------------------------------------------------
    # 4. moderately large synthetic test
    # ---------------------------------------------------------------
    def test_large_network(self):
        H = crn_large_network()
        summary = compute_thermo_summary(H)

        self.assertIsInstance(summary.conservative, bool)
        self.assertIsInstance(summary.irreversible_futile_cycles, bool)

        # Some cycles definitely exist
        self.assertTrue(summary.irreversible_futile_cycles)

        # Conservativity depends on full stoichiometry
        if summary.conservative:
            self.assertIsNotNone(summary.example_conservation_law)
            self.assertGreater(summary.example_conservation_law.sum(), 0)
        else:
            self.assertIsNone(summary.example_conservation_law)


if __name__ == "__main__":
    unittest.main()
