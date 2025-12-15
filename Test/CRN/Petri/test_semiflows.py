import unittest
import numpy as np

from synkit.CRN.Props.stoich import stoichiometric_matrix
from synkit.CRN.Petri import find_p_semiflows, find_t_semiflows
from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph


class TestSemiflows(unittest.TestCase):
    def setUp(self) -> None:
        rxns = [
            "2A + B >> C",
            "C + D >> E",
            "E + F >> D + G",
        ]
        self.G = rxns_to_hypergraph(rxns)
        self.S = stoichiometric_matrix(self.G)

    def test_compute_P_T_semiflows(self) -> None:
        P = find_p_semiflows(self.G)
        T = find_t_semiflows(self.G)

        self.assertIsNotNone(P)
        self.assertIsNotNone(T)

        # P: left kernel
        self.assertTrue(np.allclose(self.S.T @ P, 0.0, atol=1e-8))
        # T: right kernel
        if T.size > 0:
            self.assertTrue(np.allclose(self.S @ T, 0.0, atol=1e-8))
