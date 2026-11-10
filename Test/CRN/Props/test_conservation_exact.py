"""Conservation-law regression across the reference network fixtures."""

import unittest

import numpy as np

from Experiment.CRN.studies import BENCHMARK_NETWORKS
from synkit.CRN.Props.stoich import integer_conservation_laws, stoichiometric_matrix


class TestConservationLawsAreExact(unittest.TestCase):
    """Regression: the float-kernel path returned vectors that were not laws."""

    def test_every_benchmark_law_is_in_the_left_kernel(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                crn = entry.build()
                matrix = stoichiometric_matrix(crn)
                if matrix.size == 0:
                    continue
                for law in integer_conservation_laws(crn):
                    np.testing.assert_allclose(np.array(law, dtype=float) @ matrix, 0.0, atol=1e-12)
