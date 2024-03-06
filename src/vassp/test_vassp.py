import unittest

import numpy
from pyrepo_mcda.mcda_methods import VIKOR
import numpy as np
from vassp import VASSP


class TestVASSP(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scoeff_0(self):
        """
        Test the case where the s_coeffs are set to 0. The result should be the same as VIKOR.
        """
        dm = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        impacts = np.array([1, 1, 1])
        weights = np.array([0.33, 0.33, 0.34])

        vikor = VIKOR(v=0.5)
        vassp = VASSP()
        scores_vassp = vassp(dm, weights, impacts, v=0.5, s_coeffs=np.array([0]))
        scores_vikor = vikor(dm, weights, impacts)
        self.assertTrue(numpy.array_equal(scores_vikor, scores_vassp), 'VASSP s=0 should have the same results as VIKOR')

    def test_scoeff_1(self):
        """
        Test the case where the s_coeffs are set to 1. The result should be different from VIKOR.
        """
        dm = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        impacts = np.array([1, 1, 1])
        weights = np.array([0.33, 0.33, 0.34])

        vikor = VIKOR(v=0.5)
        vassp = VASSP()
        scores_vassp = vassp(dm, weights, impacts, v=0.5, s_coeffs=np.array([1]))
        scores_vikor = vikor(dm, weights, impacts)
        self.assertFalse(numpy.array_equal(scores_vikor, scores_vassp), 'VASSP s=1 should have different results from VIKOR')


if __name__ == '__main__':
    unittest.main()
