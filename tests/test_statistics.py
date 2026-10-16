"""
Unit Tests for Statistical Testing
===================================
Tests McNemar's test with Edwards' continuity correction and paired DeLong's test.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itcf.evaluation import (
    compute_mcnemar_continuity_corrected,
    paired_delong_test
)


class TestStatisticalTests(unittest.TestCase):
    def test_mcnemar_continuity_correction(self):
        # Discordant counts from Table 6(i): b = 0, c = 6
        # chi^2 = (|0 - 6| - 1)^2 / (0 + 6) = 25 / 6 = 4.1667
        chi2, p = compute_mcnemar_continuity_corrected(0, 6)
        self.assertAlmostEqual(chi2, 4.1667, places=3)
        self.assertAlmostEqual(p, 0.0412, places=3)
        self.assertLess(p, 0.05)

    def test_paired_delong(self):
        np.random.seed(42)
        y_true = np.array([1]*50 + [0]*50)
        # Model B is slightly better than Model A
        pred_a = np.clip(y_true * 0.7 + np.random.normal(0, 0.3, 100), 0, 1)
        pred_b = np.clip(y_true * 0.9 + np.random.normal(0, 0.2, 100), 0, 1)
        
        auc_a, auc_b, z, p = paired_delong_test(y_true, pred_a, pred_b)
        self.assertGreater(auc_b, auc_a)
        self.assertGreater(z, 0.0)


if __name__ == '__main__':
    unittest.main()
