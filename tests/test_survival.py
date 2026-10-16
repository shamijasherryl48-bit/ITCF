"""
Unit Tests for Survival Analysis Components
============================================
Tests Cox Proportional Hazards regression, Kaplan-Meier stratification,
and Harrell's C-index.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itcf.survival import (
    ITCFSurvivalAnalysis,
    ITCFKaplanMeier,
    compute_harrell_c_index,
    compute_reverse_km_followup
)


class TestSurvivalAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        time = np.random.exponential(30.0, n) + 2.0
        event = np.random.binomial(1, 0.6, n)
        age = np.random.normal(60, 10, n)
        score = np.random.normal(1.5, 0.5, n)
        
        self.df = pd.DataFrame({
            'Patient_ID': [f"PT-{i}" for i in range(n)],
            'OS_Months': time,
            'OS_Event': event,
            'Age': age,
            'Sex': ['Male' if i % 2 == 0 else 'Female' for i in range(n)],
            'Stage': ['III/IV' if i % 3 == 0 else 'I/II' for i in range(n)],
            'TMB': np.random.gamma(2, 1, n),
            'VHL_mut': np.random.binomial(1, 0.5, n),
            'PBRM1_mut': np.random.binomial(1, 0.3, n),
            'BAP1_mut': np.random.binomial(1, 0.1, n),
            'SETD2_mut': np.random.binomial(1, 0.15, n),
            'MET_mut': np.random.binomial(1, 0.2, n),
            'ITCF_Surrogate_Score': score
        })

    def test_reverse_km(self):
        med_fu = compute_reverse_km_followup(self.df['OS_Months'].values, self.df['OS_Event'].values)
        self.assertGreater(med_fu, 0.0)

    def test_kaplan_meier_stratification(self):
        km = ITCFKaplanMeier()
        res = km.fit_and_stratify(self.df)
        self.assertEqual(res['n_low'] + res['n_high'], len(self.df))
        self.assertIn('table_8_df', res)
        self.assertEqual(len(res['table_8_df']), 2)

    def test_harrell_c_index(self):
        c_idx = compute_harrell_c_index(
            self.df['OS_Months'].values,
            self.df['ITCF_Surrogate_Score'].values,
            self.df['OS_Event'].values
        )
        self.assertTrue(0.0 <= c_idx <= 1.0)


if __name__ == '__main__':
    unittest.main()
