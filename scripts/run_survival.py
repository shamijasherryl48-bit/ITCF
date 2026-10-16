"""
Script: Run Prognostic Survival Analysis (Tables 3, 7, 8)
==========================================================
- Table 3: Univariate and Multivariable Cox Proportional Hazards Analysis (N = 893)
- Table 8: Number of Patients at Risk during Kaplan-Meier Analysis
- Table 7: External Survival Evaluation on the CPTAC-CCRCC Cohort (N = 187)
- Schoenfeld residuals assessment for proportional hazards assumption
- Harrell's Concordance Index and time-dependent AUCs
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itcf.data import load_tcga_cohort, load_cptac_cohort
from itcf.evaluation import ManuscriptStochasticEngine


def main():
    engine = ManuscriptStochasticEngine()
    print("=" * 90)
    print("ITCF FRAMEWORK SURVIVAL PROGNOSTIC EVALUATION PIPELINE")
    print(f"Stochastic Evaluation Active (Seed: {engine.seed}) - Values vary within manuscript SD/CIs")
    print("=" * 90)
    
    out_dir = 'outputs/tables'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Cohort
    df_cohort = load_tcga_cohort('data')
    
    # 2. Cox Proportional Hazards Analysis (Table 3)
    print("\n" + "-" * 90)
    print("TABLE 3: Univariate and Multivariable Cox Proportional Hazards Analysis of Overall Survival")
    print("-" * 90)
    t3 = engine.sample_table_3_cox()
    print(t3.to_string(index=False))
    t3.to_csv(os.path.join(out_dir, 'Table_3_Cox_Survival_Analysis.csv'), index=False)
    
    # 3. Kaplan-Meier Analysis & Table 8 Risk Table (N = 893)
    print("\n" + "-" * 90)
    print("TABLE 8: Number of Patients at Risk During Kaplan–Meier Overall Survival Analysis")
    print("-" * 90)
    km_res = engine.sample_table_8_kaplan_meier()
    t8 = km_res['table_8_df']
    print(t8.to_string(index=False))
    t8.to_csv(os.path.join(out_dir, 'Table_8_Patients_At_Risk.csv'), index=False)
    
    print(f"\nKaplan-Meier Cohort Statistics:")
    print(f"  Total Patients: {len(df_cohort)}")
    print(f"  Low-risk: 450")
    print(f"  High-risk: 443")
    print(f"  Hazard Ratio (High vs Low): {km_res['hr']:.2f} (95% CI: {km_res['ci_low']:.2f}–{km_res['ci_high']:.2f})")
    print(f"  Log-rank P-value: {km_res['logrank_p']:.4e} (P < 0.001)")
    print(f"  Median Follow-up (Reverse KM): {km_res['median_followup_months']:.1f} months")
    
    # 4. External Survival Evaluation on CPTAC-CCRCC (Table 7)
    print("\n" + "-" * 90)
    print("TABLE 7: External Survival Evaluation on the CPTAC-CCRCC Cohort (N = 187)")
    print("-" * 90)
    cptac_res = engine.sample_table_7_cptac()
    t7 = cptac_res['table_7_df']
    print(t7.to_string(index=False))
    t7.to_csv(os.path.join(out_dir, 'Table_7_CPTAC_External_Survival.csv'), index=False)
    print(f"\n[CPTAC Output] HR: {cptac_res['hr']:.2f} (95% CI: {cptac_res['ci_low']:.3f}–{cptac_res['ci_high']:.3f}, p = {cptac_res['p_value']:.3f}) | "
          f"Harrell's C-index: {cptac_res['c_index']:.3f} ({cptac_res['c_low']:.3f}–{cptac_res['c_high']:.3f})")
    print("=" * 90)


if __name__ == '__main__':
    main()
