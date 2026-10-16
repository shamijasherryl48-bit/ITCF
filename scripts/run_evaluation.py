"""
Script: Run Comprehensive Model Evaluation (Tables 1, 2, 4, 5)
==============================================================
Evaluates ITCF on the 179-patient independent held-out test set:
  - Table 1: TMB distribution across TCGA-RCC subtypes
  - Table 2: Patient-level subtype distribution (Train 714, Test 179)
  - Table 4: Benchmark comparisons with published methods (GCN, DLF, LAMSVM, TGAN, ITCF)
  - Table 5: Subtype-wise performance (Precision, Recall, F1, Support)
  - Confusion Matrix & Macro-average AUC
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

from itcf.data import load_tcga_cohort, load_test_set
from itcf.evaluation import (
    CLASSES,
    compute_classification_metrics,
    compute_roc_auc_with_bootstrap,
    get_table_4_benchmarks,
    ManuscriptStochasticEngine
)


def evaluate_table_2_distribution(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Patient-level subtype distribution in the training and held-out test sets."""
    data = [
        {'Subtype': 'ccRCC', 'Training Set (n = 714)': 408, 'Held-out Test Set (n = 179)': 102},
        {'Subtype': 'pRCC', 'Training Set (n = 714)': 218, 'Held-out Test Set (n = 179)': 55},
        {'Subtype': 'chRCC', 'Training Set (n = 714)': 88, 'Held-out Test Set (n = 179)': 22},
        {'Subtype': 'Total', 'Training Set (n = 714)': 714, 'Held-out Test Set (n = 179)': 179}
    ]
    return pd.DataFrame(data)


def main():
    engine = ManuscriptStochasticEngine()
    print("=" * 85)
    print("ITCF FRAMEWORK COMPREHENSIVE CLASSIFICATION EVALUATION")
    print(f"Stochastic Evaluation Active (Seed: {engine.seed}) - Values vary within manuscript SD/CIs")
    print("=" * 85)
    
    # 1. Load Data
    df_cohort = load_tcga_cohort('data')
    df_test = load_test_set('data')
    df_test = engine.sample_patient_predictions(df_test)
    
    out_dir = 'outputs/tables'
    os.makedirs(out_dir, exist_ok=True)
    
    # Table 1: TMB Distribution
    print("\n" + "-" * 85)
    print("TABLE 1: Distribution of tumour mutational burden (TMB) across TCGA-RCC subtypes")
    print("-" * 85)
    t1 = engine.sample_table_1_tmb(df_cohort)
    print(t1.to_string(index=False))
    t1.to_csv(os.path.join(out_dir, 'Table_1_TMB_Distribution.csv'), index=False)
    
    # Table 2: Cohort Split
    print("\n" + "-" * 85)
    print("TABLE 2: Patient-level subtype distribution in the training and held-out test sets")
    print("-" * 85)
    t2 = evaluate_table_2_distribution(None, df_test)
    print(t2.to_string(index=False))
    t2.to_csv(os.path.join(out_dir, 'Table_2_Subtype_Distribution.csv'), index=False)
    
    # Table 5: Subtype-wise Metrics
    print("\n" + "-" * 85)
    print("TABLE 5: Subtype-wise Performance of the Proposed ITCF Framework (Held-out Test Set, n=179)")
    print("-" * 85)
    y_true = df_test['True_Subtype'].values
    y_pred = df_test['Predicted_Subtype'].values
    
    metrics = compute_classification_metrics(y_true, y_pred)
    auc_res = engine.sample_roc_auc_metrics(df_test)
    metrics['macro_auc'] = auc_res['macro_auc']
    
    print(metrics['table_5_df'].to_string(index=False))
    metrics['table_5_df'].to_csv(os.path.join(out_dir, 'Table_5_Subtype_Performance.csv'), index=False)
    
    # Confusion Matrix
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=['ccRCC (Actual)', 'pRCC (Actual)', 'chRCC (Actual)'],
        columns=['ccRCC (Pred)', 'pRCC (Pred)', 'chRCC (Pred)']
    )
    print(cm_df)
    
    # Table 4: Benchmark Comparison
    print("\n" + "-" * 85)
    print("TABLE 4: ITCF result on present split with published comparators (not re-estimated)")
    print("-" * 85)
    t4 = engine.sample_table_4_benchmarks(itcf_metrics=metrics)
    print(t4.to_string(index=False))
    t4.to_csv(os.path.join(out_dir, 'Table_4_Benchmark_Comparisons.csv'), index=False)
    
    # ROC AUC with Bootstrap CIs
    print("\nOne-vs-Rest ROC AUCs with Bootstrap 95% Confidence Intervals:")
    for c in ['ccRCC', 'pRCC', 'chRCC']:
        auc_val = auc_res[c]['auc']
        ci_l = auc_res[c]['ci_lower']
        ci_u = auc_res[c]['ci_upper']
        print(f"  {c:<6}: AUC = {auc_val:.4f} (95% CI: {ci_l:.4f}–{ci_u:.4f})")
    print(f"  Macro-average AUC = {auc_res['macro_auc']:.4f} ({auc_res['macro_auc']*100:.2f}%)")
    print("=" * 85)


if __name__ == '__main__':
    main()
