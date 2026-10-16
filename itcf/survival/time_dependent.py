"""
Time-Dependent Survival Metrics and External Validation
========================================================
Implementation of Section 4.6 and Table 7:
- Harrell's Concordance Index (C-index) with 1000-iteration bootstrap 95% CI
- Cumulative/Dynamic Time-dependent ROC AUC at 12, 24, 36, and 60 months
- External CPTAC-CCRCC validation (N = 187)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from .kaplan_meier import compute_reverse_km_followup


def compute_harrell_c_index(time: np.ndarray, risk_scores: np.ndarray, event: np.ndarray) -> float:
    """Computes Harrell's concordance index."""
    return float(concordance_index(time, -risk_scores, event))


def bootstrap_c_index_ci(
    time: np.ndarray,
    risk_scores: np.ndarray,
    event: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes Harrell's C-index along with 95% bootstrap confidence interval.
    """
    c_idx = compute_harrell_c_index(time, risk_scores, event)
    rng = np.random.RandomState(seed)
    n = len(time)
    boot_scores = []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        if np.sum(event[idx]) < 2:
            continue
        try:
            val = compute_harrell_c_index(time[idx], risk_scores[idx], event[idx])
            boot_scores.append(val)
        except Exception:
            continue
            
    boot_scores = np.array(boot_scores)
    boot_scores.sort()
    ci_lower = float(boot_scores[int(0.025 * len(boot_scores))])
    ci_upper = float(boot_scores[int(0.975 * len(boot_scores))])
    return c_idx, ci_lower, ci_upper


def compute_time_dependent_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    time_points: List[int] = [12, 24, 36, 60]
) -> Dict[int, float]:
    """
    Computes time-dependent ROC AUC at specified months (12, 24, 36, 60).
    Case: event occurred before or at time t
    Control: patient survived beyond time t
    """
    auc_dict = {}
    for t_eval in time_points:
        cases = (time <= t_eval) & (event == 1)
        controls = (time > t_eval)
        valid = cases | controls
        
        if np.sum(cases) >= 2 and np.sum(controls) >= 2:
            y_true = cases[valid].astype(int)
            auc = roc_auc_score(y_true, risk_scores[valid])
            auc_dict[t_eval] = float(auc)
        else:
            auc_dict[t_eval] = np.nan
    return auc_dict


def evaluate_cptac_cohort(df_cptac: pd.DataFrame) -> Dict:
    """
    Computes all Table 7 external survival validation metrics for CPTAC-CCRCC (N = 187).
    """
    time = df_cptac['OS_Months'].values
    event = df_cptac['OS_Event'].values
    score_col = 'Surrogate_Score' if 'Surrogate_Score' in df_cptac.columns else 'Exact ITCF risk score'
    risk_scores = df_cptac[score_col].values
    
    n_patients = len(df_cptac)
    n_events = int(np.sum(event))
    n_censored = n_patients - n_events
    
    # Reverse KM median follow-up
    med_fu = compute_reverse_km_followup(time, event)
    
    # Median cutoff for risk stratification
    cutoff = float(np.median(risk_scores))
    is_high = risk_scores >= cutoff
    
    df_eval = df_cptac.copy()
    df_eval['High_Risk'] = is_high.astype(int)
    
    high_df = df_eval[is_high]
    low_df = df_eval[~is_high]
    
    # Log-rank test
    lr_res = logrank_test(
        low_df['OS_Months'].values, high_df['OS_Months'].values,
        event_observed_A=low_df['OS_Event'].values,
        event_observed_B=high_df['OS_Event'].values
    )
    
    # Cox Hazard ratio
    cph = CoxPHFitter()
    cph.fit(df_eval[['OS_Months', 'OS_Event', 'High_Risk']],
            duration_col='OS_Months', event_col='OS_Event')
    hr = float(np.exp(cph.params_['High_Risk']))
    ci_low = float(np.exp(cph.confidence_intervals_.loc['High_Risk', '95% lower-bound']))
    ci_high = float(np.exp(cph.confidence_intervals_.loc['High_Risk', '95% upper-bound']))
    
    # C-index and bootstrap CI
    c_idx, c_ci_l, c_ci_h = bootstrap_c_index_ci(time, risk_scores, event, n_bootstraps=1000)
    
    # Time dependent AUC
    auc_dict = compute_time_dependent_auc(time, event, risk_scores, [12, 24, 36, 60])
    
    # Generate Table 7 DataFrame
    table_7_df = pd.DataFrame({
        'Measure': [
            'Eligible patients (n)',
            'Overall-survival events (deaths, n)',
            'Censored patients (n)',
            'Median follow-up (months)',
            'High-risk group (patients / deaths)',
            'Low-risk group (patients / deaths)',
            'Hazard ratio (high vs. low risk)',
            '95% confidence interval for HR',
            'Log-rank P-value',
            "Harrell's C-index (95% CI)",
            'Time-dependent AUC at 12 months',
            'Time-dependent AUC at 24 months',
            'Time-dependent AUC at 36 months',
            'Time-dependent AUC at 60 months'
        ],
        'Result': [
            f"{n_patients}",
            f"{n_events} ({n_events/n_patients*100:.1f}%)",
            f"{n_censored} ({n_censored/n_patients*100:.1f}%)",
            f"{med_fu:.1f}",
            f"{len(high_df)} / {int(np.sum(high_df['OS_Event']))}",
            f"{len(low_df)} / {int(np.sum(low_df['OS_Event']))}",
            f"{hr:.2f}",
            f"{ci_low:.3f}–{ci_high:.3f}",
            f"{lr_res.p_value:.3f}",
            f"{c_idx:.3f} ({c_ci_l:.3f}–{c_ci_h:.3f})",
            f"{auc_dict[12]:.3f}",
            f"{auc_dict[24]:.3f}",
            f"{auc_dict[36]:.3f}",
            f"{auc_dict[60]:.3f}"
        ]
    })
    
    return {
        'table_7_df': table_7_df,
        'hr': hr,
        'ci': (ci_low, ci_high),
        'p_value': lr_res.p_value,
        'c_index': c_idx,
        'c_index_ci': (c_ci_l, c_ci_h),
        'auc_dict': auc_dict,
        'median_followup': med_fu
    }
