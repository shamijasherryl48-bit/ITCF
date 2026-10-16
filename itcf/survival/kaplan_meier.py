"""
Kaplan-Meier Survival Analysis and Risk Table
==============================================
Implementation of Section 3.4.1, 4.7, Figure 9, and Table 8:
- Stratification into Low-Risk and High-Risk groups based on the median ITCF surrogate score
- Reverse Kaplan-Meier method for median follow-up duration (48.6 months)
- Log-rank hypothesis test (P < 0.001)
- Unadjusted Cox Hazard Ratio: HR = 2.47 (95% CI: 1.89 - 3.23)
- Computation of Number of Patients at Risk across intervals:
  [0, 20, 40, 60, 80, 100, 120] months (Table 8)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def compute_reverse_km_followup(time: np.ndarray, event: np.ndarray) -> float:
    """
    Computes median follow-up time using the reverse Kaplan-Meier method (Section 3.4.1).
    In reverse KM, the event indicator is inverted (censoring = event).
    """
    kmf = KaplanMeierFitter()
    rev_event = 1 - event
    kmf.fit(time, event_observed=rev_event)
    median_fu = kmf.median_survival_time_
    if np.isnan(median_fu) or np.isinf(median_fu):
        # Fallback to empirical median of censored times
        censored_times = time[event == 0]
        return float(np.median(censored_times)) if len(censored_times) > 0 else float(np.median(time))
    return float(median_fu)


def compute_patients_at_risk(
    times: np.ndarray,
    time_points: List[int] = [0, 20, 40, 60, 80, 100, 120]
) -> List[int]:
    """
    Computes number of patients remaining at risk at specified time milestones.
    """
    return [int(np.sum(times >= tp)) for tp in time_points]


class ITCFKaplanMeier:
    """
    Performs complete Kaplan-Meier stratification, log-rank testing,
    and Table 8 risk table extraction.
    """
    def __init__(
        self,
        duration_col: str = 'OS_Months',
        event_col: str = 'OS_Event',
        score_col: str = 'ITCF_Surrogate_Score'
    ):
        self.duration_col = duration_col
        self.event_col = event_col
        self.score_col = score_col
        self.time_points = [0, 20, 40, 60, 80, 100, 120]

    def fit_and_stratify(
        self,
        df: pd.DataFrame,
        cutoff_value: float = None
    ) -> Dict:
        """
        Stratifies cohort into low-risk and high-risk groups.
        If cutoff_value is None, uses median score.
        """
        scores = df[self.score_col].values
        if cutoff_value is None:
            cutoff_value = float(np.median(scores))
            
        is_high = scores >= cutoff_value
        df_strat = df.copy()
        df_strat['Risk_Group'] = np.where(is_high, 'High-risk', 'Low-risk')
        
        low_df = df_strat[~is_high]
        high_df = df_strat[is_high]
        
        t_low = low_df[self.duration_col].values
        e_low = low_df[self.event_col].values
        t_high = high_df[self.duration_col].values
        e_high = high_df[self.event_col].values
        
        # Log-rank test
        lr_res = logrank_test(t_low, t_high, event_observed_A=e_low, event_observed_B=e_high)
        
        # Cox PH for unadjusted Hazard Ratio (Figure 9)
        cph = CoxPHFitter()
        df_strat['High_Risk_Binary'] = is_high.astype(int)
        cph.fit(df_strat[[self.duration_col, self.event_col, 'High_Risk_Binary']],
                duration_col=self.duration_col, event_col=self.event_col)
        
        hr = float(np.exp(cph.params_['High_Risk_Binary']))
        ci_low = float(np.exp(cph.confidence_intervals_.loc['High_Risk_Binary', '95% lower-bound']))
        ci_high = float(np.exp(cph.confidence_intervals_.loc['High_Risk_Binary', '95% upper-bound']))
        
        # Reverse KM median follow-up
        med_fu = compute_reverse_km_followup(df[self.duration_col].values, df[self.event_col].values)
        
        # Table 8: Number at risk
        at_risk_low = compute_patients_at_risk(t_low, self.time_points)
        at_risk_high = compute_patients_at_risk(t_high, self.time_points)
        
        # Table 8 DataFrame
        table_8_df = pd.DataFrame({
            'Risk stratification group': [f'Low-risk (n={len(low_df)})', f'High-risk (n={len(high_df)})'],
            '0 months': [at_risk_low[0], at_risk_high[0]],
            '20 months': [at_risk_low[1], at_risk_high[1]],
            '40 months': [at_risk_low[2], at_risk_high[2]],
            '60 months': [at_risk_low[3], at_risk_high[3]],
            '80 months': [at_risk_low[4], at_risk_high[4]],
            '100 months': [at_risk_low[5], at_risk_high[5]],
            '120 months': [at_risk_low[6], at_risk_high[6]],
        })
        
        return {
            'cutoff_value': cutoff_value,
            'n_low': len(low_df),
            'n_high': len(high_df),
            'deaths_low': int(np.sum(e_low)),
            'deaths_high': int(np.sum(e_high)),
            'hr': hr,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'logrank_p': lr_res.p_value,
            'median_followup_months': med_fu,
            'at_risk_low': at_risk_low,
            'at_risk_high': at_risk_high,
            'table_8_df': table_8_df,
            'stratified_df': df_strat
        }
