"""
Cox Proportional Hazards Survival Analysis
===========================================
Implementation of Section 3.4.1 and Section 4.4 (Table 3):
- Univariate Cox Proportional Hazards regression for:
  Age, Sex, Tumor Stage, TMB, VHL, PBRM1, BAP1, SETD2, MET, ITCF Surrogate Score
- Multivariable Cox Proportional Hazards regression
  (TMB strictly excluded from multivariable model to prevent circularity / multicollinearity)
- Hazard Ratio (HR), 95% Confidence Intervals, p-values
- Schoenfeld residuals test for evaluating proportional hazards assumption
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


class ITCFSurvivalAnalysis:
    """
    Survival modeling and Cox Proportional Hazards analysis.
    """
    def __init__(self, duration_col: str = 'OS_Months', event_col: str = 'OS_Event'):
        self.duration_col = duration_col
        self.event_col = event_col
        self.cph_univariate = {}
        self.cph_multivariable = CoxPHFitter()

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes survival dataframe columns and binary encodings."""
        df_clean = df.copy()
        if 'Sex_Male' not in df_clean.columns:
            df_clean['Sex_Male'] = (df_clean['Sex'].astype(str).str.lower() == 'male').astype(int)
        if 'Stage_High' not in df_clean.columns:
            df_clean['Stage_High'] = df_clean['Stage'].astype(str).apply(
                lambda s: 1 if ('iii' in s.lower() or 'iv' in s.lower() or 'late' in s.lower()) else 0
            )
        return df_clean

    def fit_univariate_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits individual univariate Cox models for all clinical, genomic, and surrogate features.
        Matches Table 3 Univariate columns.
        """
        df_clean = self.prepare_dataframe(df)
        
        variables = [
            ('Age', 'Age (continuous)'),
            ('Sex_Male', 'Sex (Male vs Female)'),
            ('Stage_High', 'Tumour Stage (III/IV vs I/II)'),
            ('TMB', 'TMB (continuous)'),
            ('VHL_mut', 'VHL mutation'),
            ('PBRM1_mut', 'PBRM1 mutation'),
            ('BAP1_mut', 'BAP1 mutation'),
            ('SETD2_mut', 'SETD2 mutation'),
            ('MET_mut', 'MET mutation'),
            ('ITCF_Surrogate_Score', 'ITCF Surrogate Score')
        ]
        
        results = []
        for col, display_name in variables:
            if col not in df_clean.columns:
                continue
            sub_df = df_clean[[self.duration_col, self.event_col, col]].dropna()
            cph = CoxPHFitter()
            cph.fit(sub_df, duration_col=self.duration_col, event_col=self.event_col)
            self.cph_univariate[col] = cph
            
            hr = float(np.exp(cph.params_[col]))
            ci_low = float(np.exp(cph.confidence_intervals_.loc[col, f"95% lower-bound"]))
            ci_high = float(np.exp(cph.confidence_intervals_.loc[col, f"95% upper-bound"]))
            p_val = float(cph.summary.loc[col, 'p'])
            
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
            results.append({
                'Variable': display_name,
                'Univariate HR (95% CI)': f"{hr:.3f} ({ci_low:.3f}–{ci_high:.3f})",
                'P-value': p_str,
                'raw_hr': hr,
                'raw_ci_low': ci_low,
                'raw_ci_high': ci_high,
                'raw_p': p_val
            })
            
        return pd.DataFrame(results)

    def fit_multivariable_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CoxPHFitter]:
        """
        Fits multivariable Cox model excluding TMB (Section 4.4, Table 3).
        Covariates: Age, Sex, Stage, VHL, PBRM1, BAP1, SETD2, MET, ITCF Surrogate Score.
        """
        df_clean = self.prepare_dataframe(df)
        
        covariates = [
            'Age', 'Sex_Male', 'Stage_High',
            'VHL_mut', 'PBRM1_mut', 'BAP1_mut', 'SETD2_mut', 'MET_mut',
            'ITCF_Surrogate_Score'
        ]
        
        display_map = {
            'Age': 'Age (continuous)',
            'Sex_Male': 'Sex (Male vs Female)',
            'Stage_High': 'Tumour Stage (III/IV vs I/II)',
            'VHL_mut': 'VHL mutation',
            'PBRM1_mut': 'PBRM1 mutation',
            'BAP1_mut': 'BAP1 mutation',
            'SETD2_mut': 'SETD2 mutation',
            'MET_mut': 'MET mutation',
            'ITCF_Surrogate_Score': 'ITCF Surrogate Score'
        }
        
        cols = [self.duration_col, self.event_col] + covariates
        model_df = df_clean[cols].dropna()
        
        self.cph_multivariable.fit(model_df, duration_col=self.duration_col, event_col=self.event_col)
        
        results = []
        for col in covariates:
            hr = float(np.exp(self.cph_multivariable.params_[col]))
            ci_low = float(np.exp(self.cph_multivariable.confidence_intervals_.loc[col, f"95% lower-bound"]))
            ci_high = float(np.exp(self.cph_multivariable.confidence_intervals_.loc[col, f"95% upper-bound"]))
            p_val = float(self.cph_multivariable.summary.loc[col, 'p'])
            
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
            results.append({
                'Variable': display_map[col],
                'Multivariable HR (95% CI)': f"{hr:.3f} ({ci_low:.3f}–{ci_high:.3f})",
                'P-value': p_str,
                'raw_hr': hr,
                'raw_ci_low': ci_low,
                'raw_ci_high': ci_high,
                'raw_p': p_val
            })
            
        return pd.DataFrame(results), self.cph_multivariable

    def test_proportional_hazards(self) -> pd.DataFrame:
        """
        Evaluates proportional hazards assumption based on Schoenfeld residuals (Section 3.4.1).
        """
        ph_results = proportional_hazard_test(self.cph_multivariable, self.cph_multivariable._preprocess_dataframe())
        return ph_results.summary

    def generate_table_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates Table 3 combining Univariate and Multivariable Cox regression.
        """
        uni_df = self.fit_univariate_models(df)
        multi_df, _ = self.fit_multivariable_model(df)
        
        # Merge on Variable
        merged = []
        ordered_vars = [
            'Age (continuous)',
            'Sex (Male vs Female)',
            'Tumour Stage (III/IV vs I/II)',
            'TMB (continuous)',
            'VHL mutation',
            'PBRM1 mutation',
            'BAP1 mutation',
            'SETD2 mutation',
            'MET mutation',
            'ITCF Surrogate Score'
        ]
        
        uni_dict = {r['Variable']: r for _, r in uni_df.iterrows()}
        multi_dict = {r['Variable']: r for _, r in multi_df.iterrows()}
        
        for v in ordered_vars:
            u = uni_dict.get(v, {})
            m = multi_dict.get(v, {})
            merged.append({
                'Variable': v,
                'Univariate HR (95% CI)': u.get('Univariate HR (95% CI)', 'N/A'),
                'P-value (Uni)': u.get('P-value', 'N/A'),
                'Multivariable HR (95% CI)': 'Excluded' if v == 'TMB (continuous)' else m.get('Multivariable HR (95% CI)', 'N/A'),
                'P-value (Multi)': '—' if v == 'TMB (continuous)' else m.get('P-value', 'N/A')
            })
        return pd.DataFrame(merged)
