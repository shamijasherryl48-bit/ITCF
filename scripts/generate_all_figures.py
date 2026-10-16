"""
Script: Generate All Manuscript Figures (Figures 2 to 9)
=========================================================
Produces publication-quality 300-DPI figures matching the ITCF paper:
  - Figure 2: Qualitative Histopathology Interpretability & Attention Maps
  - Figure 3: Convergence Curves (Accuracy & Loss over 100 Epochs)
  - Figure 4: TMB Distribution Across TCGA-RCC Subtypes
  - Figure 5: 3D Bubble Chart of Hazard Ratios & P-values
  - Figure 6: Confusion Matrix on Held-Out Test Set (N = 179)
  - Figure 7: One-vs-Rest ROC Curves with Bootstrap 95% CIs
  - Figure 8: Ablation Study Progressive Improvement in AUC
  - Figure 9: Kaplan-Meier Overall Survival Curves with Integrated Risk Table
"""

import os
import sys
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itcf.data import load_tcga_cohort, load_test_set
from itcf.visualization import (
    plot_figure_2_qualitative,
    plot_figure_3_learning_curves,
    plot_figure_4_tmb_distribution,
    plot_figure_5_3d_bubble,
    plot_figure_6_confusion_matrix,
    plot_figure_7_roc_curves,
    plot_figure_8_ablation_bars,
    plot_figure_9_km_with_risk_table
)


def main():
    print("=" * 80)
    print("GENERATING ALL PUBLICATION FIGURES (FIGURES 2 - 9)")
    print("=" * 80)
    
    fig_dir = 'outputs/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    df_cohort = load_tcga_cohort('data')
    df_test = load_test_set('data')
    
    # Figure 2: Qualitative Analysis & Grad-CAM Heatmaps
    plot_figure_2_qualitative(
        output_path=os.path.join(fig_dir, 'Figure_2.png'),
        sample_images_dir='data/sample_images'
    )
    
    # Figure 3: Training & Validation Curves over 100 epochs
    plot_figure_3_learning_curves(
        output_path=os.path.join(fig_dir, 'Figure_3.png')
    )
    
    # Figure 4: TMB Distribution Violin & Box Plots
    plot_figure_4_tmb_distribution(
        df_cohort=df_cohort,
        output_path=os.path.join(fig_dir, 'Figure_4.png')
    )
    
    # Figure 5: 3D Bubble Chart
    plot_figure_5_3d_bubble(
        output_path=os.path.join(fig_dir, 'Figure_5.png')
    )
    
    # Figure 6: Confusion Matrix (n = 179)
    plot_figure_6_confusion_matrix(
        df_test=df_test,
        output_path=os.path.join(fig_dir, 'Figure_6.png')
    )
    
    # Figure 7: One-vs-Rest ROC Curves
    plot_figure_7_roc_curves(
        df_test=df_test,
        output_path=os.path.join(fig_dir, 'Figure_7.png')
    )
    
    # Figure 8: Ablation Study Bar Chart
    plot_figure_8_ablation_bars(
        output_path=os.path.join(fig_dir, 'Figure_8.png')
    )
    
    # Figure 9: Kaplan-Meier Survival Curves & Risk Table
    plot_figure_9_km_with_risk_table(
        output_path=os.path.join(fig_dir, 'Figure_9.png')
    )
    
    print("\nAll figures generated successfully in outputs/figures/")
    print("=" * 80)


if __name__ == '__main__':
    main()
