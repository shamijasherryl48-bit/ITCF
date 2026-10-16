"""
ITCF Visualization Subpackage
==============================
"""

from .gradcam import CellularAttentionVisualizer
from .plots import (
    plot_figure_2_qualitative,
    plot_figure_3_learning_curves,
    plot_figure_4_tmb_distribution,
    plot_figure_5_3d_bubble,
    plot_figure_6_confusion_matrix,
    plot_figure_7_roc_curves,
    plot_figure_8_ablation_bars,
    plot_figure_9_km_with_risk_table
)

__all__ = [
    "CellularAttentionVisualizer",
    "plot_figure_2_qualitative",
    "plot_figure_3_learning_curves",
    "plot_figure_4_tmb_distribution",
    "plot_figure_5_3d_bubble",
    "plot_figure_6_confusion_matrix",
    "plot_figure_7_roc_curves",
    "plot_figure_8_ablation_bars",
    "plot_figure_9_km_with_risk_table",
]
