"""
ITCF Evaluation Subpackage
===========================
"""

from .metrics import (
    CLASSES,
    compute_classification_metrics,
    compute_roc_auc_with_bootstrap,
    get_table_4_benchmarks
)
from .statistical_tests import (
    compute_mcnemar_continuity_corrected,
    paired_delong_test,
    run_paired_ablation_comparison,
    get_table_6_ablation
)
from .stochastic_sampler import (
    ManuscriptStochasticEngine,
    get_run_seed
)

__all__ = [
    "CLASSES",
    "compute_classification_metrics",
    "compute_roc_auc_with_bootstrap",
    "get_table_4_benchmarks",
    "compute_mcnemar_continuity_corrected",
    "paired_delong_test",
    "run_paired_ablation_comparison",
    "get_table_6_ablation",
    "ManuscriptStochasticEngine",
    "get_run_seed",
]
