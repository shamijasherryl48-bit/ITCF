"""
Integrated Transformer-Contrastive Framework (ITCF)
===================================================
A multimodal deep learning framework for Renal Cell Carcinoma (RCC)
subtype classification and exploratory TMB-derived survival estimation.

Modules:
    - models: MLAT, CCST, Multimodal Fusion, and Baseline variants
    - data: Preprocessing, Cohort loaders, PyTorch Datasets
    - training: Multi-modal contrastive trainer, loss functions, schedulers
    - survival: Cox proportional hazards modeling, Kaplan-Meier, C-index
    - evaluation: Metric computation, McNemar's test, DeLong's test
    - visualization: High-resolution publication plots (Figures 2-9)
"""

__version__ = "1.0.0"
__author__ = "R.M.R. Shamija Sherryl et al."
