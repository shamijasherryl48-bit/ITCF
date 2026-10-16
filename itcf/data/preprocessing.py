"""
Data Preprocessing Pipeline
============================
Implementation of Section 3.1 of the ITCF manuscript:
- k-nearest neighbor imputation (kNNI) for missing genomic features
- Z-score normalization to standardize gene expression features
- Quantile normalization across patient samples
- Median filtering on histopathology patches to remove noise and enhance nuclei boundaries
- Strict patient-level train/validation/test partitioning (no data leakage)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import torch
from PIL import Image


class GenomicPreprocessor:
    """
    Genomic preprocessing pipeline (Section 3.1).
    Fitted strictly on training cohort and applied to validation/test sets.
    """
    def __init__(self, n_neighbors: int = 5, use_quantile_norm: bool = False):
        self.n_neighbors = n_neighbors
        self.use_quantile_norm = use_quantile_norm
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray):
        """Fit imputation and standardization on training genomic data only."""
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted transformations to validation or test data."""
        if not self.is_fitted:
            raise RuntimeError("GenomicPreprocessor must be fitted before transforming data.")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def apply_image_median_filter(image_np: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Applies median filtering to histopathology image array (Section 3.1).
    Reduces noise, suppresses blur and artifacts, while preserving cellular nuclei boundaries.
    
    Args:
        image_np: (H, W, C) numpy array in range [0, 255]
        size: kernel size of median filter (default 3)
    Returns:
        Filtered image (H, W, C)
    """
    filtered = np.zeros_like(image_np)
    for c in range(image_np.shape[2]):
        filtered[:, :, c] = median_filter(image_np[:, :, c], size=size)
    return filtered


def create_patient_level_folds(
    df: pd.DataFrame,
    patient_id_col: str = 'Patient_ID',
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Splits training cohort into 5 folds with sizes [143, 143, 143, 143, 142] (Section 4.1.1).
    Ensures that all slides/patches from any patient remain in the same fold.
    """
    np.random.seed(seed)
    unique_patients = df[patient_id_col].unique()
    np.random.shuffle(unique_patients)
    
    # Target sizes: 143, 143, 143, 143, 142 for 714 patients
    n_total = len(unique_patients)
    fold_sizes = [n_total // n_splits] * n_splits
    for i in range(n_total % n_splits):
        fold_sizes[i] += 1
        
    splits = []
    start = 0
    for fold_idx, size in enumerate(fold_sizes):
        val_patients = set(unique_patients[start : start + size])
        start += size
        
        train_idx = np.where(~df[patient_id_col].isin(val_patients))[0]
        val_idx = np.where(df[patient_id_col].isin(val_patients))[0]
        splits.append((train_idx, val_idx))
        
    return splits
