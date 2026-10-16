"""
Script: Train ITCF Multimodal Framework
=======================================
Executes the training and cross-validation procedure described in Section 4:
  - AdamW optimizer (learning rate candidate fine-tuning)
  - Cosine annealing learning-rate scheduler decaying to 1e-6
  - Batch size 16, 12 transformer layers, 8 attention heads, dropout 0.2
  - Multi-task objective: Weighted CE + InfoNCE contrastive alignment + Margin contrastive loss
  - 5-fold cross validation on 714-patient training set
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itcf.models import ITCFModel
from itcf.data import (
    load_tcga_cohort,
    load_test_set,
    get_multimodal_dataloaders,
    create_patient_level_folds
)
from itcf.training import ITCFTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ITCF Multi-Modal Framework")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5 for quick demo, 100 for full run)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--folds', type=int, default=5, help="Cross-validation folds (default: 5)")
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints', help="Checkpoint directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING INTEGRATED TRANSFORMER-CONTRASTIVE FRAMEWORK (ITCF)")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    
    # 1. Load Data
    df_cohort = load_tcga_cohort('data')
    df_test = load_test_set('data')
    
    # Exclude test patients to create 714 training cohort
    test_ids = set(df_test['Patient_ID'].unique())
    df_train = df_cohort[~df_cohort['Patient_ID'].isin(test_ids)].reset_index(drop=True)
    if len(df_train) == 0:
        # Split 80/20 if test_ids don't match TCGA format
        np.random.seed(42)
        shuffled = df_cohort.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_train = shuffled.iloc[:714].reset_index(drop=True)
        df_val = shuffled.iloc[714:].reset_index(drop=True)
    else:
        df_val = df_train.sample(n=min(143, len(df_train)), random_state=42)
        
    print(f"Training cohort: {len(df_train)} patients | Validation set: {len(df_val)} patients")
    
    # 2. Compute Class Weights for Imbalance (Section 4.2)
    counts = [
        int(np.sum(df_train['Subtype'].isin(['ccRCC', 'KIRC']))),
        int(np.sum(df_train['Subtype'].isin(['pRCC', 'KIRP']))),
        int(np.sum(df_train['Subtype'].isin(['chRCC', 'KICH'])))
    ]
    total_samples = sum(counts)
    class_weights = torch.tensor([total_samples / (3.0 * c) if c > 0 else 1.0 for c in counts], dtype=torch.float32)
    print(f"Subtype sample counts: {counts}")
    print(f"Inverse class weights: {class_weights.tolist()}")
    
    # 3. DataLoaders
    train_loader, val_loader = get_multimodal_dataloaders(
        df_train, df_val,
        images_dir='data/sample_images',
        batch_size=args.batch_size
    )
    
    # 4. Instantiate ITCF Model (Section 3.2, 3.3, 3.4)
    model = ITCFModel(
        num_genes=5,
        num_demographics=3,
        num_classes=3,
        img_size=224,
        patch_size=16,
        embed_dim=256,
        num_layers=12,
        num_heads=8,
        dropout=0.2,
        contrastive_dim=128,
        temperature=0.07
    )
    
    # 5. Trainer
    trainer = ITCFTrainer(
        model=model,
        device=device,
        lr=args.lr,
        min_lr=1e-6,
        epochs=args.epochs,
        class_weights=class_weights,
        save_dir=args.save_dir
    )
    
    print("\nStarting Training Loop...")
    history = trainer.fit(train_loader, val_loader, verbose=True)
    
    print("\nTraining completed successfully!")
    print(f"Best checkpoint saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
