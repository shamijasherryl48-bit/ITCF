"""
ITCF Framework Trainer
=======================
Implementation of training workflow (Section 4):
- AdamW optimizer with candidate learning rates: [1e-3, 5e-4, 1e-4, 5e-5]
- Cosine annealing learning-rate scheduler decaying to 1e-6
- 100 epochs training with batch size 16
- 5-fold cross validation on training cohort (714 patients)
- Metric monitoring: Accuracy, Macro F1, Loss
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .losses import ITCFLoss


class ITCFTrainer:
    """
    Trainer for the ITCF model.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 1e-2,
        epochs: int = 100,
        class_weights: Optional[torch.Tensor] = None,
        save_dir: str = 'outputs/checkpoints'
    ):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = model.to(self.device)
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Section 4: AdamW optimizer and cosine annealing scheduler to 1e-6
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=min_lr
        )
        
        # Loss function with class-weighting
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = ITCFLoss(class_weights=class_weights)
        
        # History tracking for Figure 3 training curves
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            genomic = batch['genomic'].to(self.device)
            demo = batch['demographic'].to(self.device)
            tmb = batch['tmb'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(genomic, demo, tmb, images)
            
            loss, _ = self.criterion(outputs, labels, tmb)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * genomic.size(0)
            preds = torch.argmax(outputs['probs'], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_surrogates = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                genomic = batch['genomic'].to(self.device)
                demo = batch['demographic'].to(self.device)
                tmb = batch['tmb'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(genomic, demo, tmb, images)
                loss, _ = self.criterion(outputs, labels, tmb)
                
                total_loss += loss.item() * genomic.size(0)
                probs = outputs['probs']
                preds = torch.argmax(probs, dim=-1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_surrogates.extend(outputs['surrogate_score'].cpu().numpy())
                all_patient_ids.extend(batch['patient_id'])
                
        eval_loss = total_loss / max(1, total)
        eval_acc = correct / max(1, total)
        
        results = {
            'loss': eval_loss,
            'accuracy': eval_acc,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'surrogate_scores': np.array(all_surrogates).flatten(),
            'patient_ids': all_patient_ids
        }
        return eval_loss, eval_acc, results

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, verbose: bool = True):
        best_val_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            if val_loader is not None:
                val_loss, val_acc, _ = self.evaluate(val_loader)
            else:
                val_loss, val_acc = train_loss, train_acc
                
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                
            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"Epoch [{epoch:03d}/{self.epochs:03d}] | "
                      f"LR: {current_lr:.2e} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:6.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:6.2f}%")
                
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_dir, 'training_history_100_epochs.csv'), index=False)
        return self.history
