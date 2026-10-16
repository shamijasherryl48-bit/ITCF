"""
Loss Functions for ITCF Multimodal Training
===========================================
Implementation of:
- InfoNCE / NT-Xent Multimodal Contrastive Loss (Equation 10, tau = 0.07)
- Margin-Based Patch Contrastive Loss (Equation 18)
- Class-Weighted Cross-Entropy Loss for RCC Subtype Imbalance (Section 4.2)
- Combined Multitask Objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Contrastive Loss (Section 3.2.3, Eq. 10).
    Aligns molecular and cellular representations with temperature tau = 0.07.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Normalized features from modality 1 (B, D)
            z2: Normalized features from modality 2 (B, D)
        """
        B = z1.shape[0]
        # Cosine similarity matrix scaled by temperature tau
        logits = torch.matmul(z1, z2.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=z1.device)
        
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.T, labels)
        return (loss_1 + loss_2) / 2.0


class MarginContrastiveLoss(nn.Module):
    """
    Margin-based Contrastive Loss (Section 3.3.3, Eq. 18).
    L_contrast = (1 / 2N) * sum [ y * d^2 + (1 - y) * max(0, u - d)^2 ]
    where:
      - y = 1 for same subtype pair, 0 for different subtype pair
      - d = Euclidean distance
      - u = margin threshold (default 1.0)
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise margin contrastive loss across a batch.
        Args:
            features: (B, D) feature embeddings
            labels: (B,) class labels
        """
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        # Compute pairwise Euclidean distance matrix
        dist_sq = torch.cdist(features, features, p=2) ** 2
        dist = torch.sqrt(dist_sq + 1e-8)
        
        # Ground truth similarity matrix: y_ij = 1 if same class, 0 otherwise
        same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Upper triangular mask excluding diagonal
        mask = torch.triu(torch.ones(B, B, device=features.device), diagonal=1).bool()
        
        y = same_class[mask]
        d = dist[mask]
        
        loss_similar = y * (d ** 2)
        loss_dissimilar = (1.0 - y) * (F.relu(self.margin - d) ** 2)
        
        total_pairs = y.numel()
        if total_pairs == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        return torch.sum(loss_similar + loss_dissimilar) / (2.0 * total_pairs)


class ITCFLoss(nn.Module):
    """
    Unified Multi-Task Loss for ITCF:
      L_total = L_classification (Class-weighted CE)
              + lambda_cont * L_InfoNCE (Genomic-Image alignment)
              + lambda_margin * L_margin (Cellular patch contrastive)
              + lambda_surr * L_MSE (Surrogate score regression)
    """
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        temperature: float = 0.07,
        margin: float = 1.0,
        lambda_cont: float = 0.25,
        lambda_margin: float = 0.15,
        lambda_surr: float = 0.10
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.margin_loss = MarginContrastiveLoss(margin=margin)
        self.mse_loss = nn.MSELoss()
        
        self.lambda_cont = lambda_cont
        self.lambda_margin = lambda_margin
        self.lambda_surr = lambda_surr
        
    def forward(
        self,
        model_outputs: dict,
        labels: torch.Tensor,
        tmb: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        logits = model_outputs['logits']
        z_g = model_outputs['z_genomic']
        z_img = model_outputs['z_image']
        surrogate = model_outputs['surrogate_score']
        
        # 1. Classification Loss (Weighted Cross-Entropy)
        l_ce = self.ce_loss(logits, labels)
        
        # 2. Multimodal Contrastive Alignment Loss (InfoNCE, Eq. 10)
        l_infonce = self.infonce_loss(z_g, z_img)
        
        # 3. Histopathological Margin Contrastive Loss (Eq. 18)
        l_margin = self.margin_loss(z_img, labels)
        
        # 4. TMB Surrogate Regression Loss
        if tmb.dim() == 1:
            tmb = tmb.unsqueeze(-1)
        l_surr = self.mse_loss(surrogate, tmb)
        
        # Total combined loss
        total_loss = (
            l_ce
            + self.lambda_cont * l_infonce
            + self.lambda_margin * l_margin
            + self.lambda_surr * l_surr
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': l_ce.item(),
            'infonce_loss': l_infonce.item(),
            'margin_loss': l_margin.item(),
            'surrogate_loss': l_surr.item()
        }
        return total_loss, loss_dict
