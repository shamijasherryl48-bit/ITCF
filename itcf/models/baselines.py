"""
Ablation and Baseline Model Architectures (Table 6)
====================================================
Implementations of:
  1. Genomics-only Model (MLAT alone)
  2. Histopathology-only Model (CCST alone)
  3. Basic Transformer (Standard Multi-Head Attention without pathway & peripheral weighting)
  4. MLAT + CCST without Contrastive Learning (Model A in paired tests)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlat import MolecularLandscapeAwareTransformer
from .ccst import CellularCharacteristicsAwareSwinTransformer
from .fusion import MultimodalFusion


class GenomicsOnlyMLAT(nn.Module):
    """
    Genomics-only Baseline (Table 6, Row 1).
    Utilizes only the MLAT module for RCC subtype classification.
    """
    def __init__(
        self,
        num_genes: int = 5,
        num_demographics: int = 3,
        num_classes: int = 3,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.mlat = MolecularLandscapeAwareTransformer(
            num_genes=num_genes,
            num_demographics=num_demographics,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, genomic_features: torch.Tensor, demo_features: torch.Tensor, tmb: torch.Tensor):
        O_MCIA, _ = self.mlat(genomic_features, demo_features, tmb)
        logits = self.classifier(O_MCIA)
        probs = F.softmax(logits, dim=-1)
        return logits, probs, O_MCIA


class HistopathologyOnlyCCST(nn.Module):
    """
    Histopathology-only Baseline (Table 6, Row 2).
    Utilizes only the CCST module for RCC subtype classification.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 3,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.ccst = CellularCharacteristicsAwareSwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, images: torch.Tensor):
        O_SW_CPA, _ = self.ccst(images)
        logits = self.classifier(O_SW_CPA)
        probs = F.softmax(logits, dim=-1)
        return logits, probs, O_SW_CPA


class BasicTransformer(nn.Module):
    """
    Basic Transformer Baseline (Table 6, Row 3).
    A standard vanilla multi-head self-attention transformer without
    biological pathway-aware encoding and without cellular peripheral window weighting.
    """
    def __init__(
        self,
        num_genes: int = 5,
        num_demographics: int = 3,
        num_classes: int = 3,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Standard linear projections without pathway priors
        self.gene_proj = nn.Linear(num_genes, embed_dim)
        self.demo_proj = nn.Linear(num_demographics + 1, embed_dim)  # +1 for TMB
        
        # Standard Vision Transformer patch embedding for 224x224
        self.img_conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(
        self,
        genomic_features: torch.Tensor,
        demo_features: torch.Tensor,
        tmb: torch.Tensor,
        images: torch.Tensor
    ):
        B = genomic_features.shape[0]
        if tmb.dim() == 1:
            tmb = tmb.unsqueeze(-1)
            
        clinical_all = torch.cat([demo_features, tmb], dim=-1)
        token_g = self.gene_proj(genomic_features).unsqueeze(1)
        token_c = self.demo_proj(clinical_all).unsqueeze(1)
        
        img_patches = self.img_conv(images).flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        
        tokens = torch.cat([token_g, token_c, img_patches], dim=1)  # (B, 198, embed_dim)
        out = self.transformer(tokens)
        
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        return logits, probs, pooled


class MLATCCSTWithoutContrastive(nn.Module):
    """
    MLAT + CCST without Contrastive Learning (Table 6, Row 4 / Model A in paired tests).
    Identical dual-branch architecture but evaluated without contrastive feature alignment.
    """
    def __init__(
        self,
        num_genes: int = 5,
        num_demographics: int = 3,
        num_classes: int = 3,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.mlat = MolecularLandscapeAwareTransformer(
            num_genes=num_genes,
            num_demographics=num_demographics,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ccst = CellularCharacteristicsAwareSwinTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fusion = MultimodalFusion(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(
        self,
        genomic_features: torch.Tensor,
        demo_features: torch.Tensor,
        tmb: torch.Tensor,
        images: torch.Tensor
    ):
        O_MCIA, _ = self.mlat(genomic_features, demo_features, tmb)
        O_SW_CPA, _ = self.ccst(images)
        logits, probs, surrogate_score, fused = self.fusion(O_MCIA, O_SW_CPA)
        return {
            'logits': logits,
            'probs': probs,
            'surrogate_score': surrogate_score,
            'fused_features': fused,
            'O_MCIA': O_MCIA,
            'O_SW_CPA': O_SW_CPA
        }
