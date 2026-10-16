"""
Integrated Transformer-Contrastive Framework (ITCF)
===================================================
Master PyTorch model implementing Figure 1 and Equations 1-19:
- MLAT Module (Genomic pathway-aware transformer)
- CCST Module (Cellular characteristics-aware Swin transformer)
- Multimodal Information Block & Fusion Layer
- Multi-loss objectives (Classification + Multimodal Contrastive Alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlat import MolecularLandscapeAwareTransformer
from .ccst import CellularCharacteristicsAwareSwinTransformer
from .fusion import MultimodalFusion


class ITCFModel(nn.Module):
    """
    Proposed Integrated Transformer-Contrastive Framework (ITCF) for RCC Classification
    and Survival Probability Estimation based on Multi-Modal Data Integration.
    """
    def __init__(
        self,
        num_genes: int = 5,
        num_demographics: int = 3,
        num_classes: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.2,
        contrastive_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        
        # 1. Molecular Landscape-Aware Transformer (MLAT)
        self.mlat = MolecularLandscapeAwareTransformer(
            num_genes=num_genes,
            num_demographics=num_demographics,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            contrastive_dim=contrastive_dim
        )
        
        # 2. Cellular Characteristics-Aware Swin Transformer (CCST)
        self.ccst = CellularCharacteristicsAwareSwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            contrastive_dim=contrastive_dim
        )
        
        # 3. Multimodal Information Block & Classifier
        self.fusion = MultimodalFusion(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Cross-modal contrastive projection head for aligning molecular & cellular spaces
        self.cross_modal_proj_g = nn.Linear(contrastive_dim, contrastive_dim)
        self.cross_modal_proj_img = nn.Linear(contrastive_dim, contrastive_dim)

    def forward(
        self,
        genomic_features: torch.Tensor,
        demo_features: torch.Tensor,
        tmb: torch.Tensor,
        images: torch.Tensor
    ):
        """
        Forward pass of the complete ITCF framework.
        
        Args:
            genomic_features: (B, num_genes)
            demo_features: (B, num_demographics)
            tmb: (B, 1) or (B,)
            images: (B, 3, 224, 224)
            
        Returns:
            dict containing:
              - 'logits': (B, 3) classification logits
              - 'probs': (B, 3) predicted class probabilities
              - 'surrogate_score': (B, 1) continuous research surrogate score
              - 'fused_features': (B, embed_dim) fused representation
              - 'z_genomic': (B, contrastive_dim) normalized genomic projection
              - 'z_image': (B, contrastive_dim) normalized cellular image projection
              - 'O_MCIA': (B, embed_dim) genomic representation
              - 'O_SW_CPA': (B, embed_dim) histopathological representation
        """
        # Extract genomic representation and contrastive vector via MLAT
        O_MCIA, z_genomic = self.mlat(genomic_features, demo_features, tmb)
        
        # Extract histopathological representation and contrastive vector via CCST
        O_SW_CPA, z_image = self.ccst(images)
        
        # Multimodal fusion and subtype classification (Eq. 19)
        logits, probs, surrogate_score, fused = self.fusion(O_MCIA, O_SW_CPA)
        
        # Cross-modal contrastive representations
        z_g_aligned = F.normalize(self.cross_modal_proj_g(z_genomic), p=2, dim=-1)
        z_img_aligned = F.normalize(self.cross_modal_proj_img(z_image), p=2, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'surrogate_score': surrogate_score,
            'fused_features': fused,
            'z_genomic': z_g_aligned,
            'z_image': z_img_aligned,
            'O_MCIA': O_MCIA,
            'O_SW_CPA': O_SW_CPA
        }

    def compute_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE contrastive loss (Section 3.2.3, Eq. 10).
        """
        B = z1.shape[0]
        # Cosine similarity matrix scaled by temperature tau = 0.07
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(B, device=z1.device)
        
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.T, labels)
        return (loss_1 + loss_2) / 2.0
