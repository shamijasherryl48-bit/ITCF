"""
Multimodal Fusion and Classification Layer
==========================================
Implementation of Section 3.4 of the ITCF manuscript:
- Fuses genomic representation O_MCIA and histopathological representation O_SW-CPA (Equation 19)
- Generates 3-class RCC subtype probabilities: ccRCC, pRCC, chRCC
- Computes continuous TMB-based research surrogate score for prognostic risk stratification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    Multimodal Information Block (Section 3.4, Eq. 19).
    Combines MCIA feed-forward genomic representation and
    SW-CPA histopathological representation:
      T_final = softmax( O_MCIA + O_SW_CPA )  or projected fusion.
    """
    def __init__(self, embed_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Projections to align both modalities before element-wise fusion
        self.proj_genomic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_image = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion gating / projection network
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head for 3-class RCC Subtypes (Eq. 19)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Continuous TMB-based research surrogate score prediction head (Section 3.4)
        self.surrogate_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, O_MCIA: torch.Tensor, O_SW_CPA: torch.Tensor):
        """
        Args:
            O_MCIA: Genomic representation (B, embed_dim)
            O_SW_CPA: Histopathological representation (B, embed_dim)
        Returns:
            logits: (B, 3) classification logits for ccRCC, pRCC, chRCC
            probs: (B, 3) softmax probabilities
            surrogate_score: (B, 1) continuous research surrogate score
            fused_features: (B, embed_dim) unified multimodal feature representation
        """
        h_g = self.proj_genomic(O_MCIA)
        h_img = self.proj_image(O_SW_CPA)
        
        # Direct addition representation (Eq. 19: O_MCIA (+) O_SW_CPA)
        h_add = h_g + h_img
        
        # Concatenated fusion representation
        h_concat = torch.cat([h_g, h_img], dim=-1)
        fused = self.fusion_layer(h_concat) + h_add
        
        # Subtype classification logits and softmax probabilities (Eq. 19)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        
        # Exploratory TMB-based research surrogate score
        surrogate_score = self.surrogate_head(fused)
        
        return logits, probs, surrogate_score, fused
