"""
Molecular Landscape-Aware Transformer (MLAT)
=============================================
Implementation of Section 3.2 of the ITCF manuscript:
- Pathway-aware encoding for genomic alterations and driver mutations (VHL, MET, PBRM1, SETD2, BAP1)
- Demographic embedding for age, sex, and tumor stage (Equations 2, 4)
- Tumor mutational burden (TMB) continuous embedding (Equation 5)
- Biological pathway encoding addition (Equations 6, 7)
- Molecular-Contextual Interaction Attention (MCIA) mechanism (Equation 8)
- Two-layer Feed-Forward Network with ReLU activation (Equation 9)
- Molecular InfoNCE / NT-Xent Contrastive representation projection (Equation 10)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiologicalPathwayEncoding(nn.Module):
    """
    Biological Pathway-Aware Encoding (Section 3.2.2, Equations 6 & 7).
    Maps genes to biologically functional pathway embeddings (Hypoxia/VHL,
    Chromatin Remodeling/PBRM1/SETD2/BAP1, RTK Signaling/MET, etc.)
    and assigns structured pathway offsets to demographic features.
    """
    def __init__(self, num_genes: int, num_demographics: int, embed_dim: int):
        super().__init__()
        self.num_genes = num_genes
        self.num_demographics = num_demographics
        self.embed_dim = embed_dim
        
        # Learnable pathway encoding matrices Rg and Rd
        self.R_g = nn.Parameter(torch.randn(num_genes, embed_dim) * 0.02)
        self.R_d = nn.Parameter(torch.randn(num_demographics, embed_dim) * 0.02)
        
    def forward(self, X_g: torch.Tensor, X_d: torch.Tensor):
        """
        Args:
            X_g: (B, num_genes, embed_dim)
            X_d: (B, num_demographics, embed_dim)
        Returns:
            X_prime_g: (B, num_genes, embed_dim) [Eq. 6: X_g + R_g]
            X_prime_d: (B, num_demographics, embed_dim) [Eq. 7: X_d + R_d]
        """
        X_prime_g = X_g + self.R_g.unsqueeze(0)
        X_prime_d = X_d + self.R_d.unsqueeze(0)
        return X_prime_g, X_prime_d


class MolecularContextualInteractionAttention(nn.Module):
    """
    Molecular-Contextual Interaction Attention (MCIA) Layer (Section 3.2.3, Eq. 8).
    
    Attention formula:
      A = softmax( ((Q_mc + u) * (K_mc + u)^T + A_TMB) / sqrt(d_k) )
      Output = FFN( ReLU(A * V_mc * W1 + b1) * W2 + b2 )
      
    Where:
      - u: Demographic context vector (age, sex, stage)
      - A_TMB: Tumor-specific TMB mutational context bias
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projections for Q, K, V
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Projections for demographic context u and TMB context
        self.u_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.tmb_attn_proj = nn.Linear(embed_dim, num_heads, bias=True)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_g: torch.Tensor, u_demo: torch.Tensor, s_tmb: torch.Tensor):
        """
        Args:
            x_g: Genomic representations (B, N_g, embed_dim)
            u_demo: Patient demographic vector (B, embed_dim)
            s_tmb: TMB embedding (B, embed_dim)
        Returns:
            Attended representation (B, N_g, embed_dim)
        """
        B, N_g, C = x_g.shape
        
        # Project Q, K, V
        Q = self.w_q(x_g)  # (B, N_g, C)
        K = self.w_k(x_g)  # (B, N_g, C)
        V = self.w_v(x_g)  # (B, N_g, C)
        
        # Demographic bias u incorporated into Query and Key (Eq. 8)
        u = self.u_proj(u_demo).unsqueeze(1)  # (B, 1, C)
        Q_biased = Q + u
        K_biased = K + u
        
        # Reshape for multi-head attention: (B, num_heads, N_g, head_dim)
        Q_b = Q_biased.view(B, N_g, self.num_heads, self.head_dim).transpose(1, 2)
        K_b = K_biased.view(B, N_g, self.num_heads, self.head_dim).transpose(1, 2)
        V_b = V.view(B, N_g, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Raw attention scores
        attn_scores = torch.matmul(Q_b, K_b.transpose(-2, -1)) * self.scale  # (B, H, N_g, N_g)
        
        # TMB bias matrix A_TMB (Section 3.2.3, Eq. 8)
        # Adds patient-specific tumor mutational burden context
        tmb_bias = self.tmb_attn_proj(s_tmb).view(B, self.num_heads, 1, 1)  # (B, H, 1, 1)
        attn_scores = attn_scores + tmb_bias
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum of values
        context = torch.matmul(attn_probs, V_b)  # (B, H, N_g, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, N_g, C)  # (B, N_g, C)
        
        output = self.out_proj(context)
        return output, attn_probs


class MLATLayer(nn.Module):
    """
    Single Transformer Layer for MLAT combining MCIA and two-layer FFN with ReLU (Eq. 9).
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mcia = MolecularContextualInteractionAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        # Two-layer FFN with ReLU activation (Eq. 9: F = ReLU(A W1 + b1) W2 + b2)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_g: torch.Tensor, u_demo: torch.Tensor, s_tmb: torch.Tensor):
        # Attention with residual connection
        x_norm = self.norm1(x_g)
        attn_out, _ = self.mcia(x_norm, u_demo, s_tmb)
        x_g = x_g + attn_out
        
        # FFN with residual connection
        x_g = x_g + self.ffn(self.norm2(x_g))
        return x_g


class MolecularLandscapeAwareTransformer(nn.Module):
    """
    Complete MLAT Module (Section 3.2).
    - Embeds genomic mutation/expression matrix Z (Eq. 3)
    - Embeds demographic vector P (Eq. 4)
    - Embeds continuous TMB K_tmb (Eq. 5)
    - Adds pathway encoding R_g and R_d (Eq. 6, 7)
    - Stacks 12 MCIA Transformer Layers (Section 4 specifications)
    - Outputs genomic representation O_MCIA and normalized contrastive projection.
    """
    def __init__(
        self,
        num_genes: int = 5,           # Driver mutations: VHL, MET, PBRM1, SETD2, BAP1
        num_demographics: int = 3,    # Age, Sex, Stage
        embed_dim: int = 256,
        num_layers: int = 12,         # 12 transformer layers as specified in Section 4
        num_heads: int = 8,           # 8 attention heads
        dropout: float = 0.2,         # Dropout rate 0.2
        contrastive_dim: int = 128    # Contrastive projection dimension
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.num_demographics = num_demographics
        
        # 1. Genomic Linear Projection We (Eq. 3: Xg = Z * We)
        self.gene_embed = nn.Linear(1, embed_dim)
        
        # 2. Demographic Projection Wd + bd (Eq. 4: Xd = Zd * Wd + bd)
        self.demo_embed = nn.Linear(num_demographics, embed_dim)
        
        # 3. TMB continuous embedding f(K_tmb) (Eq. 5: S_tmb = f(K_tmb))
        self.tmb_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 4. Biological Pathway-Aware Encoding (Eq. 6 & 7)
        self.pathway_encoding = BiologicalPathwayEncoding(num_genes, 1, embed_dim)
        
        # 5. Stacked MLAT Layers (12 layers, 8 heads)
        self.layers = nn.ModuleList([
            MLATLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Contrastive head for genomic InfoNCE alignment (Eq. 10)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, contrastive_dim)
        )
        
    def forward(self, genomic_features: torch.Tensor, demo_features: torch.Tensor, tmb: torch.Tensor):
        """
        Args:
            genomic_features: (B, num_genes) mutation/expression status
            demo_features: (B, num_demographics) age, sex, tumor stage
            tmb: (B, 1) or (B,) tumor mutational burden values
        Returns:
            O_MCIA: (B, embed_dim) pooled genomic representation
            z_contrastive: (B, contrastive_dim) L2-normalized feature for InfoNCE loss
        """
        B = genomic_features.shape[0]
        if tmb.dim() == 1:
            tmb = tmb.unsqueeze(-1)
            
        # Reshape genomic features to (B, num_genes, 1) and project (Eq. 3)
        x_g = self.gene_embed(genomic_features.unsqueeze(-1))  # (B, N_g, embed_dim)
        
        # Project demographic features (Eq. 4)
        x_d = self.demo_embed(demo_features).unsqueeze(1)  # (B, 1, embed_dim)
        
        # Project TMB (Eq. 5)
        s_tmb = self.tmb_embed(tmb)  # (B, embed_dim)
        
        # Add Biological Pathway-Aware Encodings Rg and Rd (Eq. 6 & 7)
        x_g, x_d = self.pathway_encoding(x_g, x_d)
        u_demo = x_d.squeeze(1)  # (B, embed_dim)
        
        # Pass through 12 MLAT Transformer layers
        for layer in self.layers:
            x_g = layer(x_g, u_demo, s_tmb)
            
        x_g = self.norm(x_g)
        
        # Mean pooling across gene tokens to obtain patient genomic representation O_MCIA
        O_MCIA = x_g.mean(dim=1)  # (B, embed_dim)
        
        # Contrastive representation projection (Eq. 10)
        z_contrastive = F.normalize(self.contrastive_proj(O_MCIA), p=2, dim=-1)
        
        return O_MCIA, z_contrastive
