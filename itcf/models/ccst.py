"""
Cellular Characteristics-Aware Swin Transformer (CCST)
======================================================
Implementation of Section 3.3 of the ITCF manuscript:
- Patch partitioning and linear patch embedding for 224x224 histopathology images (Equations 11, 12)
- Window-based Cellular Peripheral Attention in encoder (Equations 13, 14, 15, 16)
  with inverse boundary distance weighting: beta(j_p) = 1 / d(j_p, dD_k)
- Shift-Window Based Cellular Peripheral Attention (SW-CPA) in decoder (Equation 17)
- Margin-based patch contrastive loss (Equation 18)
- 12 transformer layers, 8 attention heads, dropout 0.2
- Extracts histopathological cellular representation O_SW-CPA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_peripheral_weights(window_size: int, device: torch.device) -> torch.Tensor:
    """
    Computes peripheral attention weight beta(j_p) = 1 / d(j_p, dD_k) (Eq. 14).
    Distance from patch (x, y) to window boundary dD_k:
      d(j_p, dD_k) = min(x, y, window_size - 1 - x, window_size - 1 - y) + 1
    
    Returns:
      beta: (window_size * window_size,) tensor of positive weights
    """
    beta = torch.zeros((window_size, window_size), dtype=torch.float32, device=device)
    for y in range(window_size):
        for x in range(window_size):
            d = min(x, y, window_size - 1 - x, window_size - 1 - y) + 1
            beta[y, x] = 1.0 / float(d)
    return beta.view(-1)  # (N_win,)


class WindowCellularPeripheralAttention(nn.Module):
    """
    Window-based Cellular Peripheral Attention (Section 3.3.2, Eq. 13-16)
    and Shift-Window Based Cellular Peripheral Attention (Section 3.3.3, Eq. 17).
    """
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.2,
        proj_drop: float = 0.2
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Register peripheral distance buffer (Eq. 14)
        beta = compute_peripheral_weights(window_size, torch.device('cpu'))
        self.register_buffer('beta', beta)
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Relative position index coordinates
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (B * num_windows, N_win, dim)
            mask: (num_windows, N_win, N_win) or None for shifted window attention
        Returns:
            Attended window features: (B * num_windows, N_win, dim)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)
        
        # Dot-product similarity (Eq. 13)
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        sim = sim + relative_position_bias.unsqueeze(0)
        
        # Equation 15: Modulate by peripheral weight beta(j_p_i)
        # beta: (N,), expands to (1, 1, N, 1)
        beta_weight = self.beta.view(1, 1, N, 1)
        sim = sim * beta_weight
        
        if mask is not None:
            nW = mask.shape[0]
            sim = sim.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            sim = sim.view(-1, self.num_heads, N, N)
            
        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)
        
        # Equation 16: Weighted sum of features
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def window_partition(x: torch.Tensor, window_size: int):
    """
    Partitions feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
    Returns:
        windows: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Reconstructs feature map from partitioned windows.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CCSTBlock(nn.Module):
    """
    CCST Transformer Block supporting both regular window attention (encoder)
    and shifted window attention (decoder) with peripheral boundary weighting.
    """
    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        num_heads: int = 8,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        H, W = input_resolution
        if min(H, W) < self.window_size:
            self.window_size = min(H, W)
            self.shift_size = 0
            
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowCellularPeripheralAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Calculate attention mask for shifted window attention
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) does not match resolution ({H}x{W})"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift if shift_size > 0 (Shift-Window CPA, Eq. 17)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*nW, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B*nW, N_win, C)
        
        # Window-based cellular peripheral attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        
        # FFN with residual connection
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class CellularCharacteristicsAwareSwinTransformer(nn.Module):
    """
    Complete CCST Architecture (Section 3.3).
    - Processes 224x224 histopathological diagnostic H&E slides / patches
    - Non-overlapping patch partitioning (U = 16) and linear projection (Eq. 11, 12)
    - 12 Transformer layers configured with alternating Window-based CPA (encoder)
      and Shift-Window CPA (decoder)
    - Margin-based patch contrastive projection (Eq. 18)
    - Outputs histopathological feature representation O_SW_CPA.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        window_size: int = 7,
        dropout: float = 0.2,
        contrastive_dim: int = 128
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 224 / 16 = 14
        self.num_patches = self.grid_size * self.grid_size  # 14 * 14 = 196
        self.embed_dim = embed_dim
        
        # Equation 12: Patch Embedding f(j_p) = W_emb * j_p + b_emb
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 12 layers: alternating between regular window (shift=0) and shifted window (shift=window_size//2)
        shift_size = window_size // 2
        self.blocks = nn.ModuleList([
            CCSTBlock(
                dim=embed_dim,
                input_resolution=(self.grid_size, self.grid_size),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Contrastive head for patch margin contrastive loss (Eq. 18)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, contrastive_dim)
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Histopathology image tensor (B, 3, 224, 224)
        Returns:
            O_SW_CPA: (B, embed_dim) pooled histopathology feature representation
            z_contrastive: (B, contrastive_dim) normalized representation for contrastive learning
        """
        B = x.shape[0]
        # Linear patch embedding (Eq. 12)
        x_patches = self.patch_embed(x)  # (B, embed_dim, 14, 14)
        x_patches = x_patches.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        x_patches = self.pos_drop(x_patches)
        
        # Forward through CCST blocks
        for block in self.blocks:
            x_patches = block(x_patches)
            
        x_patches = self.norm(x_patches)
        
        # Global average pooling across patches to produce O_SW_CPA
        O_SW_CPA = x_patches.mean(dim=1)  # (B, embed_dim)
        
        # L2-normalized contrastive projection for patch contrastive loss (Eq. 18)
        z_contrastive = F.normalize(self.contrastive_proj(O_SW_CPA), p=2, dim=-1)
        
        return O_SW_CPA, z_contrastive
