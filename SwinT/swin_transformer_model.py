"""
Swin Transformer Model for Material Synthesizability Prediction
Based on the Swin Transformer architecture with adaptations for FTCP data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FTCPPatchEmbed(nn.Module):
    """FTCP Patch Embedding for Swin Transformer."""
    def __init__(self, ftcp_seq_length=600, ftcp_feature_dim=63, embed_dim=96, patch_size=4):
        super().__init__()
        self.ftcp_seq_length = ftcp_seq_length
        self.ftcp_feature_dim = ftcp_feature_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.num_patches_h = ftcp_seq_length // patch_size
        self.num_patches_w = ftcp_feature_dim // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding layer
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x shape: (batch_size, 600, 63)
        # Add channel dimension: (batch_size, 1, 600, 63)
        x = x.unsqueeze(1)
        
        # Apply patch embedding: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (batch_size, embed_dim, num_patches)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Apply layer norm
        x = self.norm(x)
        
        return x, H, W


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
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
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Window attention
        x = self.attn(x)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer for FTCP data classification."""
    def __init__(self, ftcp_seq_length=600, ftcp_feature_dim=63, num_classes=2, 
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        self.patch_embed = FTCPPatchEmbed(
            ftcp_seq_length=ftcp_seq_length,
            ftcp_feature_dim=ftcp_feature_dim,
            embed_dim=embed_dim
        )
        
        # Build layers - simplified version without dimension scaling
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim,  # Keep same dimension throughout
                    num_heads=num_heads[0],  # Use first num_heads value
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate * (sum(depths[:i_layer]) + j) / (sum(depths) - 1)
                ) for j, i in enumerate(range(depths[i_layer]))
            ])
            self.layers.append(layer)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Patch embedding
        x, H, W = self.patch_embed(x)
        
        # Apply Swin Transformer blocks
        for layer in self.layers:
            for block in layer:
                x = block(x, H, W)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        return x


def create_swin_transformer(num_classes=2, embed_dim=96, depths=[2, 2, 6, 2], 
                           num_heads=[3, 6, 12, 24], drop_rate=0.1):
    """
    Create a Swin Transformer model for FTCP classification.
    
    Args:
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        depths (list): Number of blocks in each stage
        num_heads (list): Number of attention heads in each stage
        drop_rate (float): Dropout rate
        
    Returns:
        SwinTransformer model
    """
    model = SwinTransformer(
        ftcp_seq_length=600,
        ftcp_feature_dim=63,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=7,
        drop_rate=drop_rate,
        attn_drop_rate=drop_rate,
        drop_path_rate=0.1
    )
    return model

