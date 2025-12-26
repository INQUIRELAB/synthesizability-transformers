"""
FT-Transformer (Feature Tokenizer Transformer) Model
Specifically designed for tabular/feature-based data like FTCP.

Reference: "Revisiting Deep Learning Models for Tabular Data" (2021)
Paper: https://arxiv.org/abs/2106.11959

Architecture:
- Feature tokenization: Convert each atom's features to embeddings
- Transformer encoder with attention over atoms
- CLS token for global representation
- Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureTokenizer(nn.Module):
    """
    Tokenizes input features by projecting them to embedding space.
    Each atom (with 63 features) becomes a token.
    """
    def __init__(self, in_features, embed_dim):
        super().__init__()
        self.projection = nn.Linear(in_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_atoms, num_features) - e.g., (B, 600, 63)
        Returns:
            tokens: (batch, num_atoms, embed_dim)
        """
        x = self.projection(x)  # (B, 600, embed_dim)
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            out: (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-normalization."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            out: (batch, seq_len, embed_dim)
        """
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for materials synthesizability prediction.
    
    Designed for FTCP data: (batch, 600 atoms, 63 features)
    
    Architecture matches Swin Transformer complexity (~1.3M parameters)
    """
    def __init__(
        self,
        in_features=63,
        num_atoms=600,
        embed_dim=96,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Feature tokenizer: project each atom's features to embedding space
        self.feature_tokenizer = FeatureTokenizer(in_features, embed_dim)
        
        # CLS token for classification (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings for atoms (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_atoms + 1, embed_dim))  # +1 for CLS
        
        # Dropout after embedding
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Initialize CLS token and positional embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_atoms, in_features) - e.g., (B, 600, 63)
        Returns:
            logits: (batch, num_classes)
        """
        B = x.shape[0]
        
        # Feature tokenization: (B, 600, 63) -> (B, 600, embed_dim)
        x = self.feature_tokenizer(x)
        
        # Prepend CLS token: (B, 601, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 601, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final norm
        x = self.norm(x)
        
        # Extract CLS token for classification
        cls_output = x[:, 0]  # (B, embed_dim)
        
        # Classification head
        logits = self.head(cls_output)  # (B, num_classes)
        
        return logits
    
    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization.
        Useful for understanding which atoms the model focuses on.
        """
        B = x.shape[0]
        
        # Feature tokenization
        x = self.feature_tokenizer(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        attention_maps = []
        
        # Collect attention from each block
        for block in self.blocks:
            # Get attention weights (before applying to values)
            qkv = block.attn.qkv(block.norm1(x)).reshape(
                B, x.shape[1], 3, block.attn.num_heads, block.attn.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            
            attention_maps.append(attn.detach())
            
            # Continue forward pass
            x = block(x)
        
        return attention_maps


def create_ft_transformer(num_atoms=600, in_features=63, num_classes=2):
    """
    Create FT-Transformer with architecture matching Swin Transformer complexity.
    
    Args:
        num_atoms: Number of atoms in structure (600 for FTCP)
        in_features: Number of features per atom (63 for FTCP)
        num_classes: Number of output classes (2 for binary classification)
    
    Returns:
        model: FTTransformer instance
    """
    model = FTTransformer(
        in_features=in_features,
        num_atoms=num_atoms,
        embed_dim=128,  # Increased from 96 to match Swin complexity
        depth=6,  # Increased from 4 to match Swin complexity
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=num_classes
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("FT-TRANSFORMER MODEL TEST")
    print("="*70)
    
    model = create_ft_transformer()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Input: (batch, 600 atoms, 63 features)")
    print(f"  Embedding dim: 128")
    print(f"  Depth: 6 transformer blocks")
    print(f"  Attention heads: 8")
    print(f"  Output: (batch, 2 classes)")
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 600, 63)
    
    print(f"\nTesting forward pass...")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output[0]}")
    
    print("\n✓ Model test passed!")
    print("="*70)

