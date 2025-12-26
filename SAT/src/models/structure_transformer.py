import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds information about the position of each element in the sequence.
    """
    def __init__(self, d_model, max_len=600):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1)]

class StructureEncoder(nn.Module):
    """
    Encodes different components of the FTCP structure separately,
    respecting their physical meaning.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Embedding layers for each component
        # Each component has shape (n_rows, 63) where n_rows varies by component
        self.element_encoding_linear = nn.Linear(93 * 63, hidden_dim)  # 93 rows with 63 features
        self.lattice_params_linear = nn.Linear(2 * 63, hidden_dim)     # 2 rows with 63 features
        self.site_coords_linear = nn.Linear(200 * 63, hidden_dim)      # 200 rows with 63 features
        self.site_occupancy_linear = nn.Linear(200 * 63, hidden_dim)   # 200 rows with 63 features
        self.element_props_linear = nn.Linear(92 * 63, hidden_dim)     # 92 rows with 63 features
        
        # Layer normalization for each component
        self.element_encoding_norm = nn.LayerNorm(hidden_dim)
        self.lattice_params_norm = nn.LayerNorm(hidden_dim)
        self.site_coords_norm = nn.LayerNorm(hidden_dim)
        self.site_occupancy_norm = nn.LayerNorm(hidden_dim)
        self.element_props_norm = nn.LayerNorm(hidden_dim)
        
        # Final layer normalization after concatenation
        self.final_norm = nn.LayerNorm(hidden_dim * 5)
        self.projection = nn.Linear(hidden_dim * 5, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x_dict):
        """
        Process each component separately, then combine.
        
        Args:
            x_dict: Dictionary of FTCP components
                'element_encoding': [batch_size, 93, 63]
                'lattice_params': [batch_size, 2, 63]
                'site_coords': [batch_size, 200, 63]
                'site_occupancy': [batch_size, 200, 63]
                'element_props': [batch_size, 92, 63]
        
        Returns:
            Encoded representation: [batch_size, hidden_dim]
        """
        batch_size = x_dict['element_encoding'].size(0)
        
        # Flatten each component for processing
        element_encoding_flat = x_dict['element_encoding'].reshape(batch_size, -1)
        lattice_params_flat = x_dict['lattice_params'].reshape(batch_size, -1)
        site_coords_flat = x_dict['site_coords'].reshape(batch_size, -1)
        site_occupancy_flat = x_dict['site_occupancy'].reshape(batch_size, -1)
        element_props_flat = x_dict['element_props'].reshape(batch_size, -1)
        
        # Process each component
        element_encoding = self.element_encoding_norm(
            self.element_encoding_linear(element_encoding_flat)
        )
        
        lattice_params = self.lattice_params_norm(
            self.lattice_params_linear(lattice_params_flat)
        )
        
        site_coords = self.site_coords_norm(
            self.site_coords_linear(site_coords_flat)
        )
        
        site_occupancy = self.site_occupancy_norm(
            self.site_occupancy_linear(site_occupancy_flat)
        )
        
        element_props = self.element_props_norm(
            self.element_props_linear(element_props_flat)
        )
        
        # Concatenate all components
        combined = torch.cat([
            element_encoding, 
            lattice_params, 
            site_coords, 
            site_occupancy, 
            element_props
        ], dim=1)
        
        # Final projection
        combined = self.final_norm(combined)
        output = self.output_norm(self.projection(combined))
        
        return output

class SequenceTransformer(nn.Module):
    """
    A transformer model that directly works with the sequential FTCP data
    without reshaping it to a 2D grid.
    """
    def __init__(self, 
                 input_dim=63, 
                 hidden_dim=128, 
                 nhead=8, 
                 num_layers=6, 
                 dropout=0.1, 
                 use_structure_encoding=True):
        super().__init__()
        
        self.use_structure_encoding = use_structure_encoding
        
        if use_structure_encoding:
            # Use structure-aware encoding
            self.structure_encoder = StructureEncoder(hidden_dim=hidden_dim)
            self.embedding = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Directly process the sequence
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: If use_structure_encoding=True, dictionary of FTCP components
               Else, [batch_size, seq_len, feature_dim] tensor
        
        Returns:
            Classification logits
        """
        if self.use_structure_encoding:
            # Structure-aware encoding
            x = self.structure_encoder(x)
            x = self.embedding(x)
            
            # Add a sequence dimension of length 1
            x = x.unsqueeze(1)
        else:
            # Sequence-based encoding
            # x is [batch_size, seq_len, feature_dim]
            x = self.embedding(x)
            x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Classification
        x = self.fc(x)
        
        return x

class FTCPTransformer(nn.Module):
    """
    Main model that handles both structure-aware and full sequence modes.
    """
    def __init__(self, mode='structure', hidden_dim=128, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.mode = mode
        
        if mode == 'structure':
            # Structure-aware transformer
            self.model = SequenceTransformer(
                hidden_dim=hidden_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                use_structure_encoding=True
            )
        elif mode == 'sequence':
            # Sequential transformer (no reshaping)
            self.model = SequenceTransformer(
                input_dim=63,
                hidden_dim=hidden_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                use_structure_encoding=False
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: For 'structure' mode: Dictionary of FTCP components
               For 'sequence' mode: [batch_size, seq_len, feature_dim] tensor
        
        Returns:
            Classification logits
        """
        if self.mode == 'structure':
            return self.model(x)
        elif self.mode == 'sequence':
            # Ensure x is the full tensor for sequence mode - shape (batch_size, 600, 63)
            full_tensor = x['full_tensor']
            return self.model(full_tensor)
        
        raise ValueError(f"Unknown mode: {self.mode}") 