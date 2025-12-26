"""
Ensemble Model combining Alternative Model and Swin Transformer
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
class_dir = current_dir.parent
sys.path.insert(0, str(class_dir / "Alternative Model" / "src"))
sys.path.insert(0, str(class_dir / "SwinTransformer_balanced"))

# Import Alternative Model
from models.structure_transformer import FTCPTransformer as AlternativeModel

# Import Swin Transformer
from swin_transformer_model import create_swin_transformer as create_swin_model

class EnsembleModel(nn.Module):
    """
    Ensemble of Alternative Model and Swin Transformer
    
    Combines predictions using weighted average:
    - Alternative Model: 60% weight (better performer: 89%)
    - Swin Transformer: 40% weight (good performer: 87.3%)
    """
    def __init__(
        self,
        alternative_model_path,
        swin_model_path,
        device='cuda',
        alternative_weight=0.6,
        swin_weight=0.4
    ):
        super().__init__()
        
        self.device = device
        self.alternative_weight = alternative_weight
        self.swin_weight = swin_weight
        
        # Initialize Alternative Model
        print("Loading Alternative Model...")
        self.alternative_model = AlternativeModel(
            mode='structure',
            hidden_dim=128,
            nhead=8,
            num_layers=6,
            dropout=0.1
        )
        
        # Load Alternative Model weights
        checkpoint = torch.load(alternative_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.alternative_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.alternative_model.load_state_dict(checkpoint)
        
        self.alternative_model.to(device)
        self.alternative_model.eval()
        print(f"✓ Alternative Model loaded from {alternative_model_path}")
        
        # Initialize Swin Transformer
        print("Loading Swin Transformer...")
        self.swin_model = create_swin_model()
        
        # Load Swin Model weights
        checkpoint = torch.load(swin_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.swin_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.swin_model.load_state_dict(checkpoint)
        
        self.swin_model.to(device)
        self.swin_model.eval()
        print(f"✓ Swin Transformer loaded from {swin_model_path}")
        
        print(f"\nEnsemble Weights:")
        print(f"  Alternative Model: {self.alternative_weight:.1%}")
        print(f"  Swin Transformer: {self.swin_weight:.1%}")
    
    def prepare_alternative_input(self, ftcp_data):
        """
        Prepare input for Alternative Model (structure-aware format)
        
        Args:
            ftcp_data: [batch_size, 600, 63] tensor
        
        Returns:
            Dictionary with 5 components
        """
        batch_size = ftcp_data.shape[0]
        
        # Split FTCP into its 5 structural components
        # Based on Alternative Model's actual dataset structure
        # NOTE: There's a padding row at index 495 that is skipped
        components = {
            'element_encoding': ftcp_data[:, 0:93, :],      # Rows 0-92: 93 rows
            'lattice_params': ftcp_data[:, 93:95, :],       # Rows 93-94: 2 rows
            'site_coords': ftcp_data[:, 95:295, :],         # Rows 95-294: 200 rows
            'site_occupancy': ftcp_data[:, 295:495, :],     # Rows 295-494: 200 rows
            'element_props': ftcp_data[:, 496:588, :]       # Rows 496-587: 92 rows (skip row 495!)
        }
        
        return components
    
    def prepare_swin_input(self, ftcp_data):
        """
        Prepare input for Swin Transformer (no change needed)
        
        Args:
            ftcp_data: [batch_size, 600, 63] tensor
        
        Returns:
            Same tensor (Swin handles reshaping internally)
        """
        return ftcp_data
    
    def forward(self, ftcp_data):
        """
        Forward pass through ensemble
        
        Args:
            ftcp_data: [batch_size, 600, 63] tensor
        
        Returns:
            Combined predictions (logits)
        """
        with torch.no_grad():
            # Get Alternative Model predictions
            alternative_input = self.prepare_alternative_input(ftcp_data)
            alternative_logits = self.alternative_model(alternative_input)
            alternative_probs = torch.softmax(alternative_logits, dim=1)
            
            # Get Swin Transformer predictions
            swin_input = self.prepare_swin_input(ftcp_data)
            swin_logits = self.swin_model(swin_input)
            swin_probs = torch.softmax(swin_logits, dim=1)
            
            # Weighted average of probabilities
            ensemble_probs = (
                self.alternative_weight * alternative_probs +
                self.swin_weight * swin_probs
            )
            
            # Convert back to logits for consistency
            # Add small epsilon to avoid log(0)
            ensemble_logits = torch.log(ensemble_probs + 1e-10)
        
        return ensemble_logits, alternative_logits, swin_logits
    
    def predict(self, ftcp_data):
        """
        Get predictions from ensemble
        
        Args:
            ftcp_data: [batch_size, 600, 63] tensor
        
        Returns:
            predictions: [batch_size] tensor of predicted classes
            ensemble_probs: [batch_size, 2] tensor of probabilities
        """
        ensemble_logits, _, _ = self.forward(ftcp_data)
        ensemble_probs = torch.softmax(ensemble_logits, dim=1)
        predictions = torch.argmax(ensemble_probs, dim=1)
        
        return predictions, ensemble_probs
    
    def predict_with_individual_models(self, ftcp_data):
        """
        Get predictions from all models (ensemble + individuals)
        
        Returns:
            Dictionary with predictions from all models
        """
        ensemble_logits, alternative_logits, swin_logits = self.forward(ftcp_data)
        
        # Ensemble
        ensemble_probs = torch.softmax(ensemble_logits, dim=1)
        ensemble_preds = torch.argmax(ensemble_probs, dim=1)
        
        # Alternative
        alternative_probs = torch.softmax(alternative_logits, dim=1)
        alternative_preds = torch.argmax(alternative_probs, dim=1)
        
        # Swin
        swin_probs = torch.softmax(swin_logits, dim=1)
        swin_preds = torch.argmax(swin_probs, dim=1)
        
        return {
            'ensemble': {
                'predictions': ensemble_preds,
                'probabilities': ensemble_probs
            },
            'alternative': {
                'predictions': alternative_preds,
                'probabilities': alternative_probs
            },
            'swin': {
                'predictions': swin_preds,
                'probabilities': swin_probs
            }
        }


def load_ensemble(
    alternative_model_path=None,
    swin_model_path=None,
    device='cuda',
    alternative_weight=0.6,
    swin_weight=0.4
):
    """
    Load ensemble model with default paths
    
    Args:
        alternative_model_path: Path to Alternative Model checkpoint
        swin_model_path: Path to Swin Transformer checkpoint
        device: Device to load models on
        alternative_weight: Weight for Alternative Model predictions
        swin_weight: Weight for Swin Transformer predictions
    
    Returns:
        EnsembleModel instance
    """
    # Default paths
    if alternative_model_path is None:
        alternative_model_path = str(
            Path(__file__).parent.parent / 
            "Alternative Model" / "results" / "best_model.pt"
        )
    
    if swin_model_path is None:
        swin_model_path = str(
            Path(__file__).parent.parent / 
            "SwinTransformer_balanced" / "checkpoints" / "best_model.pth"
        )
    
    # Check if files exist
    if not os.path.exists(alternative_model_path):
        raise FileNotFoundError(f"Alternative Model not found at: {alternative_model_path}")
    
    if not os.path.exists(swin_model_path):
        raise FileNotFoundError(f"Swin Transformer not found at: {swin_model_path}")
    
    # Create ensemble
    ensemble = EnsembleModel(
        alternative_model_path=alternative_model_path,
        swin_model_path=swin_model_path,
        device=device,
        alternative_weight=alternative_weight,
        swin_weight=swin_weight
    )
    
    return ensemble

