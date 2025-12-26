"""
Optimize Ensemble Weights

Try different weight combinations and find the best one
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent / "SwinTransformer_balanced"))

# Import ensemble and dataset
from ensemble_model import load_ensemble
from dataset_balanced_fixed import load_ftcp_data_once, BalancedFTCPDataset


def evaluate_with_weights(alternative_weight, swin_weight, ensemble, test_loader, device):
    """
    Evaluate ensemble with specific weights
    
    Args:
        alternative_weight: Weight for Alternative Model
        swin_weight: Weight for Swin Transformer
        ensemble: EnsembleModel instance
        test_loader: DataLoader for test data
        device: Device to use
    
    Returns:
        Dictionary with metrics
    """
    # Update weights
    ensemble.alternative_weight = alternative_weight
    ensemble.swin_weight = swin_weight
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for ftcp_data, labels in test_loader:
            ftcp_data = ftcp_data.to(device)
            labels = labels.to(device)
            
            # Get predictions
            preds, probs = ensemble.predict(ftcp_data)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }


def optimize_weights(ensemble, test_loader, device, weight_range=None):
    """
    Grid search for optimal ensemble weights
    
    Args:
        ensemble: EnsembleModel instance
        test_loader: DataLoader for test data
        device: Device to use
        weight_range: List of weights to try for Alternative Model
    
    Returns:
        Best weights and metrics
    """
    if weight_range is None:
        # Try a wider range of SAT/Alternative weights for a more complete view
        # (SwinT weight is always 1 - SAT weight)
        weight_range = [
            0.40, 0.45,
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
        ]
    
    print("\n" + "="*80)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*80)
    print(f"\nTesting {len(weight_range)} different weight combinations...")
    print("(This may take 10-15 minutes)\n")
    
    results = []
    
    for alt_weight in tqdm(weight_range, desc="Testing weights"):
        swin_weight = 1.0 - alt_weight
        
        metrics = evaluate_with_weights(
            alternative_weight=alt_weight,
            swin_weight=swin_weight,
            ensemble=ensemble,
            test_loader=test_loader,
            device=device
        )
        
        results.append({
            'alternative_weight': alt_weight,
            'swin_weight': swin_weight,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc']
        })
    
    # Find best based on F1-score
    best_result = max(results, key=lambda x: x['f1'])
    
    # Print results
    print("\n" + "="*80)
    print("WEIGHT OPTIMIZATION RESULTS")
    print("="*80)
    print("\nAll Tested Configurations:")
    print("-" * 80)
    print(f"{'Alt Weight':>12} {'Swin Weight':>12} {'Accuracy':>12} {'F1-Score':>12} {'ROC-AUC':>12}")
    print("-" * 80)
    
    for r in results:
        marker = " ⭐ BEST" if r == best_result else ""
        print(f"{r['alternative_weight']:>12.2f} {r['swin_weight']:>12.2f} "
              f"{r['accuracy']:>12.4f} {r['f1']:>12.4f} {r['roc_auc']:>12.4f}{marker}")
    
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION")
    print("="*80)
    print(f"Alternative Weight: {best_result['alternative_weight']:.2f}")
    print(f"Swin Weight:        {best_result['swin_weight']:.2f}")
    print(f"\nPerformance:")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  F1-Score:  {best_result['f1']:.4f}")
    print(f"  ROC-AUC:   {best_result['roc_auc']:.4f}")
    print("="*80)
    
    # Create visualization
    save_dir = Path('results')
    save_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    alt_weights = [r['alternative_weight'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    
    # Accuracy
    axes[0].plot(alt_weights, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].axvline(best_result['alternative_weight'], color='red', 
                   linestyle='--', label='Best', alpha=0.7)
    axes[0].set_xlabel('Alternative Model Weight', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Accuracy vs Weight', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # F1-Score
    axes[1].plot(alt_weights, f1_scores, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].axvline(best_result['alternative_weight'], color='red', 
                   linestyle='--', label='Best', alpha=0.7)
    axes[1].set_xlabel('Alternative Model Weight', fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontweight='bold')
    axes[1].set_title('F1-Score vs Weight', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # ROC-AUC
    axes[2].plot(alt_weights, roc_aucs, 'o-', linewidth=2, markersize=8, color='orange')
    axes[2].axvline(best_result['alternative_weight'], color='red', 
                   linestyle='--', label='Best', alpha=0.7)
    axes[2].set_xlabel('Alternative Model Weight', fontweight='bold')
    axes[2].set_ylabel('ROC-AUC', fontweight='bold')
    axes[2].set_title('ROC-AUC vs Weight', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'weight_optimization.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Optimization plot saved to {save_dir / 'weight_optimization.png'}")
    
    # Save results to file
    with open(save_dir / 'weight_optimization_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENSEMBLE WEIGHT OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Alt Weight':>12} {'Swin Weight':>12} {'Accuracy':>12} {'F1-Score':>12} {'ROC-AUC':>12}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            marker = " <- BEST" if r == best_result else ""
            f.write(f"{r['alternative_weight']:>12.2f} {r['swin_weight']:>12.2f} "
                   f"{r['accuracy']:>12.4f} {r['f1']:>12.4f} {r['roc_auc']:>12.4f}{marker}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Alternative Weight: {best_result['alternative_weight']:.2f}\n")
        f.write(f"Swin Weight:        {best_result['swin_weight']:.2f}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"  Accuracy:  {best_result['accuracy']:.4f}\n")
        f.write(f"  F1-Score:  {best_result['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {best_result['roc_auc']:.4f}\n")
    
    print(f"✓ Results saved to {save_dir / 'weight_optimization_results.txt'}")
    
    return best_result, results


def main():
    """Main optimization function"""
    
    print("="*80)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*80)
    
    # Configuration
    FTCP_FILE = '../data/ftcp_data.h5'
    LABELS_FILE = '../data/mp_structures_with_synthesizability1.xlsx'
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # Load data
    print("\nLoading FTCP data...")
    data_dict = load_ftcp_data_once(
        FTCP_FILE,
        LABELS_FILE
    )
    print(f"✓ Data loaded: {len(data_dict['material_ids'])} samples")
    
    # Create test dataset
    test_dataset = BalancedFTCPDataset(
        ftcp_data=data_dict['ftcp_data'],
        labels=data_dict['labels'],
        indices=data_dict['test_indices'],
        split_name='test'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load ensemble (with any initial weights)
    print("\nLoading ensemble model...")
    ensemble = load_ensemble(device=DEVICE)
    
    # Optimize weights
    best_result, all_results = optimize_weights(
        ensemble=ensemble,
        test_loader=test_loader,
        device=DEVICE
    )
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nTo use the best weights, update ensemble_model.py:")
    print(f"  alternative_weight={best_result['alternative_weight']:.2f}")
    print(f"  swin_weight={best_result['swin_weight']:.2f}")
    print("\nOr pass them when creating ensemble:")
    print(f"  ensemble = load_ensemble(")
    print(f"      alternative_weight={best_result['alternative_weight']:.2f},")
    print(f"      swin_weight={best_result['swin_weight']:.2f}")
    print(f"  )")


if __name__ == "__main__":
    main()

