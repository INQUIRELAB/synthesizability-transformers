"""
Training script for FT-Transformer on Materials Synthesizability Prediction.

Configuration:
- Focal Loss + Class Weights for imbalanced data
- FP32 training (no mixed precision for Blackwell GPU stability)
- Gradient accumulation for effective batch size
- Single data load with shared memory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SwinTransformer_balanced'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import model and dataset
from ft_transformer_model import create_ft_transformer
from dataset_balanced_fixed import get_balanced_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - focuses on hard examples."""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='[Train]')
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        running_loss += loss.item() * accumulation_steps  # Unscale for logging
        preds = output.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
        })
    
    # Handle any remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'f1': epoch_f1
    }


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='[Test]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1,
        'roc_auc': epoch_auc,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, epoch, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Synth', 'Synth'],
                yticklabels=['Not Synth', 'Synth'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch}.png'), dpi=150)
    plt.close()


def plot_training_history(history, save_dir):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['test_acc'], label='Test')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[0, 2].plot(history['train_f1'], label='Train')
    axes[0, 2].plot(history['test_f1'], label='Test')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Precision
    axes[1, 0].plot(history['test_precision'], label='Test Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history['test_recall'], label='Test Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # ROC-AUC
    axes[1, 2].plot(history['test_roc_auc'], label='Test ROC-AUC')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('ROC-AUC')
    axes[1, 2].set_title('ROC-AUC over Time')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


def train_model():
    """Main training function."""
    
    # Configuration
    FTCP_FILE = '../data/ftcp_data.h5'
    LABELS_FILE = '../data/mp_structures_with_synthesizability1.xlsx'
    BATCH_SIZE = 32  # Physical batch size (reduced for full attention memory)
    ACCUMULATION_STEPS = 8  # Effective batch size = 32 * 8 = 256
    NUM_EPOCHS = 200
    LEARNING_RATE = 7.5e-5  # Balanced: not too fast (1e-4) or too slow (5e-5)
    WEIGHT_DECAY = 0.005  # Reduced for faster initial learning
    WARMUP_EPOCHS = 10  # LR warmup for stable start
    NUM_WORKERS = 0  # Avoid multiprocessing issues
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("FT-TRANSFORMER ON BALANCED DATA")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("="*70)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    print("\nLoading balanced datasets...")
    train_loader, test_loader = get_balanced_dataloaders(
        FTCP_FILE,
        LABELS_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Calculate class weights for imbalanced data
    print("\nCalculating class weights...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = torch.FloatTensor([
        total_samples / (len(class_counts) * class_counts[0]),
        total_samples / (len(class_counts) * class_counts[1])
    ]).to(device)
    
    print(f"Class 0 weight: {class_weights[0]:.4f}")
    print(f"Class 1 weight: {class_weights[1]:.4f}")
    
    # Initialize model
    print("\nInitializing FT-Transformer model...")
    model = create_ft_transformer(num_atoms=600, in_features=63, num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function: Focal Loss with class weights
    criterion = FocalLoss(
        alpha=2.0,  # Focus more on hard examples
        gamma=2.0,  # Reduce loss contribution from easy examples
        weight=class_weights  # Handle class imbalance
    )
    
    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler: Warmup + CosineAnnealing
    # Warmup: Gradually increase LR from 0 to LEARNING_RATE over WARMUP_EPOCHS
    # Then: Cosine decay from LEARNING_RATE to LEARNING_RATE * 0.01
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of LR
        end_factor=1.0,     # Reach 100% of LR
        total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - WARMUP_EPOCHS,
        eta_min=LEARNING_RATE * 0.01
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'training_log_{timestamp}.txt')
    
    with open(log_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FT-TRANSFORMER TRAINING LOG\n")
        f.write("="*70 + "\n")
        f.write(f"Model: FT-Transformer\n")
        f.write(f"Architecture: embed_dim=128, depth=6, heads=8\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Batch Size: {BATCH_SIZE} (physical) x {ACCUMULATION_STEPS} (accum) = {BATCH_SIZE * ACCUMULATION_STEPS} (effective)\n")
        f.write(f"Note: Batch size reduced to 32 due to full attention memory requirements\n")
        f.write(f"Loss: Focal Loss (alpha=2.0, gamma=2.0) + Class Weights\n")
        f.write(f"Precision: FP32 (no mixed precision)\n")
        f.write(f"Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")
        f.write(f"Scheduler: Warmup ({WARMUP_EPOCHS} epochs) + CosineAnnealingLR\n")
        f.write(f"LR Schedule: {LEARNING_RATE*0.01:.2e} -> {LEARNING_RATE:.2e} (warmup) -> {LEARNING_RATE*0.01:.2e} (decay)\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*70 + "\n\n")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [],
        'test_recall': [], 'test_f1': [], 'test_roc_auc': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=ACCUMULATION_STEPS
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_precision'].append(test_metrics['precision'])
        history['test_recall'].append(test_metrics['recall'])
        history['test_f1'].append(test_metrics['f1'])
        history['test_roc_auc'].append(test_metrics['roc_auc'])
        
        # Print metrics
        print("\n" + "="*70)
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print("="*70)
        print(f"Train Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {train_metrics['f1']:.4f}")
        print(f"\nTest Metrics:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"EPOCH {epoch}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Train Metrics:\n")
            f.write(f"  Loss: {train_metrics['loss']:.4f}\n")
            f.write(f"  Accuracy: {train_metrics['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {train_metrics['f1']:.4f}\n")
            f.write(f"\nTest Metrics:\n")
            f.write(f"  Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {test_metrics['f1']:.4f}\n")
            f.write(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}\n")
        
        # Save best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'test_metrics': test_metrics,
            }, os.path.join('checkpoints', 'best_model.pth'))
            
            print(f"\n✓ Best model saved! F1: {best_f1:.4f}")
        
        print(f"Best F1: {best_f1:.4f} (Epoch {best_epoch})")
        
        # Save confusion matrix every 10 epochs
        if epoch % 10 == 0:
            plot_confusion_matrix(
                test_metrics['confusion_matrix'],
                epoch,
                'results'
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_metrics': test_metrics,
            }, os.path.join('checkpoints', f'checkpoint_epoch_{epoch}.pth'))
    
    # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best F1 Score: {best_f1:.4f} (Epoch {best_epoch})")
    
    # Plot training history
    plot_training_history(history, 'results')
    
    # Save final results
    final_results = {
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'final_metrics': {
            'loss': test_metrics['loss'],
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'roc_auc': test_metrics['roc_auc']
        },
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'history': history
    }
    
    with open(os.path.join('results', 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved to results/")
    print(f"✓ Best model saved to checkpoints/best_model.pth")
    print(f"✓ Training log saved to {log_file}")
    print("="*70)


if __name__ == "__main__":
    train_model()

