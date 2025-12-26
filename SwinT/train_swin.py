"""
Training Script for Swin Transformer on Balanced Data
Comprehensive training with epoch-by-epoch logging and evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

from dataset_balanced_fixed import get_balanced_dataloaders
from swin_transformer_model import create_swin_transformer


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"SWIN TRANSFORMER ON BALANCED DATA")
print(f"{'='*70}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
print(f"{'='*70}\n")


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


def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, num_epochs, accumulation_steps=8):
    """Train for one epoch with gradient accumulation (FP32 mode)."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Regular forward pass (no mixed precision to avoid Blackwell GPU issues)
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps  # Scale loss
        
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping to prevent NaN/Inf
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        
        # Store predictions
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item()*accumulation_steps:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def evaluate(model, test_loader, criterion, epoch, num_epochs):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]', leave=False)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Store predictions and probabilities
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    return metrics, (all_labels, all_preds, all_probs)


def plot_confusion_matrix(labels, preds, epoch, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add class names
    classes = ['Not Synthesizable', 'Synthesizable']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks + 0.5, classes, rotation=45)
    plt.yticks(tick_marks + 0.5, classes, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_epoch_results(epoch, train_metrics, test_metrics, log_file):
    """Save epoch results to log file."""
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"EPOCH {epoch+1}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Train Metrics:\n")
        f.write(f"  Loss: {train_metrics['loss']:.4f}\n")
        f.write(f"  Accuracy: {train_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {train_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {train_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {train_metrics['f1']:.4f}\n")
        f.write(f"\nTest Metrics:\n")
        f.write(f"  Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {test_metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}\n")


def train_model(
    ftcp_file='../data/ftcp_data.h5',
    labels_file='../data/mp_structures_with_synthesizability1.xlsx',
    num_epochs=200,
    batch_size=32,  # Physical batch size (effective batch = 32 * 8 = 256 with gradient accumulation)
    learning_rate=0.0001,
    save_interval=10
):
    """Main training function - data loaded once and shared between train/test."""
    
    print("Loading balanced datasets...")
    train_loader, test_loader = get_balanced_dataloaders(
        ftcp_file=ftcp_file,
        labels_file=labels_file,
        batch_size=batch_size,
        num_workers=0  # Use 0 to avoid shared memory issues
    )
    
    print("\nInitializing Swin Transformer model...")
    model = create_swin_transformer(
        num_classes=2,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        drop_rate=0.1
    ).to(device)
    
    # Multi-GPU training with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Using {torch.cuda.device_count()} GPUs for training!")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        model = nn.DataParallel(model)
    else:
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate class weights to handle imbalance in training data
    print("\nCalculating class weights...")
    all_train_labels = []
    for _, target in train_loader:
        all_train_labels.extend(target.cpu().numpy())
    
    unique, counts = np.unique(all_train_labels, return_counts=True)
    total = len(all_train_labels)
    class_weights = torch.FloatTensor([total/count for count in counts])
    class_weights = class_weights.to(device)
    
    print(f"Class distribution in training: {dict(zip(unique, counts))}")
    print(f"Class weights: {class_weights}")
    print(f"  Class 0 weight: {class_weights[0]:.4f}")
    print(f"  Class 1 weight: {class_weights[1]:.4f}\n")
    
    # Loss and optimizer (Focal Loss + class weights for best performance on imbalanced data)
    criterion = FocalLoss(
        alpha=2.0,  # Focus more on hard examples
        gamma=2.0,  # Reduce loss contribution from easy examples
        weight=class_weights  # Handle class imbalance
    )
    print(f"Using Focal Loss (alpha=2.0, gamma=2.0) with class weights")
    print(f"This matches Alternative Model's configuration for fair comparison\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=num_training_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [],
        'test_recall': [], 'test_f1': [], 'test_roc_auc': []
    }
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_log_{timestamp}.txt'
    
    with open(log_file, 'w') as f:
        f.write(f"Swin Transformer Training Log - OPTIMIZED FOR BEST RESULTS\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Swin Transformer (Full Model)\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Physical Batch Size: {batch_size}\n")
        f.write(f"Gradient Accumulation Steps: 8\n")
        f.write(f"Effective Batch Size: {batch_size * 8}\n")
        f.write(f"Loss Function: Focal Loss (alpha=2.0, gamma=2.0) + Class Weights\n")
        f.write(f"Class Weights: {class_weights[0]:.4f} (class 0), {class_weights[1]:.4f} (class 1)\n")
        f.write(f"Data Loading: Single load, shared between train/test (memory efficient)\n")
        f.write(f"Precision: FP32 (Blackwell GPU stability)\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Optimizer: AdamW (weight_decay=0.01)\n")
        f.write(f"Scheduler: OneCycleLR (cosine annealing)\n")
        f.write(f"GPUs: {torch.cuda.device_count()} GPUs\n")
        for i in range(torch.cuda.device_count()):
            f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
        f.write(f"Configuration: Matches Alternative Model for fair comparison\n")
        f.write(f"{'='*70}\n")
    
    best_f1 = 0.0
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, num_epochs, accumulation_steps=8
        )
        
        # Evaluate
        test_metrics, (test_labels, test_preds, test_probs) = evaluate(
            model, test_loader, criterion, epoch, num_epochs
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_precision'].append(test_metrics['precision'])
        history['test_recall'].append(test_metrics['recall'])
        history['test_f1'].append(test_metrics['f1'])
        history['test_roc_auc'].append(test_metrics['roc_auc'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{num_epochs} - Time: {epoch_time:.1f}s")
        print(f"{'='*70}")
        print(f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, "
              f"F1={train_metrics['f1']:.4f}")
        print(f"Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}, "
              f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, AUC={test_metrics['roc_auc']:.4f}")
        
        # Save epoch results
        save_epoch_results(epoch, train_metrics, test_metrics, log_file)
        
        # Save best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_epoch = epoch
            # Handle DataParallel: save only the module's state_dict
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_f1': best_f1,
                'test_metrics': test_metrics,
                'history': history
            }, 'checkpoints/best_model.pth')
            print(f"✓ Best model saved! F1: {best_f1:.4f}")
        
        # Save confusion matrix every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            cm_path = f'results/cm_epoch_{epoch+1}.png'
            plot_confusion_matrix(test_labels, test_preds, epoch, cm_path)
            
            # Save checkpoint
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
            print(f"✓ Checkpoint saved at epoch {epoch+1}")
        
        print(f"Best F1: {best_f1:.4f} (Epoch {best_epoch+1})")
        print(f"{'='*70}\n")
    
    # Final evaluation with best model
    print("\n" + "="*70)
    print("FINAL EVALUATION WITH BEST MODEL")
    print("="*70 + "\n")
    
    checkpoint = torch.load('checkpoints/best_model.pth')
    # Handle DataParallel: load into module if wrapped
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics, (test_labels, test_preds, test_probs) = evaluate(
        model, test_loader, criterion, best_epoch, num_epochs
    )
    
    # Print final results
    print("\nFINAL TEST RESULTS:")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1-Score:  {final_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {final_metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Not Synthesizable', 'Synthesizable']))
    
    # Save final results
    plot_confusion_matrix(test_labels, test_preds, best_epoch, 'results/final_cm.png')
    
    final_results = {
        'best_epoch': best_epoch + 1,
        'final_metrics': final_metrics,
        'confusion_matrix': cm.tolist(),
        'history': history
    }
    
    with open('results/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best model saved at epoch {best_epoch+1}")
    print(f"Results saved in 'results/' directory")
    print(f"Checkpoints saved in 'checkpoints/' directory")
    print(f"Training log saved in '{log_file}'")
    print("="*70 + "\n")
    
    return final_results


def plot_training_history(history):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Swin Transformer Training History (Balanced Data)', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=2)
    axes[0, 0].plot(epochs, history['test_loss'], label='Test Loss', marker='o', markersize=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', markersize=2)
    axes[0, 1].plot(epochs, history['test_acc'], label='Test Accuracy', marker='o', markersize=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], label='Train F1', marker='o', markersize=2)
    axes[1, 0].plot(epochs, history['test_f1'], label='Test F1', marker='o', markersize=2)
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision, Recall, ROC-AUC
    axes[1, 1].plot(epochs, history['test_precision'], label='Precision', marker='o', markersize=2)
    axes[1, 1].plot(epochs, history['test_recall'], label='Recall', marker='o', markersize=2)
    axes[1, 1].plot(epochs, history['test_roc_auc'], label='ROC-AUC', marker='o', markersize=2)
    axes[1, 1].set_title('Test Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    train_model()

