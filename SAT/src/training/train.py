import os
import wandb
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from src.data.dataset import get_dataloaders
from src.models.structure_transformer import FTCPTransformer

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance."""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device if it's a tensor
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        # If it's a dictionary, move each tensor to device
        elif isinstance(data, dict):
            for k in data:
                data[k] = data[k].to(device)
                
        target = target.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=config['training']['mixed_precision']):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        
        # Apply gradient clipping for stability
        if config['training'].get('gradient_clip_norm', 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler after optimizer
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        
        # Store predictions and labels
        preds = torch.argmax(output, dim=1)
        probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()  # Probability of class 1
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.extend(probs)
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluation')
        for data, target in pbar:
            # Move data to device if it's a tensor
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            # If it's a dictionary, move each tensor to device
            elif isinstance(data, dict):
                for k in data:
                    data[k] = data[k].to(device)
                    
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Store predictions and labels
            preds = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()  # Probability of class 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs)
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics, (all_labels, all_preds, all_probs)

def compute_metrics(labels, preds, probs=None):
    """Compute classification metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Compute ROC-AUC if probabilities are provided
    if probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(labels, probs)
        except:
            metrics['roc_auc'] = 0.5  # Default value if calculation fails
    
    return metrics

def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve and save to file."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_confusion_matrix(labels, preds, save_path):
    """Plot confusion matrix and save to file."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Not Synthesizable', 'Synthesizable']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def train_model(config_path):
    """Train the FTCP model using the given configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory for results
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    
    # Initialize WandB for experiment tracking
    wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        name=config['logging']['wandb']['name'],
        mode=config['logging']['wandb']['mode'],
        config=config
    )
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Create model based on specified mode
    model = FTCPTransformer(
        mode=config['model']['mode'],
        hidden_dim=config['model']['hidden_dim'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    # Calculate class weights to handle imbalance
    if config['training'].get('use_class_weights', True):
        all_labels = []
        for _, target in train_loader:
            all_labels.extend(target.cpu().numpy())
        
        counts = np.bincount(all_labels)
        total = len(all_labels)
        
        # More weight to minority class (inverse frequency)
        class_weights = torch.FloatTensor([total/count for count in counts])
        class_weights = class_weights.to(device)
        
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Set up loss function
    criterion = FocalLoss(
        alpha=config['training'].get('focal_loss_alpha', 1.0),
        gamma=config['training'].get('focal_loss_gamma', 2.0),
        weight=class_weights
    )
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Set up learning rate scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=num_training_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        final_div_factor=10.0
    )
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Training loop
    best_f1 = 0
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, config)
        
        # Evaluate
        test_metrics, (test_labels, test_preds, test_probs) = evaluate(model, test_loader, criterion, device)
        
        # Log metrics
        metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'epoch': epoch
        }
        wandb.log(metrics)
        
        # Every N epochs, create and log visualization artifacts
        if (epoch + 1) % config['logging']['viz_interval'] == 0:
            # Plot ROC curve
            roc_path = os.path.join(config['logging']['output_dir'], f'roc_epoch_{epoch+1}.png')
            plot_roc_curve(test_labels, test_probs, roc_path)
            wandb.log({"roc_curve": wandb.Image(roc_path)})
            
            # Plot confusion matrix
            cm_path = os.path.join(config['logging']['output_dir'], f'cm_epoch_{epoch+1}.png')
            plot_confusion_matrix(test_labels, test_preds, cm_path)
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})
            
            # Log classification report
            report = classification_report(test_labels, test_preds, output_dict=True)
            wandb.log({"classification_report": wandb.Table(
                dataframe=pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'class'})
            )})
        
        # Save best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            
            # Save model
            torch.save(model.state_dict(), os.path.join(config['logging']['output_dir'], 'best_model.pt'))
            print(f"New best model saved with F1 score: {best_f1:.4f}")
            
            # Log best metrics
            wandb.run.summary["best_f1"] = best_f1
            wandb.run.summary["best_accuracy"] = test_metrics['accuracy']
            wandb.run.summary["best_precision"] = test_metrics['precision']
            wandb.run.summary["best_recall"] = test_metrics['recall']
            wandb.run.summary["best_roc_auc"] = test_metrics['roc_auc']
            
        print(f"Train metrics: {train_metrics}")
        print(f"Test metrics: {test_metrics}")
    
    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(config['logging']['output_dir'], 'best_model.pt')))
    final_metrics, (test_labels, test_preds, test_probs) = evaluate(model, test_loader, criterion, device)
    
    print("\nFinal Test Metrics (Best Model):")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Generate final visualizations
    roc_path = os.path.join(config['logging']['output_dir'], 'final_roc.png')
    plot_roc_curve(test_labels, test_probs, roc_path)
    
    cm_path = os.path.join(config['logging']['output_dir'], 'final_cm.png')
    plot_confusion_matrix(test_labels, test_preds, cm_path)
    
    # Log final classification report
    print("\nClassification Report:")
    report_text = classification_report(test_labels, test_preds)
    print(report_text)
    
    # Save report to file
    with open(os.path.join(config['logging']['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    
    # Close wandb run
    wandb.finish()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train FTCP Transformer')
    parser.add_argument('--config', type=str, default='configs/structure_transformer.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    train_model(args.config)

if __name__ == '__main__':
    main() 