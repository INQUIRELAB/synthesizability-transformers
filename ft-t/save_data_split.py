"""
Save the train/test split indices for reproducibility and comparison with Alternative Model
"""
import numpy as np
import pandas as pd
import json

def save_split_indices(train_indices, test_indices, train_labels, test_labels, 
                       train_ids, test_ids, output_file='data_split_info.json'):
    """
    Save the data split information for reproducibility.
    
    Args:
        train_indices: Array of training indices
        test_indices: Array of test indices
        train_labels: Training labels
        test_labels: Test labels
        train_ids: Training material IDs
        test_ids: Test material IDs
        output_file: Output JSON file path
    """
    # Calculate statistics
    train_pos = (train_labels == 1).sum()
    train_neg = (train_labels == 0).sum()
    test_pos = (test_labels == 1).sum()
    test_neg = (test_labels == 0).sum()
    
    split_info = {
        'split_method': 'balanced_stratified_random',
        'random_seed': 42,
        'test_ratio': 0.2,
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'train_distribution': {
            'positive': int(train_pos),
            'negative': int(train_neg),
            'ratio': float(train_pos / train_neg) if train_neg > 0 else 0
        },
        'test_distribution': {
            'positive': int(test_pos),
            'negative': int(test_neg),
            'ratio': float(test_pos / test_neg) if test_neg > 0 else 0,
            'balanced': bool(test_pos == test_neg)
        },
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'sample_train_ids': train_ids[:10].tolist() if len(train_ids) > 10 else train_ids.tolist(),
        'sample_test_ids': test_ids[:10].tolist() if len(test_ids) > 10 else test_ids.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Data split information saved to: {output_file}")
    print(f"  Train: {len(train_indices)} samples (Pos: {train_pos}, Neg: {train_neg})")
    print(f"  Test:  {len(test_indices)} samples (Pos: {test_pos}, Neg: {test_neg})")
    print(f"  Test is balanced: {test_pos == test_neg}")
    
    return split_info

