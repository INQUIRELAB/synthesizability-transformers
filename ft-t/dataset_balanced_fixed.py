"""
Efficient Balanced Dataset - Loads data ONCE and shares between train/test
"""
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from save_data_split import save_split_indices


# Global cache for loaded data
_DATA_CACHE = {}


def load_ftcp_data_once(ftcp_file, labels_file):
    """Load FTCP data and labels once, return indices for train/test split."""
    
    cache_key = (ftcp_file, labels_file)
    if cache_key in _DATA_CACHE:
        print("Using cached data...")
        return _DATA_CACHE[cache_key]
    
    print(f"Loading data from: {ftcp_file} (will be cached)")
    
    # Load FTCP data
    with h5py.File(ftcp_file, 'r') as f:
        batch_keys = sorted(list(f.keys()))
        print(f"Found {len(batch_keys)} batches")
        
        all_material_ids = []
        ftcp_data_list = []
        
        for batch_key in batch_keys:
            batch_group = f[batch_key]
            
            # Get material IDs
            if 'cif_names' in batch_group:
                material_ids = batch_group['cif_names'][:]
                if isinstance(material_ids[0], bytes):
                    material_ids = [mid.decode('utf-8') for mid in material_ids]
                material_ids = [mid.replace('.cif', '') for mid in material_ids]
                all_material_ids.extend(material_ids)
            
            # Load FTCP data
            if 'FTCP_normalized' in batch_group:
                data = batch_group['FTCP_normalized'][:].astype(np.float32)
                ftcp_data_list.append(data)
            elif 'FTCP_padded' in batch_group:
                data = batch_group['FTCP_padded'][:].astype(np.float32)
                ftcp_data_list.append(data)
            elif 'FTCP_original' in batch_group:
                data = batch_group['FTCP_original'][:].astype(np.float32)
                ftcp_data_list.append(data)
        
        # Combine all batches
        ftcp_data = np.vstack(ftcp_data_list)
        material_ids = np.array(all_material_ids)
        print(f"Loaded FTCP data shape: {ftcp_data.shape}")
        print(f"Total samples: {len(material_ids)}")
    
    # Load labels
    print("Loading labels...")
    df = pd.read_excel(labels_file)
    material_id_to_label = dict(zip(df['material_id'].astype(str), df['synthesizable'].astype(int)))
    
    # Match with our data
    matched_indices = []
    matched_labels = []
    matched_ids = []
    
    for i, mid in enumerate(material_ids):
        if mid in material_id_to_label:
            matched_indices.append(i)
            matched_labels.append(material_id_to_label[mid])
            matched_ids.append(mid)
    
    if len(matched_indices) == 0:
        raise ValueError("No material IDs matched between data and labels!")
    
    print(f"Matched {len(matched_indices)} materials")
    
    # Keep only matched data
    ftcp_data = ftcp_data[matched_indices]
    labels = np.array(matched_labels)
    matched_ids = np.array(matched_ids)
    
    # Create balanced test set
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    print(f"Total positive samples: {len(pos_indices)}")
    print(f"Total negative samples: {len(neg_indices)}")
    
    # Balanced test set: 20% of smaller class, 50/50 split
    test_size_per_class = min(len(pos_indices), len(neg_indices)) // 5
    
    np.random.seed(42)
    test_pos_indices = np.random.choice(pos_indices, test_size_per_class, replace=False)
    test_neg_indices = np.random.choice(neg_indices, test_size_per_class, replace=False)
    
    test_indices = np.concatenate([test_pos_indices, test_neg_indices])
    test_indices_set = set(test_indices)
    train_indices = np.array([i for i in range(len(labels)) if i not in test_indices_set])
    
    # Save split info
    save_split_indices(
        train_indices, test_indices,
        labels[train_indices], labels[test_indices],
        matched_ids[train_indices], matched_ids[test_indices],
        output_file='data_split_info.json'
    )
    
    # Cache everything
    result = {
        'ftcp_data': ftcp_data,
        'labels': labels,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'material_ids': matched_ids
    }
    
    _DATA_CACHE[cache_key] = result
    return result


class BalancedFTCPDataset(Dataset):
    """Dataset that shares loaded data between train and test."""
    
    def __init__(self, ftcp_data, labels, indices, split_name):
        self.ftcp_data = ftcp_data
        self.labels = labels
        self.indices = indices
        self.split_name = split_name
        
        # Print statistics
        split_labels = labels[indices]
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} DATASET STATISTICS:")
        print(f"{'='*60}")
        for class_id, count in zip(unique, counts):
            percentage = count/len(split_labels)*100
            class_name = 'Synthesizable' if class_id == 1 else 'Not Synthesizable'
            print(f"Class {class_id} ({class_name}): {count} samples ({percentage:.2f}%)")
        print(f"Total samples: {len(split_labels)}")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get actual data index
        data_idx = self.indices[idx]
        
        # Get data and label (already in numpy)
        ftcp = self.ftcp_data[data_idx]
        label = self.labels[data_idx]
        
        # Convert to tensors
        ftcp = torch.from_numpy(ftcp)  # Already float32
        label = torch.tensor(label, dtype=torch.long)
        
        return ftcp, label


def get_balanced_dataloaders(ftcp_file, labels_file, batch_size=32, num_workers=0):
    """
    Create train and test dataloaders with shared data (loaded once).
    
    Args:
        ftcp_file (str): Path to FTCP HDF5 file
        labels_file (str): Path to labels Excel file
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    print("Creating balanced datasets (single load, shared data)...")
    
    # Load data ONCE
    data_dict = load_ftcp_data_once(ftcp_file, labels_file)
    
    # Create train dataset (references shared data)
    train_dataset = BalancedFTCPDataset(
        ftcp_data=data_dict['ftcp_data'],
        labels=data_dict['labels'],
        indices=data_dict['train_indices'],
        split_name='train'
    )
    
    # Create test dataset (references same shared data)
    test_dataset = BalancedFTCPDataset(
        ftcp_data=data_dict['ftcp_data'],
        labels=data_dict['labels'],
        indices=data_dict['test_indices'],
        split_name='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

