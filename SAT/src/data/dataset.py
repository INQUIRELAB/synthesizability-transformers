import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class StructureAwareFTCPDataset(Dataset):
    """
    Dataset class for FTCP data that respects its original structure.
    
    This implementation ensures:
    1. No arbitrary reshaping of the data
    2. Proper train/test splitting with balanced test data
    3. Structure-aware handling of the FTCP tensor
    """
    def __init__(self, ftcp_file, labels_file, transform=None, train=False, test=False):
        """
        Args:
            ftcp_file (str): Path to the HDF5 file containing FTCP data
            labels_file (str): Path to the Excel file containing synthesizability labels
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): If True, creates training dataset
            test (bool): If True, creates test dataset (balanced)
        """
        self.transform = transform
        
        if not (train ^ test):  # XOR - exactly one must be True
            raise ValueError("Exactly one of train or test must be True")
        
        # Load FTCP data from the HDF5 file
        with h5py.File(ftcp_file, 'r') as f:
            # Get all batch keys
            batch_keys = sorted(list(f.keys()))
            
            # Initialize lists to store data
            ftcp_data_list = []
            material_ids_list = []
            
            # Process each batch
            for batch_key in batch_keys:
                batch_group = f[batch_key]
                
                # Use FTCP_normalized as the feature data
                if 'FTCP_normalized' in batch_group:
                    ftcp_data_list.append(batch_group['FTCP_normalized'][:])
                elif 'FTCP_padded' in batch_group:
                    ftcp_data_list.append(batch_group['FTCP_padded'][:])
                elif 'FTCP_original' in batch_group:
                    ftcp_data_list.append(batch_group['FTCP_original'][:])
                
                # Use cif_names as the material_ids
                if 'cif_names' in batch_group:
                    material_ids = batch_group['cif_names'][:]
                    # Convert byte strings to regular strings if needed
                    if isinstance(material_ids[0], bytes):
                        material_ids = [mid.decode('utf-8') for mid in material_ids]
                    # Remove .cif extension from material IDs
                    material_ids = [mid.replace('.cif', '') for mid in material_ids]
                    material_ids_list.append(material_ids)
            
            # Combine all batches
            try:
                self.ftcp_data = np.vstack(ftcp_data_list)
                self.material_ids = np.concatenate(material_ids_list)
            except ValueError:
                # If shape mismatch, handle it
                print("Warning: Data shape mismatch across batches. Using only first batch.")
                self.ftcp_data = ftcp_data_list[0]
                self.material_ids = material_ids_list[0]
        
        # Load labels from Excel file
        df = pd.read_excel(labels_file)
        
        # Create mapping from material ID to label
        material_id_to_label = dict(zip(df['material_id'].astype(str), df['synthesizable'].astype(int)))
        
        # Match with our data
        matched_indices = []
        matched_labels = []
        matched_ids = []
        
        for i, mid in enumerate(self.material_ids):
            if mid in material_id_to_label:
                matched_indices.append(i)
                matched_labels.append(material_id_to_label[mid])
                matched_ids.append(mid)
        
        if len(matched_indices) == 0:
            raise ValueError("No material IDs matched between data and labels!")
        
        print(f"Found {len(matched_indices)} matching material IDs out of {len(self.material_ids)}")
        
        # Keep only matched data
        matched_ftcp_data = self.ftcp_data[matched_indices]
        matched_labels = np.array(matched_labels)
        
        # Create a DataFrame for easier handling
        data_df = pd.DataFrame({
            'index': range(len(matched_indices)),
            'material_id': matched_ids,
            'label': matched_labels
        })
        
        # Get indices for positive and negative samples
        pos_indices = data_df[data_df['label'] == 1]['index'].values
        neg_indices = data_df[data_df['label'] == 0]['index'].values
        
        print(f"Total positive samples: {len(pos_indices)}")
        print(f"Total negative samples: {len(neg_indices)}")
        
        # Create balanced test set with 20% of the data, 50/50 class split
        test_size_per_class = min(len(pos_indices), len(neg_indices)) // 5  # 20% of the smaller class
        test_size = test_size_per_class * 2  # Equal number from each class
        
        # Randomly sample from each class
        np.random.seed(42)  # For reproducibility
        test_pos_indices = np.random.choice(pos_indices, test_size_per_class, replace=False)
        test_neg_indices = np.random.choice(neg_indices, test_size_per_class, replace=False)
        
        # Combine for the test set
        test_indices = np.concatenate([test_pos_indices, test_neg_indices])
        
        # Create train set with everything else
        train_indices = np.array([i for i in range(len(matched_indices)) 
                                  if i not in test_indices])
        
        # Select appropriate data split
        if train:
            self.indices = train_indices
        else:  # test
            self.indices = test_indices
            
        # Update data and labels
        self.ftcp_data = matched_ftcp_data[self.indices]
        self.labels = matched_labels[self.indices]
        
        # Print dataset statistics
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"{'Train' if train else 'Test'} dataset label distribution:")
        for class_id, count in zip(unique, counts):
            print(f"Class {class_id}: {count} samples ({count/len(self.labels)*100:.2f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the FTCP tensor and label
        ftcp = self.ftcp_data[idx]
        label = self.labels[idx]
        
        # The FTCP data has shape (600, 63)
        # Based on the provided information, the rows are structured as:
        # Element One-Hot Encoding (rows 0-92)
        # Lattice Parameters (rows 93-94)
        # Site Coordinates (rows 95-294, padded to 200 sites)
        # Site Occupancy (rows 295-494, padded to 200 sites)
        # Padding Zero Row (row 495)
        # Element Properties (rows 496-587, 92 rows)
        # Additional Padding (rows 588-599, 12 rows if any)
        
        # Combine into a dictionary to preserve structure
        ftcp_dict = {
            'element_encoding': torch.FloatTensor(ftcp[0:93]),        # Element One-Hot Encoding
            'lattice_params': torch.FloatTensor(ftcp[93:95]),         # Lattice Parameters
            'site_coords': torch.FloatTensor(ftcp[95:295]),           # Site Coordinates
            'site_occupancy': torch.FloatTensor(ftcp[295:495]),       # Site Occupancy
            'padding_row': torch.FloatTensor(ftcp[495:496]),          # Padding Zero Row
            'element_props': torch.FloatTensor(ftcp[496:588]),        # Element Properties
            'additional_padding': torch.FloatTensor(ftcp[588:600]),   # Additional Padding
            'full_tensor': torch.FloatTensor(ftcp)                    # Full tensor (600, 63)
        }
        
        # Convert label to tensor
        label = torch.LongTensor([label])[0]
        
        # Apply transform if specified
        if self.transform:
            ftcp_dict = self.transform(ftcp_dict)
            
        return ftcp_dict, label

def get_dataloaders(config):
    """
    Create train and test dataloaders.
    
    Args:
        config (dict): Configuration dictionary containing data parameters
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = StructureAwareFTCPDataset(
        ftcp_file=config['data']['ftcp_file'],
        labels_file=config['data']['labels_file'],
        transform=None,  # Add transforms if needed
        train=True
    )
    
    test_dataset = StructureAwareFTCPDataset(
        ftcp_file=config['data']['ftcp_file'],
        labels_file=config['data']['labels_file'],
        transform=None,
        test=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, test_loader 