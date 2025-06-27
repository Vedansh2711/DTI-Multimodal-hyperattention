import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms

class DrugProteinDataset(Dataset):
    def __init__(self, data_file, config, mode='train', transform=None):
        """
        Dataset for drug-protein interaction prediction
        
        Args:
            data_file: Path to the data file (Davis.txt)
            config: Configuration parameters
            mode: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.config = config
        self.mode = mode
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load data from file
        self.data = self._load_data(data_file)
        
        # Create directories for storing images
        self.image_dir = os.path.join(config.data_root, f"drug_images_{config.image_size}")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        # Generate or load images for drugs
        self._process_drug_images()
        
    def _load_data(self, data_file):
        """Load data from file and parse it"""
        data_list = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    drug_id = parts[0]
                    protein_id = parts[1]
                    smiles = parts[2]
                    protein_seq = parts[3] if len(parts) > 3 else ""
                    label = int(parts[-1])
                    
                    data_list.append({
                        'drug_id': drug_id,
                        'protein_id': protein_id,
                        'smiles': smiles,
                        'protein_seq': protein_seq,
                        'label': label
                    })
        return data_list
    
    def _process_drug_images(self):
        """Generate or load images for all drugs in the dataset"""
        for i, item in enumerate(self.data):
            smiles = item['smiles']
            drug_id = item['drug_id']
            image_path = os.path.join(self.image_dir, f"{drug_id}.png")
            
            # Generate image if it doesn't exist
            if not os.path.exists(image_path):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        img = Draw.MolToImage(mol, size=(self.config.image_size, self.config.image_size), wedgeBonds=False)
                        img.save(image_path)
                        item['image_path'] = image_path
                    else:
                        print(f"Warning: Could not parse SMILES {smiles}")
                        item['image_path'] = None
                except Exception as e:
                    print(f"Error processing SMILES {smiles}: {e}")
                    item['image_path'] = None
            else:
                item['image_path'] = image_path
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process SMILES
        smiles = item['smiles']
        smiles_encoded = self._encode_smiles(smiles)
        
        # Process protein sequence
        protein_seq = item['protein_seq']
        protein_encoded = self._encode_protein(protein_seq)
        
        # Load and process image
        image_path = item['image_path']
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            # Create a blank image if the actual image is not available
            image = torch.zeros((3, 224, 224))
        
        # Get label
        label = item['label']
        
        return {
            'drug': torch.tensor(smiles_encoded, dtype=torch.long),
            'protein': torch.tensor(protein_encoded, dtype=torch.long),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _encode_smiles(self, smiles):
        """Encode SMILES string to integers"""
        # Define the vocabulary for SMILES
        vocab = {x: i+1 for i, x in enumerate("^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTUVXYZ[\\]abcdefgilmnoprstuy$")}
        
        # Pad or truncate to fixed length
        encoded = [0] * self.config.drug_max_length
        for i, char in enumerate(smiles[:self.config.drug_max_length]):
            if char in vocab:
                encoded[i] = vocab[char]
        
        return encoded
    
    def _encode_protein(self, sequence):
        """Encode protein sequence to integers"""
        # Define the vocabulary for protein sequences
        vocab = {x: i+1 for i, x in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")}
        
        # Pad or truncate to fixed length
        encoded = [0] * self.config.protein_max_length
        for i, char in enumerate(sequence[:self.config.protein_max_length]):
            if char in vocab:
                encoded[i] = vocab[char]
        
        return encoded

def get_data_loaders(config):
    """Create data loaders for train, validation, and test sets"""
    # Split the data file into train, validation, and test sets
    data_file = os.path.join(config.data_root, "Davis.txt")
    
    # Create datasets
    train_dataset = DrugProteinDataset(data_file, config, mode='train')
    
    # Split the dataset
    total_size = len(train_dataset)
    val_size = int(total_size * 0.1)
    test_size = int(total_size * 0.1)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    drug_batch = torch.stack([item['drug'] for item in batch])
    protein_batch = torch.stack([item['protein'] for item in batch])
    image_batch = torch.stack([item['image'] for item in batch])
    label_batch = torch.stack([item['label'] for item in batch])
    
    return {
        'drug': drug_batch,
        'protein': protein_batch,
        'image': image_batch,
        'label': label_batch
    }
