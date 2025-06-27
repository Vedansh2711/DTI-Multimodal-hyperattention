import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class MultimodalHyperAttentionDTI(nn.Module):
    def __init__(self, config):
        super(MultimodalHyperAttentionDTI, self).__init__()
        self.dim = config.char_dim
        self.conv = config.conv
        self.drug_MAX_LENGTH = config.drug_max_length
        self.drug_kernel = config.drug_kernel
        self.protein_MAX_LENGTH = config.protein_max_length
        self.protein_kernel = config.protein_kernel
        self.image_size = config.image_size
        
        # Embeddings for protein and drug SMILES
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        
        # CNN for drug SMILES processing
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv*2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv*4, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        
        # Max pooling for drug SMILES features
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGTH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        
        # CNN for protein sequence processing
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv*2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv*4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        
        # Max pooling for protein features
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGTH-self.protein_kernel[0]-self.protein_kernel[1]-self.protein_kernel[2]+3)
        
        # Image processing pathway
        # Using a pre-trained ResNet model with modified final layers
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last FC layer
        self.image_encoder = nn.Sequential(*modules)
        
        # Projection layer to match dimensions with SMILES features
        self.image_projection = nn.Linear(512, self.conv*4)
        
        # Fusion layer for combining SMILES and image features
        # Fix: Change the input dimension to match the actual input size
        self.fusion_layer = nn.Linear(self.conv*4, self.conv*4)
        
        # Attention mechanisms
        self.attention_layer = nn.Linear(self.conv*4, self.conv*4)
        self.protein_attention_layer = nn.Linear(self.conv*4, self.conv*4)
        self.drug_attention_layer = nn.Linear(self.conv*4, self.conv*4)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        
        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image):
        """Process a drug image and extract features"""
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Extract features using the image encoder
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = image_features.view(image_features.size(0), -1)
            
        # Project to match dimension with SMILES features
        image_features = self.image_projection(image_features)
        
        return image_features

    def forward(self, drug, protein, drug_image):
        # Process SMILES data
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)
        
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        
        # Process image data
        image_features = self.process_image(drug_image)
        
        # Reshape image features to match drugConv dimensions
        batch_size = drugConv.shape[0]
        image_features = image_features.view(batch_size, self.conv*4, 1)
        
        # Concatenate SMILES and image features along the sequence dimension
        # This creates a multimodal drug representation
        combined_drug_features = torch.cat([drugConv, image_features], dim=2)
        
        # Fix: Modify the reshaping and fusion operation
        # First, permute to get [batch_size, seq_len, features]
        combined_drug_features_reshaped = combined_drug_features.permute(0, 2, 1)
        
        # Apply fusion to each element in the sequence
        # Process each element independently
        batch_size, seq_len, feature_dim = combined_drug_features_reshaped.shape
        combined_drug_features_flat = combined_drug_features_reshaped.reshape(-1, feature_dim)
        fused_drug_features_flat = self.fusion_layer(combined_drug_features_flat)
        fused_drug_features_reshaped = fused_drug_features_flat.reshape(batch_size, seq_len, feature_dim)
        
        # Permute back to [batch_size, features, seq_len]
        fused_drug_features = fused_drug_features_reshaped.permute(0, 2, 1)
        
        # Apply attention mechanism
        drug_att = self.drug_attention_layer(fused_drug_features.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))
        
        # Create attention matrices
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, proteinConv.shape[-1], 1)
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, fused_drug_features.shape[-1], 1, 1)
        
        # Calculate attention matrix
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        
        # Calculate attention weights
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))
        
        # Apply attention weights
        fused_drug_features = fused_drug_features * 0.5 + fused_drug_features * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte
        
        # Max pooling
        drugConv = self.Drug_max_pool(fused_drug_features).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        
        # Concatenate drug and protein features
        pair = torch.cat([drugConv, proteinConv], dim=1)
        
        # Fully connected layers for prediction
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        
        return predict
