import torch
torch.manual_seed(500)
from torch import Tensor, nn
import pandas as pd
import numpy as np
from torch_geometric.nn import to_hetero, SAGEConv
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import List, Tuple
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from datetime import datetime
from data import get_movie_data_and_loaders

class GNN(nn.Module):
    """Graph Neural Network using SAGEConv layers."""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        return x

class Classifier(nn.Module):
    """Classifier for link prediction."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, user_features: Tensor, movie_features: Tensor, edge_index: Tensor) -> Tensor:
        edge_features = torch.cat([user_features[edge_index[0]], movie_features[edge_index[1]]], dim=1)
        return self.layers(edge_features).view(-1)

class RecommendationModel(nn.Module):
    """Main model for movie recommendations."""
    def __init__(self, config: dict, data: HeteroData):
        super().__init__()
        self.movie_lin = nn.Linear(config['movie_feature_dim'], config['hidden_dim'])
        self.user_emb = nn.Embedding(data['user'].num_nodes, config['hidden_dim'])
        self.movie_emb = nn.Embedding(data['movie'].num_nodes, config['hidden_dim'])
        
        self.gnn = GNN(config['hidden_dim'], config['hidden_dim'], config['num_layers'])
        self.gnn = to_hetero(self.gnn, metadata=data.metadata(), aggr='sum')
        
        self.classifier = Classifier(config['hidden_dim'])
    
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_emb(data['user'].node_id),
            "movie": self.movie_lin(data['movie'].x) + self.movie_emb(data['movie'].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )

def generate_simple_filename(auc: float) -> str:
    """
    Generate a simple filename for saving the model.
    
    Args:
    auc (float): The AUC score of the model.
    
    Returns:
    str: A formatted filename string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    auc_str = f"{auc:.4f}".replace('.', '')
    
    return f"model_{timestamp}_auc{auc_str}.pt"



def train_epoch(model: torch.nn.Module, 
                loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device) -> float:
    model.train()
    total_loss = 0
    total_examples = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        label = batch['user', 'rates', 'movie'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.numel()
        total_examples += label.numel()
    
    return total_loss / total_examples

def validate(model: torch.nn.Module, 
             loader: torch.utils.data.DataLoader, 
             device: torch.device) -> Tuple[float, List[float], List[float]]:
    model.eval()
    preds = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            pred = model(batch)
            preds.append(pred)
            ground_truths.append(batch["user", "rates", "movie"].edge_label)
    
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    
    return auc, pred.tolist(), ground_truth.tolist()

def train_model(model: torch.nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                num_epochs: int = 150, 
                lr: float = 0.001, 
                device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                save_path: str = "./model") -> Tuple[List[float], List[float]]:
    print(f'Device: {device}')
    device = torch.device(device)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_aucs = []
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_auc, _, _ = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_aucs.append(val_auc)
        
        print(f'Epoch {epoch + 1:03d}, Loss: {train_loss:.4f}, Validation AUC: {val_auc:.4f}')
        
        if val_auc == max(val_aucs):
            filename = generate_simple_filename(val_auc)
            torch.save(model.state_dict(), f"{save_path}/{filename}")

    
    return train_losses, val_aucs

config = {
    'movie_feature_dim': 24,
    'hidden_dim': 64,
    'num_layers': 2
}

# Usage
num_epochs = 5
learning_rate = 0.001
data, train_loader, val_loader = get_movie_data_and_loaders()

model = RecommendationModel(config, data)
train_losses, val_aucs = train_model(model, train_loader, val_loader, 
                                     num_epochs=num_epochs, 
                                     lr=learning_rate)