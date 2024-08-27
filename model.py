import torch
import torch.utils.data
torch.manual_seed(500)
from torch import Tensor, nn
from torch_geometric.nn import to_hetero, SAGEConv
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import List, Tuple
from torch_geometric.data import HeteroData
from datetime import datetime
from data import get_sageconv_movie_data_and_loaders

class EdgeSAGEConv(SAGEConv):
    def __init__(self, *args, edge_dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_lin = nn.Linear(edge_dim, self.out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            self._edge_attr = edge_attr
        out = super().forward(x, edge_index)
        self._edge_attr = None
        return out

    def message(self, x_j):
        if self._edge_attr is not None:
            return x_j + self.edge_lin(self._edge_attr)
        return x_j

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = EdgeSAGEConv(in_channels, hidden_channels, edge_dim=edge_dim)
            self.convs.append(conv)
            in_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
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
        self.gnn = GNN(config['hidden_dim'], config['hidden_dim'], config['num_layers'], edge_dim=config['edge_dim'])
        self.gnn = to_hetero(self.gnn, metadata=data.metadata(), aggr='sum')
        
        self.classifier = Classifier(config['hidden_dim'])
    
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_emb(data['user'].node_id),
            "movie": self.movie_lin(data['movie'].x) + self.movie_emb(data['movie'].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)
        
        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )

def generate_simple_filename(auc, small=False):
    """
    Generate a simple filename for saving the model.
    
    Args:
    auc (float): The AUC score of the model.
    
    Returns:
    str: A formatted filename string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    auc_str = f"{auc:.4f}".replace('.', '')
    filename = f"model_{timestamp}_auc{auc_str}_small.pt" if small else f"model_{timestamp}_auc{auc_str}.pt"
    
    return filename

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
             device: torch.device):
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
                device_name: str = "cuda" if torch.cuda.is_available() else "cpu", 
                save_path: str = "./model",
                small: bool = False) -> Tuple[List[float], List[float]]:
    print(f'Device: {device_name}')
    device = torch.device(device_name)
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
            filename = generate_simple_filename(val_auc, small)
            torch.save(model.state_dict(), f"{save_path}/{filename}")

    
    return train_losses, val_aucs

config = {
    'movie_feature_dim': 24,
    'hidden_dim': 64,
    'num_layers': 2,
    'edge_dim': 1,
}

small_data = True

# Usage
num_epochs = 30
learning_rate = 0.001
data, train_loader, val_loader = get_sageconv_movie_data_and_loaders(small_data)

model = RecommendationModel(config, data)
train_losses, val_aucs = train_model(model, train_loader, val_loader, 
                                     num_epochs=num_epochs, 
                                     lr=learning_rate,
                                     small=small_data)
