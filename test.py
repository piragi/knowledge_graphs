import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from data import get_sageconv_movie_data_and_loaders

class EdgeSAGEConv(SAGEConv):
    def __init__(self, *args, edge_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_lin = Linear(edge_dim, self.out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        self._edge_attr = edge_attr
        out = super().forward(x, edge_index)
        self._edge_attr = None
        return out
    
    def message(self, x_j):
        return (x_j + self.edge_lin(self._edge_attr))

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim):
        super().__init__()
        self.conv1 = EdgeSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = EdgeSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_label))
        x = self.conv2(x, edge_index, edge_label)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.lin = torch.nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        edge_feat = torch.cat([edge_feat_user, edge_feat_movie], dim=-1)
        return self.lin(edge_feat)

class Model(torch.nn.Module):
    def __init__(self, config, num_users, num_movies, num_movie_features, metadata):
        super().__init__()
        hidden_channels = config['hidden_channels']
        self.movie_lin = torch.nn.Linear(num_movie_features, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.gnn = GNN(hidden_channels, config['edge_dim'])
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = Classifier(hidden_channels, config['num_classes'])

    def forward(self, data) -> Tensor:
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        
        edge_label_dict = {
            ('user', 'rates', 'movie'): data['user', 'rates', 'movie'].edge_attr,
            ('movie', 'rev_rates', 'user'): data['movie', 'rev_rates', 'user'].edge_attr
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict, edge_label_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred

def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = total_examples = 0
    for data in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        data = data.to(device)
        pred = model(data)
        loss = F.cross_entropy(pred, data["user", "rates", "movie"].edge_label.long())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)
        total_examples += pred.size(0)
    return total_loss / total_examples

def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, truths = [], []
    for data in tqdm(loader, desc="Validating", leave=False):
        with torch.no_grad():
            data = data.to(device)
            logits = model(data)
            preds.append(logits.argmax(dim=-1))
            truths.append(data["user", "rates", "movie"].edge_label)
    preds = torch.cat(preds, dim=0).cpu()
    truths = torch.cat(truths, dim=0).cpu()
    return (preds == truths).float().mean().item()

def train_and_validate(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                       optimizer: torch.optim.Optimizer, device: torch.device, epochs: int) -> None:
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        accuracy = validate(model, val_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

def test(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    accuracy = validate(model, loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

config = {
    'hidden_channels': 512,
    'lr': 0.0001,
    'epochs': 25,
    'edge_dim': 7,
    'num_classes': 7,
    'small_dataset': True,  # Added this for the dataset size parameter
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, train_loader, val_loader, test_loader = get_sageconv_movie_data_and_loaders(small=config['small_dataset'])
    print(f"Using device: {device}")

    model = Model(
        config,
        num_users=train_data["user"].num_nodes,
        num_movies=train_data["movie"].num_nodes,
        num_movie_features=train_data["movie"].num_features,
        metadata=train_data.metadata(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_and_validate(model, train_loader, val_loader, optimizer, device, config['epochs'])
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
