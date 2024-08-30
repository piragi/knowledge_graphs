import torch
from torch import Tensor
from tqdm import tqdm
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from data import get_sageconv_movie_data_and_loaders
from sklearn.metrics import  roc_auc_score
import sys


class EdgeSAGEConv(SAGEConv):
    def _init__(self, *args, edge_dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_lin = Linear(edge_dim, self.out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        self._edge_attr = edge_attr
        out = super().forward(x, edge_index)
        self._edge_attr = None
        return out

    def message(self, x_j, edge_weight):
        return (x_j + self.edge_lin(self._edge_attr)) * edge_weight.view(-1, 1)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes=7):
        super().__init__()
        self.lin = torch.nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Concatenate user and movie features
        edge_feat = torch.cat([edge_feat_user, edge_feat_movie], dim=-1)
        # Apply linear layer to get logits for each class
        return self.lin(edge_feat)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_users, num_movies, num_movie_features, metadata):
        super().__init__()
        self.movie_lin = torch.nn.Linear(num_movie_features, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.movie_emb = torch.nn.Embedding(num_movies, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = Classifier(hidden_channels)

    def forward(self, data) -> Tensor:
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_examples = 0
    for data in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        data = data.to(device)
        pred = model(data)
        ground_truth = data["user", "rates", "movie"].edge_label
        loss = F.cross_entropy(pred, ground_truth.long())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)
        total_examples += pred.size(0)
    return total_loss / total_examples

def validate(model, loader, device):
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

def train_and_validate(model, train_loader, val_loader, optimizer, device, epochs=5):
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        accuracy = validate(model, val_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

def test(model, loader, device):
    model.eval()
    preds, truths = [], []
    for data in tqdm(loader, desc="Testing", leave=False):
        with torch.no_grad():
            data = data.to(device)
            logits = model(data)
            preds.append(logits.argmax(dim=-1))
            truths.append(data["user", "rates", "movie"].edge_label)
    preds = torch.cat(preds, dim=0).cpu()
    truths = torch.cat(truths, dim=0).cpu()
    accuracy = (preds == truths).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def main(hidden_channels=512, lr=0.001, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, train_loader, val_loader, test_loader = get_sageconv_movie_data_and_loaders(small=True)
    print(f"Using device: {device}")

    model = Model(hidden_channels, train_data["user"].num_nodes, 
                  train_data["movie"].num_nodes, train_data["movie"].num_features, train_data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_and_validate(model, train_loader, val_loader, optimizer, device, epochs)
    
    # Evaluate on the test set
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
