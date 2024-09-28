from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, SAGEConv, to_hetero
from tqdm import tqdm

from data import get_sageconv_movie_data_and_loaders


class GAT(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, heads, edge_dim
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )
        self.skips.append(nn.Linear(in_channels, hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    heads * hidden_channels,
                    hidden_channels,
                    heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
            self.skips.append(
                nn.Linear(hidden_channels * heads, hidden_channels * heads)
            )

        self.convs.append(
            GATConv(
                heads * hidden_channels,
                out_channels,
                heads,
                concat=False,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )
        self.skips.append(nn.Linear(hidden_channels * heads, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x = conv(x, edge_index, edge_attr) + skip(x)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class EdgeSAGEConv(SAGEConv):
    def __init__(self, *args, edge_dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_lin = nn.Linear(edge_dim, self.out_channels)
        self._edge_attr = None

    def forward(self, x, edge_index, size=None, edge_attr=None):
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
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = torch.relu(x)
        return x


class Classifier(nn.Module):
    """Classifier for link prediction."""

    def __init__(self, hidden_dim: int, num_classes: int = 7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, user_features: Tensor, movie_features: Tensor, edge_index: Tensor
    ) -> Tensor:
        edge_features = torch.cat(
            [user_features[edge_index[0]], movie_features[edge_index[1]]], dim=1
        )
        return self.layers(edge_features)


class RecommendationModel(nn.Module):
    """Main model for movie recommendations."""

    def __init__(self, config: dict, data_info: dict, model_type: str):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.data_info = data_info

        self.genre_emb = nn.Embedding(
            data_info["num_genre_nodes"], config["hidden_dim"]
        )
        self.user_emb = nn.Embedding(data_info["num_user_nodes"], config["hidden_dim"])
        self.user_lin = nn.Linear(data_info["user_feature_dim"], config["hidden_dim"])

        self.director_emb = nn.Embedding(
            data_info["num_director_nodes"], config["hidden_dim"]
        )
        self.movie_emb = nn.Embedding(
            data_info["num_movie_nodes"], config["hidden_dim"]
        )
        self.movie_lin = nn.Linear(data_info["movie_feature_dim"], config["hidden_dim"])

        if self.model_type == "sage":
            self.conv = GNN(
                config["hidden_dim"],
                config["hidden_dim"],
                config["num_layers"],
                edge_dim=config["edge_dim"],
            )
        elif self.model_type == "gat":
            self.conv = GAT(
                config["hidden_dim"],
                config["hidden_dim"],
                config["hidden_dim"],
                config["num_layers"],
                config["num_heads"],
                config["edge_dim"],
            )
        self.conv = to_hetero(self.conv, data_info["metadata"], aggr="sum")

        self.classifier = Classifier(config["hidden_dim"])

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
            "director": self.director_emb(data["director"].node_id),
            "genre": self.genre_emb(data["genre"].node_id),
            "movie": self.movie_lin(data["movie"].x)
            + self.movie_emb(data["movie"].node_id),
        }
        x_dict = self.conv(x_dict, data.edge_index_dict, data.edge_attr_dict)

        return self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        labels = batch["user", "rates", "movie"].edge_label
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.numel()
    return total_loss / len(loader.dataset)


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            pred = model(batch)
            labels = batch["user", "rates", "movie"].edge_label
            loss = criterion(pred, labels)
            total_loss += loss.item() * labels.numel()
            all_preds.append(F.softmax(pred, dim=1))
            all_labels.append(labels)
    preds = torch.cat(all_preds, dim=0)
    # print_distribution(preds.argmax(dim=1))
    labels = torch.cat(all_labels, dim=0)
    return total_loss / len(loader.dataset), compute_accuracy(labels, preds)


def print_distribution(ratings_tensor: torch.Tensor):
    # Get unique values and their counts
    ratings_tensor = ratings_tensor.cpu()
    unique_values, counts = torch.unique(ratings_tensor, return_counts=True)

    # Calculate percentages
    total_ratings = len(ratings_tensor)
    percentages = (counts / total_ratings * 100).numpy()

    # Sort the results
    sorted_indices = torch.argsort(unique_values)
    unique_values = unique_values[sorted_indices].numpy()
    percentages = percentages[sorted_indices]

    # Print percentage distribution
    print("Percentage distribution of ratings:")
    for rating, percentage in zip(unique_values, percentages):
        print(f"Rating {rating}: {percentage:.2f}%")


def inference(model: nn.Module, input: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in input:
            batch = batch.to(device)
            pred = model(batch)
            probabilities = F.softmax(pred, dim=1)
            all_preds.append(probabilities.argmax(dim=1))
    return all_preds


def compute_accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """Compute accuracy for multi-class classification."""
    return accuracy_score(labels.cpu().numpy(), preds.argmax(dim=1).cpu().numpy())


def save_model(model: nn.Module, accuracy: float, save_path: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_type = model.model_type
    filename = f"model_{model_type}_{timestamp}_acc{accuracy:.4f}.pt"

    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": model.config,  # Assume we add this attribute to the model
        "data_info": model.data_info,
        "model_type": model_type,
        "accuracy": accuracy,
    }

    torch.save(save_dict, f"{save_path}/{filename}")
    print(f"Model saved: {filename}")


def train_gnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 15,
    lr: float = 0.0001,
    save_path: str = "./model",
) -> Tuple[nn.Module, List[float], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = (
        nn.CrossEntropyLoss()
    )  # Changed to CrossEntropyLoss for multi-class classification
    best_accuracy = float("-inf")
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validation
        _, val_accuracy = validate(model, val_loader, criterion, device)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1:03d}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, val_accuracy, save_path)

    # Testing
    _, test_accuracy = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return model, train_losses, val_accuracies


def print_rating_distribution(data):
    ratings = data["user", "rates", "movie"].edge_attr.argmax(dim=1)
    unique, counts = torch.unique(ratings, return_counts=True)
    total = counts.sum().item()
    print("Rating distribution:")
    for rating, count in zip(unique.tolist(), counts.tolist()):
        percentage = count / total * 100
        print(f"Rating {rating:.1f}: {count} ({percentage:.2f}%)")


def load_model(model_path: str, device: torch.device):
    saved_dict = torch.load(model_path, map_location=device)

    config = saved_dict["config"]
    data_info = saved_dict["data_info"]

    model = RecommendationModel(config, data_info, saved_dict["model_type"]).to(device)
    model.load_state_dict(saved_dict["model_state_dict"])
    model.eval()

    return model, config, data_info


def run_training(model_type="gat", small=False):
    config_sageconv = {
        "batch_size": 2048,
        "movie_feature_dim": 4,
        "user_feature_dim": 1,
        "hidden_dim": 256,
        "num_layers": 5,
        "edge_dim": 6,
        "learning_rate": 0.001,
        "epochs": 10,
        "neighbors": [10, 5, 5],
    }

    config_gat = {
        "batch_size": 2048,
        "movie_feature_dim": 4,
        "user_feature_dim": 1,
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 2,
        "edge_dim": 6,
        "learning_rate": 0.001,
        "epochs": 5,
        "neighbors": [10, 5, 5],
    }
    if model_type == "gat":
        config = config_gat
    else:
        config = config_sageconv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, train_loader, val_loader, test_loader = get_sageconv_movie_data_and_loaders(
        batch_size=config["batch_size"],
        neighbors=config["neighbors"],
        small=small,
    )
    print_rating_distribution(data)

    data_info = {
        "num_user_nodes": data["user"].num_nodes,
        "num_movie_nodes": data["movie"].num_nodes,
        "num_director_nodes": data["director"].num_nodes,
        "num_genre_nodes": data["genre"].num_nodes,
        "movie_feature_dim": data["movie"].x.shape[1],
        "user_feature_dim": data["user"].x.shape[1],
        "metadata": data.metadata(),
    }

    model = RecommendationModel(config, data_info, model_type).to(device)
    trained_model, losses, accuracies = train_gnn(
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=config["epochs"],
        lr=config["learning_rate"],
    )

    return trained_model, losses, accuracies
