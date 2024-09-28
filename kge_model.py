from datetime import datetime

import torch
import torch.optim as optim
from torch_geometric.nn import ComplEx, RotatE, TransE
from tqdm import tqdm

from data import get_movie_data_kge


class MovieRecommendationModel:
    def __init__(
        self, num_nodes, num_relations, hidden_channels=100, model_type="transe"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.config = {
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "hidden_channels": hidden_channels,
            "model_type": model_type,
        }
        self.data_info = None  # This will be set when loading data

        if model_type == "transe":
            self.model = TransE(num_nodes, num_relations, hidden_channels).to(
                self.device
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        elif model_type == "rotate":
            self.model = RotatE(
                num_nodes, num_relations, hidden_channels, margin=2.5
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        elif model_type == "complex":
            self.model = ComplEx(num_nodes, num_relations, hidden_channels).to(
                self.device
            )
            self.optimizer = optim.Adagrad(
                self.model.parameters(), lr=0.001, weight_decay=1e-6
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def create_loader(self, data, batch_size=20000):
        return self.model.loader(
            head_index=data.edge_index[0].to(self.device),
            rel_type=data.edge_type.to(self.device),
            tail_index=data.edge_index[1].to(self.device),
            batch_size=batch_size,
            shuffle=True,
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(*[b.to(self.device) for b in batch])
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * batch[0].size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = sum(
            float(self.model.loss(*[b.to(self.device) for b in batch]))
            * batch[0].size(0)
            for batch in tqdm(loader, desc="Evaluating")
        )
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def test(self, data, batch_size=20000, k=10):
        self.model.eval()
        return self.model.test(
            head_index=data.edge_index[0].to(self.device),
            rel_type=data.edge_type.to(self.device),
            tail_index=data.edge_index[1].to(self.device),
            batch_size=batch_size,
            k=k,
        )

    def save_model(self, accuracy: float, save_path: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"model_{self.model_type}_{timestamp}_acc{accuracy:.4f}.pt"
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "data_info": self.data_info,
            "model_type": self.model_type,
            "accuracy": accuracy,
        }
        torch.save(save_dict, f"{save_path}/{filename}")
        print(f"Model saved: {filename}")

    @classmethod
    def load_model(cls, load_path: str):
        save_dict = torch.load(load_path)
        config = save_dict["config"]
        model = cls(
            num_nodes=config["num_nodes"],
            num_relations=config["num_relations"],
            hidden_channels=config["hidden_channels"],
            model_type=config["model_type"],
        )
        model.model.load_state_dict(save_dict["model_state_dict"])
        model.data_info = save_dict["data_info"]
        print(f"Model loaded: {load_path}")
        print(f"Loaded model accuracy: {save_dict['accuracy']:.4f}")
        return model


def run_kge_train(model_type="transe", small=False):
    train_data, val_data, test_data = get_movie_data_kge(small=small)
    model = MovieRecommendationModel(
        train_data.num_nodes, train_data.num_edge_types, model_type=model_type
    )
    model.data_info = {
        "num_nodes": train_data.num_nodes,
        "num_edge_types": train_data.num_edge_types,
    }
    train_loader = model.create_loader(train_data)
    num_epochs = 700

    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        model.train_epoch(train_loader)
        if epoch % 25 == 0:
            rank, mrr, hits = model.test(val_data)
            accuracy = hits  # Using Hits@10 as accuracy
            print(
                f"Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, "
                f"Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}"
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model.save_model(accuracy, "./model")

    # Final test
    rank, mrr, hits_at_10 = model.test(test_data)
    model.save_model(hits_at_10, "./model")
    print(
        f"Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, "
        f"Test Hits@10: {hits_at_10:.4f}"
    )
