import torch
import torch.optim as optim
from torch_geometric.nn import TransE
from torch_geometric.loader import DataLoader
from data import get_movie_data_kge
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

class MovieRecommendationModel:
    def __init__(self, num_nodes, num_relations, hidden_channels=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransE(num_nodes, num_relations, hidden_channels).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def create_loader(self, data, batch_size=1024000):
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
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            loss = self.model.loss(*[b.to(self.device) for b in batch])
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * batch[0].size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = sum(float(self.model.loss(*[b.to(self.device) for b in batch])) * batch[0].size(0) for batch in tqdm(loader, desc="Evaluating"))
        return total_loss / len(loader.dataset)

    def generate_negative_samples(self, edge_index, num_nodes, num_neg_samples):
        pos_edge_set = set(map(tuple, edge_index.t().cpu().numpy()))
        neg_edges = set()
        while len(neg_edges) < num_neg_samples:
            i, j = np.random.randint(0, num_nodes, size=2)
            if i != j and (i, j) not in pos_edge_set and (i, j) not in neg_edges:
                neg_edges.add((i, j))
        return torch.tensor(list(neg_edges), device=self.device).t()

    @torch.no_grad()
    def compute_auc(self, data, batch_size=512000):
        self.model.eval()
        pos_edge_index, edge_type = data.edge_index.to(self.device), data.edge_type.to(self.device)

        def process_edges(edges, types):
            preds = []
            for i in range(0, edges.size(1), batch_size):
                batch = edges[:, i:i+batch_size]
                pred = self.model(batch[0], types[i:i+batch_size], batch[1])
                preds.append(pred)
            return torch.cat(preds, dim=0)

        pos_pred = process_edges(pos_edge_index, edge_type)
        
        neg_edge_index = self.generate_negative_samples(pos_edge_index, data.num_nodes, pos_edge_index.size(1))
        neg_pred = process_edges(neg_edge_index, torch.zeros(neg_edge_index.size(1), dtype=torch.long, device=self.device))

        pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        true = np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))])

        return roc_auc_score(true, pred)

def main():
    train_data, val_data, test_data = get_movie_data_kge()
    model = MovieRecommendationModel(train_data.num_nodes, train_data.num_edge_types)
    
    train_loader = model.create_loader(train_data)
    val_loader = model.create_loader(val_data)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        loss = model.train_epoch(train_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 5 == 0:
            val_loss = model.evaluate(val_loader)
            print(f'Epoch: {epoch:03d}, Val Loss: {val_loss:.4f}')
        print(f'GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB')

    test_auc = model.compute_auc(test_data)
    print(f'Test AUC: {test_auc:.4f}')

if __name__ == "__main__":
    main()