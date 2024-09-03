import torch
import torch.optim as optim
from torch_geometric.nn import TransE
from data import get_movie_data_kge
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

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

    def generate_negative_samples_batch(self, edge_index, edge_type, num_nodes, num_neg_samples, batch_size=1000000):
        pos_edges = set(map(tuple, edge_index.t().cpu().numpy()))
        existing_edges = defaultdict(set)
        for (head, tail) in pos_edges:
            existing_edges[head].add(tail)
    
        neg_edges = []
        neg_types = []
    
        while len(neg_edges) < num_neg_samples:
            # Generate a batch of candidate edges
            heads = np.random.randint(0, num_nodes, batch_size)
            tails = np.random.randint(0, num_nodes, batch_size)
            types = np.random.choice(edge_type.cpu().numpy(), batch_size)
        
            # Filter the batch
            for i in range(batch_size):
                if len(neg_edges) >= num_neg_samples:
                    break
                if tails[i] not in existing_edges[heads[i]] and heads[i] != tails[i]:
                    neg_edges.append((heads[i], tails[i]))
                    neg_types.append(types[i])
    
        neg_edge_index = torch.tensor(neg_edges, device=self.device).t()
        neg_edge_type = torch.tensor(neg_types, device=self.device)
    
        return neg_edge_index, neg_edge_type

    @torch.no_grad()
    def compute_auc(self, edge_index, edge_type, batch_size=512000):
        self.model.eval()
        pos_edge_index, pos_edge_type = edge_index.to(self.device), edge_type.to(self.device)

        def process_edges(edges, types):
            preds = []
            for i in range(0, edges.size(1), batch_size):
                batch_edges = edges[:, i:i+batch_size]
                batch_types = types[i:i+batch_size]
                pred = self.model(batch_edges[0], batch_types, batch_edges[1])
                preds.append(pred)
            return torch.cat(preds, dim=0)

        pos_pred = process_edges(pos_edge_index, pos_edge_type)
    
        # Generate negative samples
        neg_edge_index, neg_edge_type = self.generate_negative_samples_batch(
            pos_edge_index, pos_edge_type, self.model.num_nodes, pos_edge_index.size(1)
        )
        neg_pred = process_edges(neg_edge_index, neg_edge_type)

        pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        true = np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))])

        return roc_auc_score(true, pred)

def main():
    train_data, val_data, test_data = get_movie_data_kge()
    model = MovieRecommendationModel(train_data.num_nodes, train_data.num_edge_types)
    
    train_loader = model.create_loader(train_data)
    val_loader = model.create_loader(val_data)

    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        loss = model.train_epoch(train_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 5 == 0:
            val_loss = model.evaluate(val_loader)
            val_auc = model.compute_auc(val_data.edge_index, val_data.edge_type)
            print(f'Epoch: {epoch:03d}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        print(f'GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB')

    test_auc = model.compute_auc(test_data.edge_index, test_data.edge_type)
    print(f'Test AUC: {test_auc:.4f}')

if __name__ == "__main__":
    main()

def train_kge_model():
    train_data, val_data, test_data = get_movie_data_kge(small=True)
    model = MovieRecommendationModel(train_data.num_nodes, train_data.num_edge_types)
    
    train_loader = model.create_loader(train_data)
    val_loader = model.create_loader(val_data)

    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        loss = model.train_epoch(train_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 5 == 0:
            val_loss = model.evaluate(val_loader)
            val_auc = model.compute_auc(val_data.edge_index, val_data.edge_type)
            print(f'Epoch: {epoch:03d}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        print(f'GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB')

    test_auc = model.compute_auc(test_data.edge_index, test_data.edge_type)
    print(f'Test AUC: {test_auc:.4f}')
    return model
