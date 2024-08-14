import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import LinkNeighborLoader

class MovieDataProcessor:
    def __init__(self):
        torch.manual_seed(500)

    def read_tmdb_movies(self):
        columns_of_interest = ['budget', 'revenue', 'vote_average', 'vote_count', 'production_companies', 'id']
        movies_df = pd.read_csv("./data/tmdb_5000_movies.csv", usecols=columns_of_interest)
        movies_df.fillna({
            'budget': 0,
            'revenue': 0,
            'vote_count': 0,
            'production_companies': ''
        })
        links_df = pd.read_csv("./data/links.csv", usecols=['movieId', 'tmdbId'])
        movies_linked_df = pd.merge(movies_df, links_df, left_on='id', right_on='tmdbId', how='inner')
        movies_info_df = pd.read_csv("./data/movies.csv")
        movies_df = pd.merge(movies_linked_df, movies_info_df, on='movieId', how='inner')
        ratings_df = pd.read_csv("./data/ratings.csv")
        ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])]
        return movies_df, ratings_df

    @staticmethod
    def standardize(x):
        return (x - x.mean()) / x.std()

    def process_data(self):
        movies_df, ratings_df = self.read_tmdb_movies()
        
        # Filter movies based on ratings
        movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'])]
        
        # Process genres
        genres = movies_df['genres'].str.get_dummies('|')
        
        # Process numerical features
        numerical_features = ['vote_average', 'revenue', 'budget', 'vote_count']
        movies_df['vote_average'] = movies_df['vote_average'].fillna(5.0)
        
        # Convert to numpy, standardize, and create tensors
        num_features = movies_df[numerical_features].values.astype(float)
        num_features_standardized = self.standardize(num_features)
        num_tensors = torch.from_numpy(num_features_standardized).float()
        
        # Combine all features
        movie_feat = torch.cat([torch.from_numpy(genres.values).float(), num_tensors], dim=1)
        
        # Construct a compact representation of the data
        unique_user_id = ratings_df['userId'].unique()
        unique_user_id = pd.DataFrame(data={
            'userId': unique_user_id,
            'mappedId': pd.RangeIndex(len(unique_user_id)),
        })
        unique_movie_id = ratings_df['movieId'].unique()
        unique_movie_id = pd.DataFrame(data={
            'movieId': unique_movie_id,
            'mappedId': pd.RangeIndex(len(unique_movie_id)),
        })
        
        ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id, on='userId', how='left')
        ratings_user_id = torch.from_numpy(ratings_user_id['mappedId'].values)
        ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id, on='movieId', how='left')
        ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedId'].values)
        
        edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
        
        return movie_feat, edge_index_user_to_movie, ratings_df, unique_user_id, unique_movie_id

    def create_hetero_data(self):
        movie_feat, edge_index_user_to_movie, ratings_df, unique_user_id, unique_movie_id = self.process_data()
        
        data = HeteroData()
        
        # Save node indices
        data['user'].node_id = torch.arange(len(unique_user_id))
        data['movie'].node_id = torch.arange(len(unique_movie_id))
        
        # Add node features
        data['movie'].x = movie_feat
        data['user', 'rates', 'movie'].edge_index = edge_index_user_to_movie
        data['user', 'rates', 'movie'].edge_label = torch.from_numpy(ratings_df['rating'].values).to(torch.long)
        
        mask = data['user', 'rates', 'movie'].edge_label >= 4
        del data['user', 'rates', 'movie'].edge_label 
        data['user', 'rates', 'movie'].edge_index = data['user', 'rates', 'movie'].edge_index[:, mask]
        
        return data

    def prepare_data(self):
        data = self.create_hetero_data()
        data = T.ToUndirected()(data)
        
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=[("user", "rates", "movie")],
            rev_edge_types=[("movie", "rev_rates", "user")],
        )
        train_data, val_data, test_data = transform(data)
        
        return data, train_data, val_data, test_data

    def process_data_kge(self):
        movies_df, ratings_df = self.read_tmdb_movies()
        
        # Create entity mappings
        user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
        movie_mapping = {movie_id: idx + len(user_mapping) for idx, movie_id in enumerate(movies_df['movieId'].unique())}
        
        # Create edge index
        user_indices = torch.tensor([user_mapping[user] for user in ratings_df['userId']], dtype=torch.long)
        movie_indices = torch.tensor([movie_mapping[movie] for movie in ratings_df['movieId']], dtype=torch.long)
        edge_index = torch.stack([user_indices, movie_indices], dim=0)

        # Create edge type (all 0 for 'rates' relation)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        assert len(user_indices) == len(movie_indices)
        assert len(edge_type) == len(movie_indices)

        # Create Data object
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(user_mapping) + len(movie_mapping),
            num_edge_types=1
        )

        return data

    def prepare_data_kge(self):
        data = self.process_data_kge()
        
        # Split data
        num_edges = data.edge_index.size(1)
        perm = torch.randperm(num_edges)
        train_edges = int(0.8 * num_edges)
        val_edges = int(0.1 * num_edges)

        train_data = Data(
            edge_index=data.edge_index[:, perm[:train_edges]],
            edge_type=data.edge_type[perm[:train_edges]],
            num_nodes=data.num_nodes,
            num_edge_types=data.num_edge_types
        )

        val_data = Data(
            edge_index=data.edge_index[:, perm[train_edges:train_edges+val_edges]],
            edge_type=data.edge_type[perm[train_edges:train_edges+val_edges]],
            num_nodes=data.num_nodes,
            num_edge_types=data.num_edge_types
        )

        test_data = Data(
            edge_index=data.edge_index[:, perm[train_edges+val_edges:]],
            edge_type=data.edge_type[perm[train_edges+val_edges:]],
            num_nodes=data.num_nodes,
            num_edge_types=data.num_edge_types
        )

        return train_data, val_data, test_data
    
    def create_loaders(self, train_data, val_data):
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[10, 10, 10],
            edge_label_index=(('user', 'rates', 'movie'), train_data['user', 'rates', 'movie'].edge_label_index),
            edge_label=train_data['user', 'rates', 'movie'].edge_label,
            batch_size=128,
            shuffle=True,
            neg_sampling_ratio=2.0
        )

        edge_label_index = val_data["user", "rates", "movie"].edge_label_index
        edge_label = val_data["user", "rates", "movie"].edge_label
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[10, 10, 10],
            edge_label_index=(("user", "rates", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=128*3,
            shuffle=False,
        )

        return train_loader, val_loader

def get_sageconv_movie_data_and_loaders():
    processor = MovieDataProcessor()
    data, train_data, val_data, _ = processor.prepare_data()
    train_loader, val_loader = processor.create_loaders(train_data, val_data)
    return data, train_loader, val_loader

def get_movie_data_kge():
    processor = MovieDataProcessor()
    return processor.prepare_data_kge()