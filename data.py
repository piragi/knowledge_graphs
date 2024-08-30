import torch
import sys
import torch.nn.functional as F
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData, Data, RandomNodeLoader
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

class MovieDataProcessor:
    def __init__(self, small=False):
        torch.manual_seed(500)
        self.path = './data/small' if small else './data'

    def read_tmdb_movies(self):
        columns_of_interest = ['budget', 'revenue', 'vote_average', 'vote_count', 'production_companies', 'id']
        movies_df = pd.read_csv(f"./data/tmdb_5000_movies.csv", usecols=columns_of_interest)
        movies_df.fillna({
            'budget': 0,
            'revenue': 0,
            'vote_count': 0,
            'production_companies': ''
        })
        links_df = pd.read_csv(f"{self.path}/links.csv", usecols=['movieId', 'tmdbId'])
        movies_linked_df = pd.merge(movies_df, links_df, left_on='id', right_on='tmdbId', how='inner')
        movies_info_df = pd.read_csv(f"{self.path}/movies.csv")
        movies_df = pd.merge(movies_linked_df, movies_info_df, on='movieId', how='inner')
        ratings_df = pd.read_csv(f"{self.path}/ratings.csv")
        ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])]
        return movies_df, ratings_df

    @staticmethod
    def standardize(x):
        return (x - x.mean()) / x.std()

    def process_data(self):
        movies_df, ratings_df = self.read_tmdb_movies()

        # Remove entries with null or undefined ratings
        ratings_df = ratings_df.dropna(subset=['rating'])

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

        # New rating grouping
        def group_ratings(rating):
            if pd.isna(rating) or rating <= 1.0:
                return 0  # Low (including any remaining NaN or undefined values)
            elif rating <= 2.0:
                return 1  # Below Average
            elif rating <= 2.5:
                return 2  # Fair
            elif rating <= 3.0:
                return 3  # Average
            elif rating <= 4.0:
                return 4  # Good
            else:
                return 5  # Excellent

        ratings_df['grouped_rating'] = ratings_df['rating'].apply(group_ratings)

        return movie_feat, edge_index_user_to_movie, ratings_df, unique_user_id, unique_movie_id

    def create_hetero_data(self):
        movie_feat, edge_index_user_to_movie, ratings_df, unique_user_id, unique_movie_id = self.process_data()

        data = HeteroData()

        # Save node indices
        data['user'].node_id = torch.arange(len(unique_user_id))
        data['movie'].node_id = torch.arange(len(unique_movie_id))

        # Add node features
        data['movie'].x = movie_feat

        # Create a single edge type with rating as edge attribute
        data['user', 'rates', 'movie'].edge_index = edge_index_user_to_movie

        grouped_ratings = torch.from_numpy(ratings_df['grouped_rating'].values).long()
        # one_hot_ratings = F.one_hot(grouped_ratings, num_classes=7).float()
        # data['user', 'rates', 'movie'].edge_attr = one_hot_ratings
        data['user', 'rates', 'movie'].edge_label = grouped_ratings

        data = T.ToUndirected()(data)
        return data

    def prepare_data(self, batch_size=2048, num_neighbors=[10, 5]):
        data = self.create_hetero_data()
        
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
            key='edge_label',
        )
        train_data, val_data, test_data = transform(data)
        
        create_loader = lambda data, shuffle=False, batch_size=2048, neg_sampling_ratio=0.0: LinkNeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle,
                                                        neg_sampling_ratio=neg_sampling_ratio,
                                                        edge_label_index=(('user', 'rates', 'movie'), data['user','rates', 'movie'].edge_label_index), 
                                                        edge_label= data['user', 'rates', 'movie'].edge_label)#torch.argmax(data['user', 'rates', 'movie'].edge_attr, dim=1))
        train_loader = create_loader(train_data, shuffle=True, neg_sampling_ratio=2.0)
        val_loader = create_loader(val_data, batch_size=2048*3)
        test_loader = create_loader(test_data)

#        print(train_data)
#        print(train_data['user', 'rates', 'movie'].edge_index)
#        print(torch.argmax(train_data['user', 'rates', 'movie'].edge_attr, dim=1))
#        for batch in train_loader:
#            edge = batch['user', 'rates', 'movie']
#            #print(batch)
#            print(f'edge.edge_label_index={edge.edge_label_index}')
#            print(f'edge.edge_label={edge.edge_label}')
#            print(torch.argmax(edge.edge_attr, dim=1))
#            print(edge.input_id[0])
#            print(train_data['user', 'rates', 'movie'].edge_index.T[edge.input_id[0]])
#            print('-----')
#            print(torch.argmax(train_data['user', 'rates', 'movie'].edge_attr[edge.input_id[0]], dim=0))
#            print('------')

        return train_data, train_loader, val_loader, test_loader

    def process_data_kge(self):
        movies_df, ratings_df = self.read_tmdb_movies()
    
        # Create entity mappings
        user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
        movie_mapping = {movie_id: idx + len(user_mapping) for idx, movie_id in enumerate(movies_df['movieId'].unique())}
    
        # Create edge index
        user_indices = torch.tensor([user_mapping[user] for user in ratings_df['userId']], dtype=torch.long)
        movie_indices = torch.tensor([movie_mapping[movie] for movie in ratings_df['movieId']], dtype=torch.long)
        edge_index = torch.stack([user_indices, movie_indices], dim=0)
    
        # Create edge type based on ratings
        # Ratings are from 0.5 to 5.0 with 0.5 step, so we have 10 distinct values
        ratings = ratings_df['rating'].values
        edge_type = torch.tensor((ratings * 2 - 1).astype(int), dtype=torch.long)
    
        assert len(user_indices) == len(movie_indices) == len(edge_type)
        assert edge_type.max() == 9 and edge_type.min() == 0, "Edge types should be between 0 and 9"
    
        # Create Data object
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(user_mapping) + len(movie_mapping),
            num_edge_types=10  # We now have 10 possible ratings
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

def get_sageconv_movie_data_and_loaders(small=False):
    processor = MovieDataProcessor(small)
    train_data, train_loader, val_loader, test_loader = processor.prepare_data()
    return train_data, train_loader, val_loader, test_loader

def get_movie_data_kge(small=False):
    processor = MovieDataProcessor(small)
    return processor.prepare_data_kge()
