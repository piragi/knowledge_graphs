from collections import defaultdict
import torch
import sys
import torch.nn.functional as F
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData, Data, RandomNodeLoader
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import analyze

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

    
    def split_user_ratings(self, ratings_df, train_ratio=0.8):
        # Convert timestamp to datetime if it's not already
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # Sort ratings by timestamp for each user
        ratings_df = ratings_df.sort_values(['userId', 'timestamp'])
        
        # Group by user and split
        train_ratings = ratings_df.groupby('userId').apply(lambda x: x.iloc[:int(len(x) * train_ratio)]).reset_index(drop=True)
        test_ratings = ratings_df[~ratings_df.index.isin(train_ratings.index)]
        
        return train_ratings, test_ratings

    def generate_user_features(self, ratings_df, movies_df):
        merged_df = ratings_df.merge(movies_df, on='movieId')
        user_avg_rating = merged_df.groupby('userId')['rating'].mean()

        genre_dummies = merged_df['genres'].str.get_dummies('|')
        genre_preference = merged_df.apply(lambda row: row['rating'] * genre_dummies.loc[row.name], axis=1)
        user_genre_preference = genre_preference.groupby(merged_df['userId']).sum()
        user_genre_preference = user_genre_preference.div(user_genre_preference.sum(axis=1), axis=0)

        # Compute other preferences
        preferences = ['budget', 'revenue', 'vote_count']
        user_preferences = merged_df.groupby('userId').apply(lambda x: pd.Series({
            f'{pref}_preference': (x['rating'] * x[pref]).sum() / x['rating'].sum()
            for pref in preferences
        }))

        # Combine all features
        user_features = pd.concat([
            user_avg_rating.rename('avg_rating'),
            user_genre_preference,
            user_preferences
        ], axis=1).fillna(0)

        # Normalize non-genre features
        non_genre_columns = ['avg_rating'] + [f'{pref}_preference' for pref in preferences]
        user_features[non_genre_columns] = (user_features[non_genre_columns] - user_features[non_genre_columns].mean()) / user_features[non_genre_columns].std()
        
        return user_features  # Return DataFrame with userId as index

    def apply_user_features(self, test_ratings_df, train_user_features):
        # Get unique users in the test set
        test_users = test_ratings_df['userId'].unique()
        
        # Create a DataFrame for test user features
        test_user_features = pd.DataFrame(index=test_users, columns=train_user_features.columns)
        
        # Fill in features for users that exist in the training set
        common_users = test_user_features.index.intersection(train_user_features.index)
        test_user_features.loc[common_users] = train_user_features.loc[common_users]
        
        # For new users, use the mean of the training features
        new_users = test_user_features.index.difference(train_user_features.index)
        mean_features = train_user_features.mean()
        
        # Only set mean features if there are new users
        if len(new_users) > 0:
            test_user_features.loc[new_users] = mean_features
        
        # Add a flag for new users
        test_user_features['is_new_user'] = 0
        if len(new_users) > 0:
            test_user_features.loc[new_users, 'is_new_user'] = 1
        
        # Fill NaN values with 0 (in case any remain)
        test_user_features = test_user_features.fillna(0)
        
        return test_user_features

    def generate_movie_features(self, movies_df, ratings_df):
        movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'])]
        genres = movies_df['genres'].str.get_dummies('|')
        numerical_features = ['vote_average', 'revenue', 'budget', 'vote_count']
        movies_df['vote_average'] = movies_df['vote_average'].fillna(5.0)

        # Convert to numpy, standardize, and create tensors
        num_features = movies_df[numerical_features].values.astype(float)
        num_features_standardized = self.standardize(num_features)
        num_tensors = torch.from_numpy(num_features_standardized).float()
        return torch.cat([torch.from_numpy(genres.values).float(), num_tensors], dim=1)

    def process_data(self, ratings_df):
        ratings_df = ratings_df.dropna(subset=['rating'])
        # self.analyze_user_data(ratings_df, movies_df) 

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
        return edge_index_user_to_movie, ratings_df, unique_user_id, unique_movie_id

    def create_hetero_data(self, ratings_df, movies_df, edge_index_user_to_movie, unique_user_id, unique_movie_id, user_features=None):
        data = HeteroData()

        data['user'].node_id = torch.arange(len(unique_user_id))
        data['movie'].node_id = torch.arange(len(unique_movie_id))

        data['movie'].x = self.generate_movie_features(movies_df, ratings_df)
        # data['user'].x = torch.tensor(user_features.values, dtype=torch.float)

        data['user', 'rates', 'movie'].edge_index = edge_index_user_to_movie

        grouped_ratings = torch.from_numpy(ratings_df['grouped_rating'].values).long()
        one_hot_ratings = F.one_hot(grouped_ratings, num_classes=7).float()
        data['user', 'rates', 'movie'].edge_attr = one_hot_ratings
        data['user', 'rates', 'movie'].edge_label = grouped_ratings

        data = T.ToUndirected()(data)
        return data

    def generate_loaders(self, train_data, batch_size=1024, num_neighbors=[5,5]):
        transform = T.RandomLinkSplit(
            num_val=0.2,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
            key='edge_label',
        )
        train_data, val_data, test_data = transform(train_data)

        create_loader = lambda data, batch_size=batch_size, shuffle=False, neg_sampling_ratio=0.0: LinkNeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle,
                                                        neg_sampling_ratio=neg_sampling_ratio,
                                                        edge_label_index=(('user', 'rates', 'movie'), data['user','rates', 'movie'].edge_label_index), 
                                                        edge_label= data['user', 'rates', 'movie'].edge_label)#torch.argmax(data['user', 'rates', 'movie'].edge_attr, dim=1))

        train_loader = create_loader(train_data, shuffle=True, neg_sampling_ratio=2.0)
        val_loader = create_loader(val_data, batch_size=2048*3)
        test_loader = create_loader(test_data)

        return train_loader, val_loader, test_loader

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
    
def get_sageconv_movie_data_and_loaders(small=False):
    processor = MovieDataProcessor(small)
    movies_df, ratings_df = processor.read_tmdb_movies() 
    edge_index_user_to_movie, train_ratings_df, unique_user_id, unique_movie_id = processor.process_data(ratings_df)
    train_data = processor.create_hetero_data(train_ratings_df, movies_df, edge_index_user_to_movie, unique_user_id, unique_movie_id) 
    train_loader, val_loader, test_loader = processor.generate_loaders(train_data)
    return train_data, train_loader, val_loader, test_loader

def get_movie_data_kge(small=False):
    processor = MovieDataProcessor(small)
    return processor.prepare_data_kge()

