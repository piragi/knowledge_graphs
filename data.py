import ast
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData, RandomNodeLoader
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader


class MovieDataProcessor:
    def __init__(self, small=False):
        # torch.manual_seed(500)
        self.path = "./data/small" if small else "./data"
        self.processed_ratings_path = os.path.join(self.path, "ratings_processed.csv")
        self.processed_movies_path = os.path.join(self.path, "movies_processed.csv")
        self.movie_mapping = None
        self.director_mapping = None
        self.genre_mapping = None
        self.reverse_movie_mapping = None
        self.reverse_director_mapping = None
        self.reverse_genre_mapping = None
        self.movie_titles = None
        self.director_names = None

    def read_tmdb_movies(self, ratings_path="ratings.csv"):
        # Check if processed files exist
        if os.path.exists(self.processed_ratings_path) and os.path.exists(
            self.processed_movies_path
        ):
            print("Loading processed files...")
            ratings_df = pd.read_csv(self.processed_ratings_path)
            movies_df = pd.read_csv(self.processed_movies_path)
            return movies_df, ratings_df

        print("Processing files...")
        # If processed files don't exist, process the data
        columns_of_interest = [
            "budget",
            "revenue",
            "vote_average",
            "vote_count",
            "production_companies",
            "id",
        ]
        movies_df = pd.read_csv(
            f"./data/tmdb_5000_movies.csv", usecols=columns_of_interest
        )
        movies_df.fillna(
            {"budget": 0, "revenue": 0, "vote_count": 0, "production_companies": ""}
        )
        links_df = pd.read_csv(f"{self.path}/links.csv", usecols=["movieId", "tmdbId"])
        movies_linked_df = pd.merge(
            movies_df, links_df, left_on="id", right_on="tmdbId", how="inner"
        )
        movies_info_df = pd.read_csv(f"{self.path}/movies.csv")
        movies_df = pd.merge(
            movies_linked_df, movies_info_df, on="movieId", how="inner"
        )
        ratings_df = pd.read_csv(f"{self.path}/{ratings_path}")
        ratings_df = ratings_df[ratings_df["movieId"].isin(movies_df["movieId"])]

        credits_df = pd.read_csv("./data/tmdb_5000_credits.csv")
        credits_df["crew"] = credits_df["crew"].apply(ast.literal_eval)

        def get_director_info(crew):
            director = next(
                (member for member in crew if member["job"] == "Director"), None
            )
            return (director["id"], director["name"]) if director else (None, None)

        credits_df[["directorId", "directorName"]] = (
            credits_df["crew"].apply(get_director_info).apply(pd.Series)
        )

        # Merge director information with movies_df
        movies_df = pd.merge(
            movies_df,
            credits_df[["movie_id", "directorId", "directorName"]],
            left_on="id",
            right_on="movie_id",
            how="left",
        )
        movies_df = movies_df[movies_df["movieId"].isin(ratings_df["movieId"])]

        # Merge director information with ratings_df
        ratings_df = pd.merge(
            ratings_df, movies_df["movieId"], on="movieId", how="left"
        )

        # Save processed files
        movies_df.to_csv(self.processed_movies_path, index=False)
        ratings_df.to_csv(self.processed_ratings_path, index=False)

        return movies_df, ratings_df

    @staticmethod
    def standardize(x):
        return (x - x.mean()) / x.std()

    def split_user_ratings(self, ratings_df, train_ratio=0.8):
        # Convert timestamp to datetime if it's not already
        ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], unit="s")

        # Sort ratings by timestamp for each user
        ratings_df = ratings_df.sort_values(["userId", "timestamp"])

        # Group by user and split
        train_ratings = (
            ratings_df.groupby("userId")
            .apply(lambda x: x.iloc[: int(len(x) * train_ratio)])
            .reset_index(drop=True)
        )
        test_ratings = ratings_df[~ratings_df.index.isin(train_ratings.index)]

        return train_ratings, test_ratings

    def generate_user_features(self, ratings_df, movies_df):
        merged_df = ratings_df.merge(movies_df, on="movieId")
        user_avg_rating = merged_df.groupby("userId")["rating"].mean()

        genre_dummies = merged_df["genres"].str.get_dummies("|")
        genre_preference = merged_df.apply(
            lambda row: row["rating"] * genre_dummies.loc[row.name], axis=1
        )
        user_genre_preference = genre_preference.groupby(merged_df["userId"]).sum()
        user_genre_preference = user_genre_preference.div(
            user_genre_preference.sum(axis=1), axis=0
        )

        # Compute other preferences
        preferences = ["budget", "revenue", "vote_count"]
        user_preferences = merged_df.groupby("userId").apply(
            lambda x: pd.Series(
                {
                    f"{pref}_preference": (x["rating"] * x[pref]).sum()
                    / x["rating"].sum()
                    for pref in preferences
                }
            )
        )

        # Combine all features
        user_features = pd.concat(
            [
                user_avg_rating.rename("avg_rating"),
                user_genre_preference,
                user_preferences,
            ],
            axis=1,
        ).fillna(0)

        # Normalize non-genre features
        non_genre_columns = ["avg_rating"] + [
            f"{pref}_preference" for pref in preferences
        ]
        user_features[non_genre_columns] = (
            user_features[non_genre_columns] - user_features[non_genre_columns].mean()
        ) / user_features[non_genre_columns].std()

        return user_features  # Return DataFrame with userId as index

    def apply_user_features(self, test_ratings_df, train_user_features):
        # Get unique users in the test set
        test_users = test_ratings_df["userId"].unique()

        # Create a DataFrame for test user features
        test_user_features = pd.DataFrame(
            index=test_users, columns=train_user_features.columns
        )

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
        test_user_features["is_new_user"] = 0
        if len(new_users) > 0:
            test_user_features.loc[new_users, "is_new_user"] = 1

        # Fill NaN values with 0 (in case any remain)
        test_user_features = test_user_features.fillna(0)

        return test_user_features

    def generate_movie_features(self, movies_df, ratings_df):
        movies_df = movies_df[movies_df["movieId"].isin(ratings_df["movieId"])]
        genres = movies_df["genres"].str.get_dummies("|")
        numerical_features = ["vote_average", "revenue", "budget", "vote_count"]
        movies_df["vote_average"] = movies_df["vote_average"].fillna(5.0)

        # Convert to numpy, standardize, and create tensors
        num_features = movies_df[numerical_features].values.astype(float)
        num_features_standardized = self.standardize(num_features)
        num_tensors = torch.from_numpy(num_features_standardized).float()
        return num_tensors
        # return torch.cat([torch.from_numpy(genres.values).float(), num_tensors], dim=1)

    def process_data(self, ratings_df, movies_df):
        ratings_df = ratings_df.dropna(subset=["rating"])
        # self.analyze_user_data(ratings_df, movies_df)

        # Process genres
        genres = movies_df["genres"].str.get_dummies("|")
        genres["movieId"] = movies_df["movieId"]  # Add movieId to the genres DataFrame

        # Melt the genres DataFrame to create a long format
        genres_long = genres.melt(
            id_vars=["movieId"], var_name="genreName", value_name="hasGenre"
        )

        # Keep only the rows where the movie has the genre
        genres_long = genres_long[genres_long["hasGenre"] == 1].drop("hasGenre", axis=1)

        # Create unique mappings
        unique_genre_id = pd.DataFrame(
            data={
                "genreName": genres.columns[:-1],  # Exclude 'movieId'
                "mappedId": pd.RangeIndex(len(genres.columns) - 1),
            }
        )

        unique_director_id = movies_df["directorId"].unique()
        unique_director_id = pd.DataFrame(
            data={
                "directorId": unique_director_id,
                "mappedId": pd.RangeIndex(len(unique_director_id)),
            }
        )

        unique_movie_id = movies_df["movieId"].unique()
        unique_movie_id = pd.DataFrame(
            data={
                "movieId": unique_movie_id,
                "mappedId": pd.RangeIndex(len(unique_movie_id)),
            }
        )
        # Create edge indices for genre to movie
        genre_movie_mapped = pd.merge(
            genres_long, unique_genre_id, on="genreName", how="left"
        )
        genre_movie_mapped = pd.merge(
            genre_movie_mapped, unique_movie_id, on="movieId", how="left"
        )

        genre_ids = torch.from_numpy(genre_movie_mapped["mappedId_x"].values)
        movie_ids = torch.from_numpy(genre_movie_mapped["mappedId_y"].values)
        edge_index_genre_to_movie = torch.stack([genre_ids, movie_ids], dim=0)

        directed_director_id = pd.merge(
            movies_df["directorId"], unique_director_id, on="directorId", how="left"
        )
        directed_director_id = torch.from_numpy(directed_director_id["mappedId"].values)
        directed_movie_id = pd.merge(
            movies_df["movieId"], unique_movie_id, on="movieId", how="left"
        )
        directed_movie_id = torch.from_numpy(directed_movie_id["mappedId"].values)

        edge_index_director_to_movie = torch.stack(
            [directed_director_id, directed_movie_id], dim=0
        )
        # Construct a compact representation of the data
        unique_user_id = ratings_df["userId"].unique()
        unique_user_id = pd.DataFrame(
            data={
                "userId": unique_user_id,
                "mappedId": pd.RangeIndex(len(unique_user_id)),
            }
        )
        unique_movie_id = ratings_df["movieId"].unique()
        unique_movie_id = pd.DataFrame(
            data={
                "movieId": unique_movie_id,
                "mappedId": pd.RangeIndex(len(unique_movie_id)),
            }
        )

        ratings_user_id = pd.merge(
            ratings_df["userId"], unique_user_id, on="userId", how="left"
        )
        ratings_user_id = torch.from_numpy(ratings_user_id["mappedId"].values)
        ratings_movie_id = pd.merge(
            ratings_df["movieId"], unique_movie_id, on="movieId", how="left"
        )
        ratings_movie_id = torch.from_numpy(ratings_movie_id["mappedId"].values)

        edge_index_user_to_movie = torch.stack(
            [ratings_user_id, ratings_movie_id], dim=0
        )

        group_ratings = lambda rating: (
            min(5, max(0, int(rating))) if pd.notna(rating) else 0
        )
        ratings_df["grouped_rating"] = ratings_df["rating"].apply(group_ratings)
        print(ratings_df["grouped_rating"].unique())

        return (
            edge_index_user_to_movie,
            edge_index_director_to_movie,
            edge_index_genre_to_movie,
            ratings_df,
            unique_user_id,
            unique_movie_id,
            unique_director_id,
            unique_genre_id,
        )

    def create_hetero_data(
        self,
        ratings_df,
        movies_df,
        edge_index_user_to_movie,
        edge_index_director_to_movie,
        edge_index_genre_to_movie,
        unique_user_id,
        unique_movie_id,
        unique_director_id,
        unique_genre_id,
        user_features=None,
    ):
        data = HeteroData()

        data["user"].node_id = torch.arange(len(unique_user_id))
        data["movie"].node_id = torch.arange(len(unique_movie_id))
        data["director"].node_id = torch.arange(len(unique_director_id))
        data["genre"].node_id = torch.arange(len(unique_genre_id))

        data["movie"].x = self.generate_movie_features(movies_df, ratings_df)
        # data['user'].x = torch.tensor(user_features.values, dtype=torch.float)

        data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
        data["director", "directs", "movie"].edge_index = edge_index_director_to_movie
        data["genre", "is_genre", "movie"].edge_index = edge_index_genre_to_movie

        grouped_ratings = torch.from_numpy(ratings_df["grouped_rating"].values).long()
        one_hot_ratings = F.one_hot(grouped_ratings, num_classes=6).float()
        data["user", "rates", "movie"].edge_attr = one_hot_ratings
        data["user", "rates", "movie"].edge_label = grouped_ratings

        data = T.ToUndirected()(data)
        return data

    def generate_loaders(self, train_data, batch_size=2048, num_neighbors=[5, 5]):
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=[
                ("user", "rates", "movie"),
                ("director", "directs", "movie"),
                ("genre", "is_genre", "movie"),
            ],
            rev_edge_types=[
                ("movie", "rev_rates", "user"),
                ("movie", "rev_directs", "director"),
                ("movie", "rev_is_genre", "genre"),
            ],
            key="edge_label",
        )
        train_data, val_data, test_data = transform(train_data)

        create_loader = lambda data, batch_size=batch_size, shuffle=False, neg_sampling_ratio=0.0: LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_label_index=(
                ("user", "rates", "movie"),
                data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=data["user", "rates", "movie"].edge_label,
        )  # torch.argmax(data['user', 'rates', 'movie'].edge_attr, dim=1))

        train_loader = create_loader(train_data, shuffle=True, neg_sampling_ratio=2.0)
        val_loader = create_loader(val_data, batch_size=batch_size)
        test_loader = create_loader(test_data, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def discretize_features(self, movies_df, num_bins=5):
        features_to_discretize = ["budget", "revenue", "vote_average", "vote_count"]
        discretized_features = {}
        feature_mappings = {}

        for feature in features_to_discretize:
            if feature in ["vote_average", "vote_count"]:
                bins = pd.qcut(movies_df[feature], q=num_bins, duplicates="drop")
            else:
                bins = pd.qcut(
                    np.log1p(movies_df[feature]), q=num_bins, duplicates="drop"
                )

            # Get unique bin labels
            unique_bins = bins.cat.categories

            labels = [f"{feature}_{i}" for i in range(len(unique_bins))]
            discretized_features[feature] = bins.cat.codes
            feature_mappings[feature] = labels

        for feature, discretized in discretized_features.items():
            movies_df[f"{feature}_bin"] = discretized

        return movies_df, feature_mappings

    def add_feature_nodes_and_edges(
        self, edge_index, edge_type, movies_df, feature_mappings, num_existing_nodes
    ):
        new_edges = []
        new_edge_types = []
        feature_node_offset = num_existing_nodes
        feature_to_edge_type = {
            "budget": 4,
            "revenue": 5,
            "vote_average": 6,
            "vote_count": 7,
        }
        actual_feature_nodes = 0

        for feature, mapping in feature_mappings.items():
            unique_bin_values = movies_df[f"{feature}_bin"].unique()
            for bin_value in unique_bin_values:
                feature_node_idx = feature_node_offset + actual_feature_nodes
                movie_indices = movies_df.index[
                    movies_df[f"{feature}_bin"] == bin_value
                ].tolist()
                new_edges.extend(
                    [(movie_idx, feature_node_idx) for movie_idx in movie_indices]
                )
                new_edge_types.extend(
                    [feature_to_edge_type[feature]] * len(movie_indices)
                )
                actual_feature_nodes += 1

        combined_edge_index = torch.cat(
            [edge_index, torch.tensor(new_edges, dtype=torch.long).t()], dim=1
        )
        combined_edge_type = torch.cat(
            [edge_type, torch.tensor(new_edge_types, dtype=torch.long)]
        )

        total_nodes = num_existing_nodes + actual_feature_nodes
        self.num_feature_nodes = actual_feature_nodes  # Store this for later use

        return combined_edge_index, combined_edge_type, total_nodes

    def generate_mappings(self):
        movies_df, _ = self.read_tmdb_movies()
        movies_df = movies_df.dropna(
            subset=["directorId", "budget", "revenue", "vote_average", "vote_count"]
        )
        movies_df, self.feature_mappings = self.discretize_features(movies_df)

        genres = movies_df["genres"].str.get_dummies("|")
        all_genres = genres.columns.tolist()

        self.movie_mapping = {
            movie_id: idx for idx, movie_id in enumerate(movies_df["movieId"].unique())
        }
        self.director_mapping = {
            director_id: idx + len(self.movie_mapping)
            for idx, director_id in enumerate(movies_df["directorId"].unique())
        }
        self.genre_mapping = {
            genre: idx + len(self.movie_mapping) + len(self.director_mapping)
            for idx, genre in enumerate(all_genres)
        }

        # Create reverse mappings
        self.reverse_movie_mapping = {v: k for k, v in self.movie_mapping.items()}
        self.reverse_director_mapping = {v: k for k, v in self.director_mapping.items()}
        self.reverse_genre_mapping = {v: k for k, v in self.genre_mapping.items()}

        # Store movie titles
        self.movie_titles = dict(zip(movies_df["movieId"], movies_df["title"]))
        self.director_names = dict(
            zip(movies_df["directorId"], movies_df["directorName"])
        )

        return movies_df, genres, all_genres

    def get_movie_title(self, movie_id):
        return self.movie_titles.get(movie_id, f"Unknown Movie (ID: {movie_id})")

    def get_director_name(self, director_id):
        return self.director_names.get(
            director_id, f"Unknown Director (ID: {director_id})"
        )

    def process_data_kge(self):
        movies_df, genres, all_genres = self.generate_mappings()

        # Create director-movie edges
        director_indices = torch.tensor(
            [self.director_mapping[director] for director in movies_df["directorId"]],
            dtype=torch.long,
        )
        director_movie_edge_index = torch.stack(
            [
                director_indices,
                torch.tensor(
                    [self.movie_mapping[movie] for movie in movies_df["movieId"]],
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        director_movie_edge_type = torch.full((director_movie_edge_index.size(1),), 2)

        # Create movie-genre edges
        movie_genre_edges = []
        for movie_id, movie_genres in zip(movies_df["movieId"], genres.values):
            for genre, has_genre in zip(all_genres, movie_genres):
                if has_genre:
                    movie_genre_edges.append(
                        (self.movie_mapping[movie_id], self.genre_mapping[genre])
                    )
        movie_genre_edge_index = torch.tensor(movie_genre_edges, dtype=torch.long).t()
        movie_genre_edge_type = torch.full((movie_genre_edge_index.size(1),), 3)

        # Combine edges
        combined_edge_index = torch.cat(
            [director_movie_edge_index, movie_genre_edge_index], dim=1
        )
        combined_edge_type = torch.cat(
            [director_movie_edge_type, movie_genre_edge_type]
        )

        # Add feature nodes and edges
        num_existing_nodes = (
            len(self.movie_mapping)
            + len(self.director_mapping)
            + len(self.genre_mapping)
        )
        combined_edge_index, combined_edge_type, total_nodes = (
            self.add_feature_nodes_and_edges(
                combined_edge_index,
                combined_edge_type,
                movies_df,
                self.feature_mappings,
                num_existing_nodes,
            )
        )

        # Create Data object
        data = Data(
            edge_index=combined_edge_index,
            edge_type=combined_edge_type,
            num_nodes=total_nodes,
            num_edge_types=8,  # 0-1 for ratings, 2 for director-movie, 3 for movie-genre, 4-7 for features
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
            num_edge_types=data.num_edge_types,
        )

        val_data = Data(
            edge_index=data.edge_index[:, perm[train_edges : train_edges + val_edges]],
            edge_type=data.edge_type[perm[train_edges : train_edges + val_edges]],
            num_nodes=data.num_nodes,
            num_edge_types=data.num_edge_types,
        )

        test_data = Data(
            edge_index=data.edge_index[:, perm[train_edges + val_edges :]],
            edge_type=data.edge_type[perm[train_edges + val_edges :]],
            num_nodes=data.num_nodes,
            num_edge_types=data.num_edge_types,
        )

        return train_data, val_data, test_data

    def get_mappings(self):
        if (
            self.movie_mapping is None
            or self.director_mapping is None
            or self.genre_mapping is None
        ):
            self.generate_mappings()
        return {
            "movie_mapping": self.movie_mapping,
            "director_mapping": self.director_mapping,
            "genre_mapping": self.genre_mapping,
        }


def get_movie_data_kge(small=False):
    processor = MovieDataProcessor(small)
    return processor.prepare_data_kge()


def get_sageconv_movie_data_and_loaders(neighbors=[5, 5], batch_size=2048, small=False):
    processor = MovieDataProcessor(small)
    movies_df, ratings_df = processor.read_tmdb_movies()
    (
        edge_index_user_to_movie,
        edge_index_director_to_movie,
        edge_index_genre_to_movie,
        train_ratings_df,
        unique_user_id,
        unique_movie_id,
        unique_director_id,
        unique_genre_id,
    ) = processor.process_data(ratings_df, movies_df)
    train_data = processor.create_hetero_data(
        train_ratings_df,
        movies_df,
        edge_index_user_to_movie,
        edge_index_director_to_movie,
        edge_index_genre_to_movie,
        unique_user_id,
        unique_movie_id,
        unique_director_id,
        unique_genre_id,
    )
    train_loader, val_loader, test_loader = processor.generate_loaders(
        train_data, batch_size=batch_size, num_neighbors=neighbors
    )
    return train_data, train_loader, val_loader, test_loader
