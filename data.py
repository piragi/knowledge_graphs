import ast
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader


class MovieDataProcessor:
    def __init__(self, small=False):
        self.small = small
        self.base_url = "https://files.grouplens.org/datasets/movielens/"
        self.zip_file = "ml-latest-small.zip" if small else "ml-latest.zip"
        self.folder_name = "ml-latest-small" if small else "ml-latest"
        self.path = "./data/small" if small else "./data"
        self.processed_ratings_path = os.path.join(self.path, "ratings_processed.csv")
        self.processed_movies_path = os.path.join(self.path, "movies_processed.csv")
        self.movie_mapping = {}
        self.director_mapping = {}
        self.genre_mapping = {}
        self.reverse_movie_mapping = {}
        self.reverse_director_mapping = {}
        self.reverse_genre_mapping = {}
        self.movie_titles = {}
        self.director_names = {}

    def ensure_data_availability(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not (
            os.path.exists(self.processed_ratings_path)
            and os.path.exists(self.processed_movies_path)
        ):
            self.download_and_extract_data()

    def download_and_extract_data(self):
        zip_path = os.path.join(self.path, self.zip_file)

        # Download the zip file
        url = self.base_url + self.zip_file
        print(f"Downloading {url}...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        print(f"Extracting {self.zip_file}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.path)

        # Move files from the extracted folder to the data directory
        extracted_folder = os.path.join(self.path, self.folder_name)
        for file in os.listdir(extracted_folder):
            shutil.move(os.path.join(extracted_folder, file), self.path)

        # Remove the zip file and the empty extracted folder
        os.remove(zip_path)
        os.rmdir(extracted_folder)

        print("Data downloaded and extracted successfully.")

    def read_tmdb_movies(self, gnn_inference_path=None):
        self.ensure_data_availability()

        # Check if processed files exist
        if (
            os.path.exists(self.processed_ratings_path)
            and os.path.exists(self.processed_movies_path)
            and gnn_inference_path is None
        ):
            print("Loading processed files...")
            ratings_df = pd.read_csv(self.processed_ratings_path)
            movies_df = pd.read_csv(self.processed_movies_path)
            return movies_df, ratings_df

        print("Processing files...")
        if gnn_inference_path is not None:
            ratings_path = gnn_inference_path
        else:
            ratings_path = "ratings.csv"

        movies_df = self.process_movies_df()
        ratings_df = pd.read_csv(os.path.join(self.path, ratings_path))
        ratings_df = ratings_df[ratings_df["movieId"].isin(movies_df["movieId"])]
        ratings_df = pd.merge(
            ratings_df, movies_df["movieId"], on="movieId", how="left"
        )
        movies_df = movies_df[movies_df["movieId"].isin(ratings_df["movieId"])]

        # Merge director information with movies_df
        credits_df = self.process_credits_df()
        movies_df = pd.merge(
            movies_df,
            credits_df[["movie_id", "directorId", "directorName"]],
            left_on="id",
            right_on="movie_id",
            how="left",
        )

        # Save processed files
        if gnn_inference_path is None:
            movies_df.to_csv(self.processed_movies_path, index=False)
            ratings_df.to_csv(self.processed_ratings_path, index=False)

        return movies_df, ratings_df

    def process_movies_df(self):
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

        # link internal id to tmdb id
        links_df = pd.read_csv(f"{self.path}/links.csv", usecols=["movieId", "tmdbId"])
        movies_linked_df = pd.merge(
            movies_df, links_df, left_on="id", right_on="tmdbId", how="inner"
        )
        movies_info_df = pd.read_csv(f"{self.path}/movies.csv")
        return pd.merge(movies_linked_df, movies_info_df, on="movieId", how="inner")

    def process_credits_df(self):
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
        return credits_df

    @staticmethod
    def standardize(x):
        return (x - x.mean()) / x.std()

    def determine_preferred_genre(self, ratings_df, movies_df, unique_mappings):
        merged_df = ratings_df.merge(movies_df[["movieId", "genres"]], on="movieId")
        merged_df["genre"] = merged_df["genres"].str.split("|")
        merged_df = merged_df.explode("genre")
        genre_counts = (
            merged_df.groupby(["userId", "genre"]).size().unstack(fill_value=0)
        )
        genre_proportions = genre_counts.div(genre_counts.sum(axis=1), axis=0)

        # Determine the preferred genre (where proportion > 0.5, if any)
        preferred_genre = genre_proportions.apply(
            lambda row: row.index[row > 0.5][0] if any(row > 0.5) else "None", axis=1
        )
        all_users = pd.DataFrame(unique_mappings["user"])
        complete_preferred_genre = all_users.merge(
            preferred_genre.reset_index(),
            left_on="userId",
            right_on="userId",
            how="left",
        ).fillna("None")

        # Sort by mappedId to ensure correct order
        complete_preferred_genre = complete_preferred_genre.sort_values("mappedId")

        genre_to_int = {
            genre: i
            for i, genre in enumerate(genre_proportions.columns.tolist() + ["None"])
        }
        preferred_genre_int = complete_preferred_genre[0].map(genre_to_int)
        preferred_genre_tensor = torch.tensor(
            preferred_genre_int.values, dtype=torch.float
        ).unsqueeze(1)

        return preferred_genre_tensor, genre_to_int

    def read_movie_features(self, movies_df, ratings_df):
        movies_df = movies_df[movies_df["movieId"].isin(ratings_df["movieId"])]
        numerical_features = ["vote_average", "revenue", "budget", "vote_count"]
        movies_df["vote_average"] = movies_df["vote_average"].fillna(5.0)

        # Convert to numpy, standardize, and create tensors
        num_features = movies_df[numerical_features].values.astype(float)
        num_features_standardized = self.standardize(num_features)
        num_tensors = torch.from_numpy(num_features_standardized).float()
        return num_tensors

    def create_unique_id_mapping(self, df, column_name):
        unique_values = df[column_name].unique()
        return pd.DataFrame(
            {
                column_name: unique_values,
                "mappedId": pd.RangeIndex(len(unique_values)),
            }
        )

    def create_edge_index(self, df1, col1, df2, col2, mapping1, mapping2):
        merged = pd.merge(df1[col1], mapping1, on=col1, how="left")
        merged = pd.merge(merged, df2[col2], left_index=True, right_index=True)
        merged = pd.merge(merged, mapping2, on=col2, how="left")
        return torch.stack(
            [
                torch.from_numpy(merged["mappedId_x"].values),
                torch.from_numpy(merged["mappedId_y"].values),
            ],
            dim=0,
        )

    def process_data(self, ratings_df, movies_df):
        ratings_df = ratings_df.dropna(subset=["rating"])

        # Process genres
        genres = movies_df["genres"].str.get_dummies("|")
        genres["movieId"] = movies_df["movieId"]
        genres_long = genres.melt(
            id_vars=["movieId"], var_name="genreName", value_name="hasGenre"
        )
        genres_long = genres_long[genres_long["hasGenre"] == 1].drop("hasGenre", axis=1)

        # Create unique mappings
        unique_mappings = {
            "genre": self.create_unique_id_mapping(
                pd.DataFrame({"genreName": genres.columns[:-1]}), "genreName"
            ),
            "director": self.create_unique_id_mapping(movies_df, "directorId"),
            "movie": self.create_unique_id_mapping(movies_df, "movieId"),
            "user": self.create_unique_id_mapping(ratings_df, "userId"),
        }

        # Create edge indices
        edge_index_genre_to_movie = self.create_edge_index(
            genres_long,
            "genreName",
            genres_long,
            "movieId",
            unique_mappings["genre"],
            unique_mappings["movie"],
        )

        edge_index_director_to_movie = self.create_edge_index(
            movies_df,
            "directorId",
            movies_df,
            "movieId",
            unique_mappings["director"],
            unique_mappings["movie"],
        )

        edge_index_user_to_movie = self.create_edge_index(
            ratings_df,
            "userId",
            ratings_df,
            "movieId",
            unique_mappings["user"],
            unique_mappings["movie"],
        )

        # Group ratings
        group_ratings = lambda rating: (
            min(5, max(0, int(rating))) if pd.notna(rating) else 0
        )
        ratings_df["grouped_rating"] = ratings_df["rating"].apply(group_ratings)

        edge_indices = {
            "user_to_movie": edge_index_user_to_movie,
            "director_to_movie": edge_index_director_to_movie,
            "genre_to_movie": edge_index_genre_to_movie,
        }

        return edge_indices, ratings_df, unique_mappings

    def create_hetero_data(
        self,
        ratings_df,
        movies_df,
        edge_index,
        unique_mappings,
    ):
        data = HeteroData()
        data["user"].node_id = torch.arange(len(unique_mappings["user"]))
        data["movie"].node_id = torch.arange(len(unique_mappings["movie"]))
        data["director"].node_id = torch.arange(len(unique_mappings["director"]))
        data["genre"].node_id = torch.arange(len(unique_mappings["genre"]))
        data["movie"].x = self.read_movie_features(movies_df, ratings_df)
        data["user", "rates", "movie"].edge_index = edge_index["user_to_movie"]
        data["director", "directs", "movie"].edge_index = edge_index[
            "director_to_movie"
        ]
        data["genre", "is_genre", "movie"].edge_index = edge_index["genre_to_movie"]
        grouped_ratings = torch.from_numpy(ratings_df["grouped_rating"].values).long()
        one_hot_ratings = F.one_hot(grouped_ratings, num_classes=6).float()
        data["user", "rates", "movie"].edge_attr = one_hot_ratings
        data["user", "rates", "movie"].edge_label = grouped_ratings
        data = T.ToUndirected()(data)
        return data

    def clean_ratings_df(self, ratings_df, val_data, test_data, unique_mappings):
        def edge_index_to_df(edge_index, user_mapping, movie_mapping):
            user_ids = (
                user_mapping.set_index("mappedId")
                .loc[edge_index[0].numpy(), "userId"]
                .values
            )
            movie_ids = (
                movie_mapping.set_index("mappedId")
                .loc[edge_index[1].numpy(), "movieId"]
                .values
            )
            return pd.DataFrame({"userId": user_ids, "movieId": movie_ids})

        # Combine val and test edges
        val_edges = edge_index_to_df(
            val_data["user", "rates", "movie"].edge_label_index,
            unique_mappings["user"],
            unique_mappings["movie"],
        )
        test_edges = edge_index_to_df(
            test_data["user", "rates", "movie"].edge_label_index,
            unique_mappings["user"],
            unique_mappings["movie"],
        )

        edges_to_remove = pd.concat([val_edges, test_edges], ignore_index=True)

        # Use vectorized operations for filtering
        ratings_array = ratings_df[["userId", "movieId"]].values
        mask = ~np.isin(ratings_array, edges_to_remove.values).all(axis=1)
        cleaned_ratings_df = ratings_df[mask]

        return cleaned_ratings_df

    def generate_loaders(
        self,
        train_data,
        ratings_df,
        unique_mappings,
        movies_df,
        batch_size=2048,
        num_neighbors=[5, 5],
    ):
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

        # generate user features only on training data, hence clean ratings_df
        cleaned_ratings_df = self.clean_ratings_df(
            ratings_df, val_data, test_data, unique_mappings
        )
        preferred_genre_tensor, _ = self.determine_preferred_genre(
            cleaned_ratings_df, movies_df, unique_mappings
        )
        train_data["user"].x = preferred_genre_tensor
        val_data["user"].x = preferred_genre_tensor
        test_data["user"].x = preferred_genre_tensor

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
        )
        train_loader = create_loader(train_data, shuffle=True, neg_sampling_ratio=2.0)
        val_loader = create_loader(val_data, batch_size=batch_size)
        test_loader = create_loader(test_data, batch_size=batch_size)
        return train_loader, val_loader, test_loader, train_data

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

        self.reverse_movie_mapping = {v: k for k, v in self.movie_mapping.items()}
        self.reverse_director_mapping = {v: k for k, v in self.director_mapping.items()}
        self.reverse_genre_mapping = {v: k for k, v in self.genre_mapping.items()}
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
        edge_index,
        train_ratings_df,
        unique_mappings,
    ) = processor.process_data(ratings_df, movies_df)
    data = processor.create_hetero_data(
        train_ratings_df,
        movies_df,
        edge_index,
        unique_mappings,
    )
    train_loader, val_loader, test_loader, train_data = processor.generate_loaders(
        data,
        ratings_df,
        unique_mappings,
        movies_df,
        batch_size=batch_size,
        num_neighbors=neighbors,
    )
    return train_data, train_loader, val_loader, test_loader


def get_gnn_inference_data(
    inference_path, small=False, batch_size=1, num_neighbors=[10, 5, 5]
):
    processor = MovieDataProcessor(small)
    movies_df, ratings_df = processor.read_tmdb_movies()
    _, inference_ratings_df = processor.read_tmdb_movies(
        gnn_inference_path=inference_path
    )
    full_ratings_df = pd.concat([ratings_df, inference_ratings_df])
    (
        edge_index,
        train_ratings_df,
        unique_mappings,
    ) = processor.process_data(full_ratings_df, movies_df)
    data = processor.create_hetero_data(
        train_ratings_df,
        movies_df,
        edge_index,
        unique_mappings,
    )
    preferred_genre_tensor, _ = processor.determine_preferred_genre(
        full_ratings_df, movies_df, unique_mappings
    )
    data["user"].x = preferred_genre_tensor

    edge_index = processor.create_edge_index(
        inference_ratings_df,
        "userId",
        inference_ratings_df,
        "movieId",
        unique_mappings["user"],
        unique_mappings["movie"],
    )

    loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=(
            ("user", "rates", "movie"),
            edge_index,
        ),
        edge_label=data["user", "rates", "movie"].edge_label,
    )
    return loader
