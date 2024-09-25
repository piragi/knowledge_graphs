import csv
from typing import Dict, List, Tuple

import numpy as np
import torch

from data import MovieDataProcessor
from kge_model import MovieRecommendationModel


class KGEMovieRecommender:
    def __init__(
        self,
        kge_model,
        movie_mapping: Dict[int, int],
        director_mapping: Dict[int, int],
        genre_mapping: Dict[str, int],
        node_info: Dict[str, int],
        data_processor: MovieDataProcessor,
    ):
        self.kge_model = kge_model.model
        self.device = next(self.kge_model.parameters()).device
        self.movie_mapping = movie_mapping
        self.director_mapping = director_mapping
        self.genre_mapping = genre_mapping
        self.node_info = node_info
        self.data_processor = data_processor

        self.reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}
        self.reverse_director_mapping = {v: k for k, v in director_mapping.items()}
        self.reverse_genre_mapping = {v: k for k, v in genre_mapping.items()}

        self.num_entities = self.kge_model.node_emb.num_embeddings
        print(f"Number of entities in KGE model: {self.num_entities}")
        print(f"Total nodes from processor: {self.node_info['total_nodes']}")

        if self.num_entities != self.node_info["total_nodes"]:
            raise ValueError(
                "Mismatch between KGE model nodes and processed data nodes"
            )

    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        if entity_id >= self.num_entities:
            raise ValueError(
                f"Entity ID {entity_id} is out of bounds. Max entity ID is {self.num_entities - 1}"
            )
        with torch.no_grad():
            entity_tensor = torch.tensor(entity_id, device=self.device)
            return self.kge_model.node_emb(entity_tensor).cpu().numpy()

    def recommend_movies(
        self,
        liked_movie_id: int,
        liked_director_id: int,
        liked_genre: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        movie_embedding = self.get_entity_embedding(self.movie_mapping[liked_movie_id])
        director_embedding = self.get_entity_embedding(
            self.director_mapping[liked_director_id]
        )
        genre_embedding = self.get_entity_embedding(self.genre_mapping[liked_genre])

        combined_embedding = (
            movie_embedding + director_embedding + genre_embedding
        ) / 3

        # Get all movie embeddings
        movie_embeddings = (
            self.kge_model.node_emb.weight[list(self.movie_mapping.values())]
            .detach()
            .cpu()
            .numpy()
        )

        # Calculate similarities only for movie entities
        similarities = np.dot(movie_embeddings, combined_embedding) / (
            np.linalg.norm(movie_embeddings, axis=1)
            * np.linalg.norm(combined_embedding)
        )

        # Get top k similar movies
        top_indices = np.argsort(similarities)[::-1][
            : top_k + 1
        ]  # +1 to account for the liked movie

        movie_recommendations = [
            (list(self.movie_mapping.keys())[idx], similarities[idx])
            for idx in top_indices
            if list(self.movie_mapping.keys())[idx] != liked_movie_id
        ][:top_k]

        return movie_recommendations

    def find_similar_entities(
        self, entity_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        # This method is now used only for explanation purposes
        all_embeddings = self.kge_model.node_emb.weight.detach().cpu().numpy()
        similarities = np.dot(all_embeddings, entity_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(entity_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]

    def explain_recommendation(
        self,
        movie_id: int,
        liked_movie_id: int,
        liked_director_id: int,
        liked_genre: str,
    ) -> str:
        movie_embedding = self.get_entity_embedding(self.movie_mapping[movie_id])
        liked_movie_embedding = self.get_entity_embedding(
            self.movie_mapping[liked_movie_id]
        )
        director_embedding = self.get_entity_embedding(
            self.director_mapping[liked_director_id]
        )
        genre_embedding = self.get_entity_embedding(self.genre_mapping[liked_genre])

        movie_similarity = np.dot(movie_embedding, liked_movie_embedding) / (
            np.linalg.norm(movie_embedding) * np.linalg.norm(liked_movie_embedding)
        )
        director_similarity = np.dot(movie_embedding, director_embedding) / (
            np.linalg.norm(movie_embedding) * np.linalg.norm(director_embedding)
        )
        genre_similarity = np.dot(movie_embedding, genre_embedding) / (
            np.linalg.norm(movie_embedding) * np.linalg.norm(genre_embedding)
        )

        explanation = f"This movie ({self.data_processor.get_movie_title(movie_id)}) was recommended because:\n"
        explanation += f"- It's {movie_similarity:.2f}% similar to the movie you liked ({self.data_processor.get_movie_title(liked_movie_id)}).\n"
        explanation += f"- It's {director_similarity:.2f}% similar to movies by the director you like ({self.data_processor.get_director_name(liked_director_id)}).\n"
        explanation += (
            f"- It's {genre_similarity:.2f}% similar to the {liked_genre} genre."
        )

        return explanation


def load_coldstart_users(file_path: str) -> List[Dict[str, int]]:
    users = []
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            users.append(
                {
                    "userId": int(row["userId"]),
                    "likedMovieId": int(row["likedMovieId"]),
                    "likedDirectorId": int(row["likedDirectorId"]),
                    "likedGenreId": row["likedGenreId"],
                }
            )
    return users


def main():
    kge_model = MovieRecommendationModel.load_model(
        "./model/model_transe_20240925_1618_acc0.8191.pt"
    )

    processor = MovieDataProcessor(small=False)
    data = processor.process_data_kge()
    mappings = processor.get_mappings()

    movie_mapping = mappings["movie_mapping"]
    director_mapping = mappings["director_mapping"]
    genre_mapping = mappings["genre_mapping"]

    recommender = KGEMovieRecommender(
        kge_model,
        movie_mapping,
        director_mapping,
        genre_mapping,
        node_info,
        processor,
    )

    # Load cold start users
    users = load_coldstart_users("./data/coldstart_users.csv")

    # Generate recommendations for each user
    for user in users:
        print(f"\nRecommendations for user {user['userId']}:")
        recommendations = recommender.recommend_movies(
            user["likedMovieId"], user["likedDirectorId"], user["likedGenreId"]
        )

        for movie_id, score in recommendations:
            print(f"Movie ID: {movie_id} (Similarity score: {score:.2f})")
            print(
                recommender.explain_recommendation(
                    movie_id,
                    user["likedMovieId"],
                    user["likedDirectorId"],
                    user["likedGenreId"],
                )
            )
            print()


if __name__ == "__main__":
    main()
