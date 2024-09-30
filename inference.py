import os
from datetime import datetime

import pandas as pd
import torch

from data import MovieDataProcessor, get_gnn_inference_data
from model import inference, load_model


def create_inference_data_for_user(user_id, small=False):
    processor = MovieDataProcessor(small)
    movies_df, ratings_df = processor.read_tmdb_movies()

    # Get all movies
    all_movies = set(movies_df["movieId"])

    # Get movies rated by the user
    user_rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])

    # Get movies not rated by the user
    unrated_movies = all_movies - user_rated_movies

    # Create inference data
    inference_data = pd.DataFrame(
        {
            "userId": [user_id] * len(unrated_movies),
            "movieId": list(unrated_movies),
            "rating": [0.0] * len(unrated_movies),
            "timestamp": [int(datetime.now().timestamp())] * len(unrated_movies),
        }
    )

    # Save inference data to a temporary file
    path = "./data/small" if small else "./data"
    temp_file = os.path.join(path, "temp_inference_data.csv")
    inference_data.to_csv(temp_file, index=False)

    return temp_file


def run_gnn_inference(inference_user=9, small=False, model_path=None):
    if small and model_path is None:
        model_path = "./model/model_gat_20240928_2102_acc0.5789.pt"

    # Create inference data for the specified user
    inference_path = create_inference_data_for_user(inference_user, small)

    loader = get_gnn_inference_data("temp_inference_data.csv", small)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(model_path, device)
    all_preds = inference(model, loader, device)
    evolve_kg(all_preds, inference_user, inference_path, small=small)

    # Remove the temporary file
    os.remove(inference_path)


def evolve_kg(all_preds, inference_user, inference_path, small=False):
    path = "./data/small" if small else "./data"

    # Read the temporary inference data
    original_ratings = pd.read_csv(inference_path)

    # Create a new DataFrame for potential ratings
    potential_ratings = original_ratings.copy()

    # Convert tensor of predictions to a list of Python integers
    all_preds_list = all_preds.cpu().tolist()

    # Ensure the length matches
    if len(all_preds_list) != len(potential_ratings):
        raise ValueError(
            f"Length mismatch: all_preds ({len(all_preds_list)}) vs potential_ratings ({len(potential_ratings)})"
        )

    potential_ratings["rating"] = all_preds_list
    potential_ratings["timestamp"] = int(datetime.now().timestamp())

    # Filter out ratings that are 0
    filtered_ratings = potential_ratings[potential_ratings["rating"] > 0]

    # Define the output file path
    output_file = os.path.join(path, f"potential_ratings_user_{inference_user}.csv")

    # Save the filtered data
    filtered_ratings.to_csv(output_file, index=False)
    print(f"Created new file: {output_file}")

    print(f"Total predictions: {len(potential_ratings)}")
    print(f"Predictions > 0: {len(filtered_ratings)}")
    print(filtered_ratings)
