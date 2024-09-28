import os
from datetime import datetime

import pandas as pd
import torch

from data import get_gnn_inference_data
from model import inference, load_model


def run_gnn_inference(
    inference_path="inference_data.csv", small=False, model_path=None
):
    if small and model_path is None:
        model_path = "./model/model_sage_20240928_1732_acc0.7090.pt"
    loader = get_gnn_inference_data(inference_path, small)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(model_path, device)
    all_preds = inference(model, loader, device)
    evolve_kg(all_preds, small=small)


def evolve_kg(all_preds, small=False):
    path = "./data"
    if small:
        path = "./data/small"

    original_ratings = pd.read_csv(f"{path}/inference_data.csv")

    # Create a new DataFrame for potential ratings
    potential_ratings = original_ratings.copy()

    # Convert list of CUDA tensors to a list of Python integers
    all_preds_list = [tensor.item() for tensor in all_preds]

    # Ensure the length matches
    if len(all_preds_list) != len(potential_ratings):
        raise ValueError(
            f"Length mismatch: all_preds ({len(all_preds_list)}) vs potential_ratings ({len(potential_ratings)})"
        )

    potential_ratings["rating"] = all_preds_list
    potential_ratings["timestamp"] = int(datetime.now().timestamp())

    # Define the output file path
    output_file = f"{path}/potential_ratings.csv"

    # Check if the file exists
    if not os.path.exists(output_file):
        # If the file doesn't exist, create it and save the data
        potential_ratings.to_csv(output_file, index=False)
        print(f"Created new file: {output_file}")
    else:
        # If the file exists, append the new data without writing the header
        potential_ratings.to_csv(output_file, mode="a", header=False, index=False)
        print(f"Appended to existing file: {output_file}")

    print(potential_ratings)
