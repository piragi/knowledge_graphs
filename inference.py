import pandas as pd
import torch
from torch_geometric.loader import LinkNeighborLoader

from data import get_gnn_inference_data
from model import inference, load_model


def run_inference():
    inference_path = "inference_data.csv"
    model_path = "./model/model_gat_20240923_2305_acc0.7529.pt"
    loader = get_gnn_inference_data(inference_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(model_path, device)
    all_preds = inference(model, loader, device)
    print(all_preds)
    return all_preds


run_inference()
