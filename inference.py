import torch

from data import get_gnn_inference_data
from model import inference, load_model


def run_inference():
    inference_path = "inference_data.csv"
    model_path = "./model/model_gat_20240925_2109_acc0.7324.pt"
    loader = get_gnn_inference_data(inference_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(model_path, device)
    all_preds = inference(model, loader, device)
    print(all_preds)
    return all_preds


run_inference()
