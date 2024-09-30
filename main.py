import argparse

from inference import run_gnn_inference
from kge_inference import run_kge_inference
from kge_model import run_kge_train
from model import run_training


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument(
        "command",
        choices=["gnn_training", "gnn_inference", "kge_training", "kge_inference"],
        help="Command to execute",
    )
    parser.add_argument(
        "--small", action="store_true", help="Use small dataset (only for GNN)"
    )
    parser.add_argument(
        "--model",
        choices=["sage", "gat", "transe", "rotate", "complex"],
        default="gat",
        help="Choose the model type. For GNN: 'sage' for GraphSAGE, 'gat' for Graph Attention Network. "
        "For KGE: 'transe' for TransE, 'rotate' for RotatE, 'complex' for ComplEx. (default: gat)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model file for inference",
    )
    args = parser.parse_args()

    if args.command == "gnn_training":
        run_training(model_type=args.model, small=args.small)
    elif args.command == "gnn_inference":
        run_gnn_inference(small=args.small, model_path=args.model_path)
    elif args.command == "kge_training":
        run_kge_train(model_type=args.model)
    elif args.command == "kge_inference":
        run_kge_inference(model_path=args.model_path)


if __name__ == "__main__":
    main()
