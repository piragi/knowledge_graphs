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
    parser.add_argument("--small", action="store_true", help="Use small dataset")
    parser.add_argument(
        "--model",
        choices=["sage", "gat", "transe", "rotate", "complex"],
        default="gat",
        help="Choose the model type. For GNN: 'sage' for GraphSAGE, 'gat' for Graph Attention Network. "
        "For KGE: 'transe' for TransE, 'rotate' for RotatE, 'complex' for ComplEx. (default: gat)",
    )
    args = parser.parse_args()

    if args.command == "gnn_training":
        run_training(model_type=args.model, small=args.small)
    elif args.command == "gnn_inference":
        run_gnn_inference(small=args.small)
    elif args.command == "kge_training":
        run_kge_train(model_type=args.model, small=args.small)
    elif args.command == "kge_inference":
        run_kge_inference(small=args.small)


if __name__ == "__main__":
    main()
