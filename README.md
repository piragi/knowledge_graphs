# Movie Recommendation System

This project implements a Movie Recommendation System using Graph Neural Networks (GNN) and Knowledge Graph Embeddings (KGE).
Model files are located under "./model/", dataset and inference files are located under "./data/" and "./data/small" respectively.

## Installation

It is recommended to create a new Python environment for this project.

1. Install all requirements:
```
pip install -r requirements.txt
```
2. Install additional binaries:
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+${CUDA}.html
```
Replace `${CUDA}` with either `cpu`, `cu118`, `cu121`, or `cu124` depending on your PyTorch installation.
For MacOS, only `cpu` is supported.

## Usage

The program supports four main commands: GNN training, GNN inference, KGE training, and KGE inference. You can also choose between different models and dataset sizes.

### General Command Structure
```
python main.py <command> [--small] [--model <model_type>] [--model_path <path_to_model>]
```

- `<command>`: Choose from `gnn_training`, `gnn_inference`, `kge_training`, or `kge_inference`
- `--small`: Optional flag to use a smaller dataset
- `--model`: Optional argument to specify the model type
- `--model_path`: Optional argument to specify a custom path to a trained model file (for inference only)

If model_path is not used, selected checkpoints are already in the project and will be chosen as default.

### GNN Training

To train a GNN model:
```
python main.py gnn_training [--small] [--model <sage|gat>]
```
Example:
```
python main.py gnn_training --small --model gat
```

### GNN Inference

To run inference with a trained GNN model:
```
python main.py gnn_inference [--small] [--model_path <path_to_model>]
```

Example:
```
python main.py gnn_inference --small --model_path ./model/my_custom_gnn_model.pt
```

### KGE Training

To train a KGE model:
```
python main.py kge_training [--model <transe|rotate|complex>]
```
Example:
```
python main.py kge_training --model transe
```


### KGE Inference

To run inference with a trained KGE model:
```
python main.py kge_inference [--model_path <path_to_model>]
```

## Contents
- Inference data for the KGE approach to generate recommendations
    - "./data/coldstart_users.csv"
- Pre-generated inference data from a large GNN model (useful when hardware constraints prevent loading the full model)
    - "./data/potential_ratings_user_9.csv"
- Pre-trained model files for small versions of SAGEConv and GAT models
    - "./model/model_transe_20240927_1953_acc0.8165.pt"
    - "./model/model_complex_20240927_2003_acc0.7785.pt"
    - "./model/model_rotate_20240927_1959_acc0.8111.pt"
    - "./model/sage"
    - "./model/gat"
## Additional Notes

- When training the big GNN models and hitting hardware constraints, try reducing the neighborhood parameter in the model.py and after that batch_size.
- The `--small` flag can be used with any GNN commands to use a smaller dataset, which is useful for testing or when computational resources are limited.
- If no model is specified, default models (GAT for GNN, TransE for KGE) will be used.
- For inference, you can specify a custom model path using the `--model_path` argument. If not provided, default paths will be used.
- No dataset downloads are needed, all will be downloaded and extracted automatically.

