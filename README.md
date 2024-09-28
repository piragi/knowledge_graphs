# Movie Recommendation System

This project implements a Movie Recommendation System using Graph Neural Networks (GNN) and Knowledge Graph Embeddings (KGE).

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
python main.py <command> [--small] [--model <model_type>]
```

- `<command>`: Choose from `gnn_training`, `gnn_inference`, `kge_training`, or `kge_inference`
- `--small`: Optional flag to use a smaller dataset
- `--model`: Optional argument to specify the model type

### GNN Training

To train a GNN model:
```
python main.py gnn_training [--small] [--model <sage|gat>]
```
Example:
```
python main.py gnn_training --model gat
```

### GNN Inference

To run inference with a trained GNN model:
```
python main.py gnn_inference [--small]
```

### KGE Training

To train a KGE model:
```
python main.py kge_training [--small] [--model <transe|rotate|complex>]
```
Example:
```
python main.py kge_training --model transe
```


### KGE Inference

To run inference with a trained KGE model:
```
python main.py kge_inference [--small]
```

## Additional Notes

- The `--small` flag can be used with any command to use a smaller dataset, which is useful for testing or when computational resources are limited.
- If no model is specified, the default model (GAT for GNN, TransE for KGE) will be used.
- Make sure you have the necessary datasets in place as specified by the program. The script mentions that no additional datasets need to be manually downloaded, so it should handle data preparation automatically.

Remember to run the script from the directory containing `main.py`. If you encounter any issues, make sure all dependencies are correctly installed and that you have the necessary computational resources (especially if using GPU-accelerated models).
