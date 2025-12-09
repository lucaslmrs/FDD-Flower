# FDD-Flower: Federated Bearing Fault Detection

Federated Learning application for bearing fault detection using Flower and PyTorch.

## Features

- 8-class bearing fault classification using multiclass logistic regression
- Non-IID data partitioning with Dirichlet distribution
- Automated class distribution visualization
- W&B integration for experiment tracking
- Configurable parameters via `pyproject.toml`

## Setup

```bash
pip install -e .
```

## Run

```bash
flwr run .
```

Configuration options in `pyproject.toml`:
- `num-server-rounds`: Number of federated rounds
- `dirichlet-alpha`: Data heterogeneity (higher = more IID)
- `local-epochs`: Training epochs per client
- `num-supernodes`: Number of federated clients

## Output

- Model checkpoints: `artifacts/global_model_round_*.pt`
- Class distribution plots: `artifacts/class_distribution_*.png`
- Training metrics: `results.json` and W&B dashboard