#!/usr/bin/env python3
"""
Compare LR, FM, and DeepFM models on Criteo dataset.

Trains all three models and compares:
- Training time
- Model size
- AUC/LogLoss
- Inference latency
- Throughput

Usage:
    python scripts/criteo/compare_models.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from liteads.ml_engine.features.builder import FeatureBuilder
from liteads.ml_engine.models.deepfm import DeepFM
from liteads.ml_engine.models.lr import FactorizationMachineLR, LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CTR models")
    parser.add_argument("--data-dir", type=str, default="data/criteo")
    parser.add_argument("--config", type=str, default="configs/criteo_features_config.yaml")
    parser.add_argument("--output-dir", type=str, default="models/criteo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_data(data_dir: str, config_path: str, device: torch.device):
    """Load and prepare data."""
    data_dir = Path(data_dir)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Extract labels (column might be 'label' or 'click')
    label_col = "label" if "label" in train_df.columns else "click"
    train_labels = train_df[label_col].values
    val_labels = val_df[label_col].values
    test_labels = test_df[label_col].values

    # Convert to feature dicts
    feature_cols = [c for c in train_df.columns if c not in ("label", "click")]
    train_data = train_df[feature_cols].to_dict("records")
    val_data = val_df[feature_cols].to_dict("records")
    test_data = test_df[feature_cols].to_dict("records")

    # Build features
    feature_builder = FeatureBuilder(config_path=config_path, device=str(device))
    feature_builder.fit(train_data)

    train_inputs = feature_builder.transform(train_data, train_labels)
    val_inputs = feature_builder.transform(val_data, val_labels)
    test_inputs = feature_builder.transform(test_data, test_labels)

    return feature_builder, train_inputs, val_inputs, test_inputs


def create_model(model_type: str, model_config: dict, device: torch.device) -> nn.Module:
    """Create model by type."""
    if model_type == "lr":
        model = LogisticRegression(
            sparse_feature_dims=model_config["sparse_feature_dims"],
            dense_feature_dim=model_config["dense_feature_dim"],
            l2_reg=model_config.get("l2_reg_embedding", 0.0001),
        )
    elif model_type == "fm":
        model = FactorizationMachineLR(
            sparse_feature_dims=model_config["sparse_feature_dims"],
            dense_feature_dim=model_config["dense_feature_dim"],
            embedding_dim=model_config.get("fm_k", 8),
            l2_reg=model_config.get("l2_reg_embedding", 0.0001),
        )
    elif model_type == "deepfm":
        model = DeepFM(
            sparse_feature_dims=model_config["sparse_feature_dims"],
            sparse_embedding_dims=model_config.get("sparse_embedding_dims", 8),
            dense_feature_dim=model_config["dense_feature_dim"],
            fm_k=model_config.get("fm_k", 8),
            dnn_hidden_units=model_config.get("dnn_hidden_units", [256, 128, 64]),
            dnn_dropout=model_config.get("dnn_dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_inputs,
    val_inputs,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train a model and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_sparse = train_inputs.sparse_features
    train_dense = train_inputs.dense_features
    train_labels = train_inputs.labels

    val_sparse = val_inputs.sparse_features
    val_dense = val_inputs.dense_features
    val_labels = val_inputs.labels

    n_train = len(train_labels)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_auc = 0.0
    best_val_loss = float("inf")
    train_losses = []
    val_aucs = []
    val_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle
        perm = torch.randperm(n_train, device=device)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = perm[start_idx:end_idx]

            sparse_batch = train_sparse[batch_indices]
            dense_batch = train_dense[batch_indices]
            labels_batch = train_labels[batch_indices]

            optimizer.zero_grad()
            outputs = model(sparse_batch, dense_batch)
            pred = outputs["ctr"]
            loss = criterion(pred, labels_batch)

            # Add regularization if available
            if hasattr(model, "get_regularization_loss"):
                loss = loss + model.get_regularization_loss()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_sparse, val_dense)
            val_pred = val_outputs["ctr"].cpu().numpy()
            val_true = val_labels.cpu().numpy()

            val_auc = roc_auc_score(val_true, val_pred)
            val_loss = log_loss(val_true, val_pred)

            val_aucs.append(val_auc)
            val_losses.append(val_loss)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_loss = val_loss

        print(f"  Epoch {epoch + 1}/{epochs} - loss: {avg_train_loss:.4f} - val_auc: {val_auc:.4f}")

    training_time = time.time() - start_time

    return {
        "training_time": training_time,
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_aucs": val_aucs,
        "val_losses": val_losses,
    }


def evaluate_model(model: nn.Module, test_inputs, device: torch.device) -> dict:
    """Evaluate model on test set and measure inference speed."""
    model.eval()

    test_sparse = test_inputs.sparse_features
    test_dense = test_inputs.dense_features
    test_labels = test_inputs.labels

    # Single batch inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model(test_sparse, test_dense)
        inference_time = time.time() - start_time

        pred = outputs["ctr"].cpu().numpy()
        true = test_labels.cpu().numpy()

    test_auc = roc_auc_score(true, pred)
    test_logloss = log_loss(true, pred)

    # Measure latency with multiple runs
    latencies = []
    batch_size = 32
    n_samples = len(test_labels)

    with torch.no_grad():
        for i in range(0, min(n_samples, 1000), batch_size):
            batch_sparse = test_sparse[i:i+batch_size]
            batch_dense = test_dense[i:i+batch_size]

            start = time.time()
            _ = model(batch_sparse, batch_dense)
            latencies.append((time.time() - start) * 1000)  # ms

    return {
        "test_auc": test_auc,
        "test_logloss": test_logloss,
        "total_inference_time": inference_time,
        "throughput": n_samples / inference_time,
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
    }


def save_model(model: nn.Module, model_type: str, model_config: dict, output_dir: Path):
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "model_config": model_config,
    }
    path = output_dir / f"{model_type}_criteo.pt"
    torch.save(checkpoint, path)

    # Get file size
    size_mb = path.stat().st_size / (1024 * 1024)
    return size_mb


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LR vs FM vs DeepFM Model Comparison")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Load data
    print("\n" + "-" * 70)
    print("Loading data...")
    print("-" * 70)
    feature_builder, train_inputs, val_inputs, test_inputs = load_data(
        args.data_dir, args.config, device
    )
    model_config = feature_builder.get_model_config()

    print(f"  Training samples: {len(train_inputs.labels):,}")
    print(f"  Validation samples: {len(val_inputs.labels):,}")
    print(f"  Test samples: {len(test_inputs.labels):,}")
    print(f"  Sparse features: {len(model_config['sparse_feature_dims'])}")
    print(f"  Dense features: {model_config['dense_feature_dim']}")

    # Save feature builder
    feature_builder.save(str(output_dir / "feature_builder.pkl"))

    # Train and evaluate each model
    model_types = ["lr", "fm", "deepfm"]
    results = {}

    for model_type in model_types:
        print("\n" + "=" * 70)
        print(f"Training {model_type.upper()} Model")
        print("=" * 70)

        # Create model
        model = create_model(model_type, model_config, device)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")

        # Train
        train_metrics = train_model(
            model, train_inputs, val_inputs,
            args.epochs, args.batch_size, args.lr, device
        )

        # Evaluate
        eval_metrics = evaluate_model(model, test_inputs, device)

        # Save model
        model_size = save_model(model, model_type, model_config, output_dir)

        # Collect results
        results[model_type] = {
            "parameters": n_params,
            "model_size_mb": model_size,
            **train_metrics,
            **eval_metrics,
        }

        print(f"\n  Results:")
        print(f"    Training time: {train_metrics['training_time']:.2f}s")
        print(f"    Best val AUC: {train_metrics['best_val_auc']:.4f}")
        print(f"    Test AUC: {eval_metrics['test_auc']:.4f}")
        print(f"    Test LogLoss: {eval_metrics['test_logloss']:.4f}")
        print(f"    Throughput: {eval_metrics['throughput']:.0f} samples/sec")
        print(f"    Avg latency: {eval_metrics['avg_latency_ms']:.2f}ms")
        print(f"    P99 latency: {eval_metrics['p99_latency_ms']:.2f}ms")
        print(f"    Model size: {model_size:.2f} MB")

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    headers = ["Metric", "LR", "FM", "DeepFM"]
    rows = [
        ("Parameters", *[f"{results[m]['parameters']:,}" for m in model_types]),
        ("Model Size (MB)", *[f"{results[m]['model_size_mb']:.2f}" for m in model_types]),
        ("Training Time (s)", *[f"{results[m]['training_time']:.1f}" for m in model_types]),
        ("Val AUC", *[f"{results[m]['best_val_auc']:.4f}" for m in model_types]),
        ("Test AUC", *[f"{results[m]['test_auc']:.4f}" for m in model_types]),
        ("Test LogLoss", *[f"{results[m]['test_logloss']:.4f}" for m in model_types]),
        ("Throughput (samples/s)", *[f"{results[m]['throughput']:.0f}" for m in model_types]),
        ("Avg Latency (ms)", *[f"{results[m]['avg_latency_ms']:.2f}" for m in model_types]),
        ("P99 Latency (ms)", *[f"{results[m]['p99_latency_ms']:.2f}" for m in model_types]),
    ]

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(4)]

    # Print header
    header_line = "│".join(h.center(col_widths[i]) for i, h in enumerate(headers))
    print(f"│{header_line}│")
    print("├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Print rows
    for row in rows:
        row_line = "│".join(str(cell).center(col_widths[i]) for i, cell in enumerate(row))
        print(f"│{row_line}│")

    print("└" + "┴".join("─" * w for w in col_widths) + "┘")

    # Find best model
    best_auc_model = max(model_types, key=lambda m: results[m]["test_auc"])
    best_speed_model = max(model_types, key=lambda m: results[m]["throughput"])

    print(f"\n🏆 Best AUC: {best_auc_model.upper()} ({results[best_auc_model]['test_auc']:.4f})")
    print(f"⚡ Best Throughput: {best_speed_model.upper()} ({results[best_speed_model]['throughput']:.0f} samples/s)")

    # Save results
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results_path = output_dir / "model_comparison.json"
    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nModel files saved:")
    for model_type in model_types:
        print(f"  - {output_dir}/{model_type}_criteo.pt")


if __name__ == "__main__":
    main()
