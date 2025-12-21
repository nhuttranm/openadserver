#!/usr/bin/env python3
"""
Train DeepFM model on Criteo Click Logs data.

Trains a CTR prediction model and saves:
- Model checkpoint (.pt)
- Feature builder state (.pkl)
- Model config (.json)
- Training metrics (.json)

Usage:
    python scripts/criteo/train_criteo_model.py [--epochs 10]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DeepFM model on Criteo data"
    )
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/criteo",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/criteo_features_config.yaml",
        help="Path to feature configuration",
    )

    # Model arguments
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Default embedding dimension",
    )
    parser.add_argument(
        "--dnn-hidden-units",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="DNN hidden layer sizes",
    )
    parser.add_argument(
        "--dnn-dropout",
        type=float,
        default=0.2,
        help="DNN dropout rate",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Early stopping patience",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/criteo",
        help="Output directory for model files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepfm_criteo",
        help="Model name for saving",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    return parser.parse_args()


def train_model(args: argparse.Namespace) -> None:
    """Train the model."""
    import numpy as np
    import pandas as pd
    import torch

    from liteads.ml_engine.data import AdDataModule
    from liteads.ml_engine.features import FeatureBuilder
    from liteads.ml_engine.models import DeepFM
    from liteads.ml_engine.training import Trainer, TrainingConfig

    print("=" * 60)
    print("Criteo DeepFM Model Training")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Feature config: {args.config_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")

    # Load data
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    if not train_path.exists():
        print(f"\nError: Training data not found at {train_path}")
        print("Please run preprocess_criteo.py first")
        sys.exit(1)

    print(f"\nLoading data...")
    train_df = pd.read_parquet(train_path)
    print(f"  Training samples: {len(train_df):,}")

    val_df = None
    if val_path.exists():
        val_df = pd.read_parquet(val_path)
        print(f"  Validation samples: {len(val_df):,}")

    # Setup feature builder with Criteo config
    print(f"\nInitializing feature builder...")
    feature_builder = FeatureBuilder(
        config_path=args.config_path,
        device=args.device if args.device != "auto" else "cpu",
    )

    # Setup data module
    print(f"Setting up data module...")
    data_module = AdDataModule(
        feature_builder=feature_builder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data_module.setup_from_dataframe(
        train_df=train_df,
        val_df=val_df,
        label_cols=["click"],
        val_split=0.0 if val_df is not None else 0.1,
    )

    # Get model configuration
    model_config = data_module.get_model_config()

    print(f"\nModel configuration:")
    print(f"  Sparse features: {len(model_config['sparse_feature_dims'])}")
    print(f"  Dense features: {model_config['dense_feature_dim']}")
    print(f"  DNN hidden units: {args.dnn_hidden_units}")

    # Create model
    print(f"\nCreating DeepFM model...")
    model = DeepFM(
        sparse_feature_dims=model_config["sparse_feature_dims"],
        sparse_embedding_dims=model_config.get("sparse_embedding_dims", args.embedding_dim),
        dense_feature_dim=model_config["dense_feature_dim"],
        sequence_feature_dims=model_config.get("sequence_feature_dims", {}),
        sequence_embedding_dims=model_config.get("sequence_embedding_dims", {}),
        fm_k=model_config.get("fm_k", 8),
        dnn_hidden_units=args.dnn_hidden_units,
        dnn_dropout=args.dnn_dropout,
        l2_reg_embedding=args.weight_decay,
        l2_reg_dnn=args.weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Create trainer
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device,
        log_every_n_steps=100,
    )

    trainer = Trainer(model=model, config=training_config)

    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Train
    print(f"\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    start_time = time.time()
    metrics = trainer.fit(train_loader, val_loader)
    training_time = time.time() - start_time

    # Save model
    print(f"\n" + "=" * 60)
    print("Saving Model Files...")
    print("=" * 60)

    # 1. Save model checkpoint
    model_path = output_dir / f"{args.model_name}.pt"
    checkpoint = {
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "model_type": "deepfm",
        "model_config": model_config,
        "version": args.model_name,
    }
    torch.save(checkpoint, model_path)
    print(f"  Model checkpoint: {model_path}")

    # 2. Save feature builder
    feature_builder_path = output_dir / f"{args.model_name}_features.pkl"
    feature_builder.save(str(feature_builder_path))
    print(f"  Feature builder: {feature_builder_path}")

    # 3. Save model config as JSON
    config_path = output_dir / "model_config.json"
    config_to_save = {
        "model_type": "deepfm",
        "model_name": args.model_name,
        "sparse_feature_dims": model_config["sparse_feature_dims"],
        "dense_feature_dim": model_config["dense_feature_dim"],
        "sparse_feature_names": model_config.get("sparse_feature_names", []),
        "dense_feature_names": model_config.get("dense_feature_names", []),
        "embedding_dim": args.embedding_dim,
        "dnn_hidden_units": args.dnn_hidden_units,
        "fm_k": model_config.get("fm_k", 8),
        "num_parameters": num_params,
    }
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"  Model config: {config_path}")

    # 4. Save training metrics
    metrics_path = output_dir / "training_metrics.json"
    metrics_to_save = {
        "training_time_seconds": training_time,
        "num_epochs": trainer.current_epoch + 1,
        "best_epoch": metrics.best_epoch + 1,
        "best_val_loss": metrics.best_val_loss,
        "best_val_auc": metrics.best_val_auc,
        "train_loss_history": metrics.train_loss,
        "val_loss_history": metrics.val_loss,
        "val_auc_history": metrics.val_auc,
        "epoch_times": metrics.epoch_times,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"  Training metrics: {metrics_path}")

    # Print summary
    print(f"\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    print(f"\nResults:")
    print(f"  Training time: {training_time:.1f} seconds")
    print(f"  Best epoch: {metrics.best_epoch + 1}")
    print(f"  Best validation loss: {metrics.best_val_loss:.4f}")
    print(f"  Best validation AUC: {metrics.best_val_auc:.4f}")

    print(f"\nSaved files:")
    print(f"  {model_path}")
    print(f"  {feature_builder_path}")
    print(f"  {config_path}")
    print(f"  {metrics_path}")

    print(f"\n" + "=" * 60)
    print("Model Ready for Serving!")
    print("=" * 60)
    print(f"\nTo load the model:")
    print(f"  from liteads.ml_engine.serving.predictor import ModelPredictor")
    print(f"  predictor = ModelPredictor(")
    print(f"      model_path='{model_path}',")
    print(f"      feature_builder_path='{feature_builder_path}'")
    print(f"  )")
    print(f"  predictor.load()")

    print(f"\nNext step: Simulate ad requests")
    print(f"  python scripts/criteo/simulate_ad_request.py")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
