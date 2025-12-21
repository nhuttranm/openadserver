#!/usr/bin/env python3
"""
Preprocess Criteo Click Logs data for model training.

Handles:
- Missing value imputation
- Feature hashing for categorical features
- Train/validation split
- Output to processed format

Usage:
    python scripts/criteo/preprocess_criteo.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess Criteo Click Logs data"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/criteo",
        help="Input directory with raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/criteo",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--hash-buckets",
        type=int,
        default=100000,
        help="Number of hash buckets for categorical features",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def preprocess_data(args: argparse.Namespace) -> None:
    """Preprocess Criteo data."""
    import numpy as np
    import pandas as pd

    print("=" * 60)
    print("Criteo Data Preprocessing")
    print("=" * 60)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    train_raw_path = input_dir / "train_raw.parquet"
    test_raw_path = input_dir / "test_raw.parquet"

    if not train_raw_path.exists():
        print(f"Error: Training data not found at {train_raw_path}")
        print("Please run download_criteo.py first")
        sys.exit(1)

    print(f"\nLoading data from {input_dir}...")
    train_df = pd.read_parquet(train_raw_path)
    print(f"  Training data: {len(train_df):,} samples")

    if test_raw_path.exists():
        test_df = pd.read_parquet(test_raw_path)
        print(f"  Test data: {len(test_df):,} samples")
    else:
        test_df = None
        print("  No test data found")

    # Define feature columns
    int_cols = [f"I{i}" for i in range(1, 14)]
    cat_cols = [f"C{i}" for i in range(1, 27)]

    print("\nProcessing features...")

    # Process integer features
    print("  Processing integer features (I1-I13)...")
    for col in int_cols:
        # Fill missing values with 0
        train_df[col] = train_df[col].fillna(0).astype(np.float32)
        if test_df is not None:
            test_df[col] = test_df[col].fillna(0).astype(np.float32)

    # Process categorical features
    print("  Processing categorical features (C1-C26)...")
    for col in cat_cols:
        # Fill missing values with special token
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        if test_df is not None:
            test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

        # Hash to fixed bucket size
        train_df[col] = train_df[col].apply(
            lambda x: hash(x) % args.hash_buckets
        )
        if test_df is not None:
            test_df[col] = test_df[col].apply(
                lambda x: hash(x) % args.hash_buckets
            )

    # Rename label column to 'click' for compatibility
    train_df = train_df.rename(columns={"label": "click"})
    if test_df is not None:
        test_df = test_df.rename(columns={"label": "click"})

    # Split training into train/validation
    print(f"\nSplitting training data (val_ratio={args.val_ratio})...")
    np.random.seed(args.seed)
    n_val = int(len(train_df) * args.val_ratio)
    indices = np.random.permutation(len(train_df))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    val_df = train_df.iloc[val_indices].reset_index(drop=True)
    train_df = train_df.iloc[train_indices].reset_index(drop=True)

    print(f"  Training set: {len(train_df):,} samples")
    print(f"  Validation set: {len(val_df):,} samples")
    if test_df is not None:
        print(f"  Test set: {len(test_df):,} samples")

    # Save processed data
    print("\nSaving processed data...")

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    print(f"  Training data: {train_path}")

    val_df.to_parquet(val_path, index=False)
    print(f"  Validation data: {val_path}")

    if test_df is not None:
        test_df.to_parquet(test_path, index=False)
        print(f"  Test data: {test_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Data Statistics:")
    print("=" * 60)

    print("\nLabel distribution:")
    print(f"  Training - Click rate: {train_df['click'].mean():.4f}")
    print(f"  Validation - Click rate: {val_df['click'].mean():.4f}")
    if test_df is not None:
        print(f"  Test - Click rate: {test_df['click'].mean():.4f}")

    print("\nInteger features (I1-I13) stats:")
    for col in int_cols[:5]:  # Show first 5
        print(f"  {col}: mean={train_df[col].mean():.2f}, "
              f"std={train_df[col].std():.2f}, "
              f"max={train_df[col].max():.0f}")
    print("  ...")

    print("\nCategorical features (C1-C26) unique values (after hashing):")
    for col in cat_cols[:5]:  # Show first 5
        n_unique = train_df[col].nunique()
        print(f"  {col}: {n_unique:,} unique values")
    print("  ...")

    # File sizes
    train_size = train_path.stat().st_size / (1024 * 1024)
    val_size = val_path.stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  Training data: {train_size:.1f} MB")
    print(f"  Validation data: {val_size:.1f} MB")
    if test_df is not None:
        test_size = test_path.stat().st_size / (1024 * 1024)
        print(f"  Test data: {test_size:.1f} MB")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nNext step: Train the model")
    print(f"  python scripts/criteo/train_criteo_model.py")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    preprocess_data(args)


if __name__ == "__main__":
    main()
