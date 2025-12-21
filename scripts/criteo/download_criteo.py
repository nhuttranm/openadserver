#!/usr/bin/env python3
"""
Download Criteo Click Logs dataset from Hugging Face.

Downloads a subset of the data (~200MB, ~1-2 million samples) for testing.

Usage:
    python scripts/criteo/download_criteo.py [--num-samples 1000000]
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
        description="Download Criteo Click Logs dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000000,
        help="Number of samples to download (default: 1M for ~200MB)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/criteo",
        help="Output directory for data files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for testing",
    )
    return parser.parse_args()


def download_and_save(args: argparse.Namespace) -> None:
    """Download Criteo dataset and save to files."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Please install it with: pip install datasets")
        sys.exit(1)

    import numpy as np
    import pandas as pd

    print("=" * 60)
    print("Criteo Click Logs Dataset Downloader")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Number of samples: {args.num_samples:,}")
    print(f"  Output directory: {output_dir}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Random seed: {args.seed}")

    # Load dataset from Hugging Face
    print("\nDownloading Criteo dataset from Hugging Face...")
    print("(This may take a few minutes depending on your connection)")

    # Try different dataset sources
    dataset = None
    dataset_sources = [
        ("criteo/CriteoClickLogs", None),  # Official Criteo dataset
        ("reczoo/Criteo_x1", "train"),     # Preprocessed version
    ]

    for dataset_name, split in dataset_sources:
        try:
            print(f"\nTrying to load: {dataset_name}")
            if split:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                )
            else:
                # For datasets without explicit splits, try streaming
                dataset = load_dataset(
                    dataset_name,
                    streaming=True,
                )
                # If it's an IterableDatasetDict, get the first available split
                if hasattr(dataset, 'keys'):
                    available_splits = list(dataset.keys())
                    if available_splits:
                        print(f"  Available splits: {available_splits}")
                        dataset = dataset[available_splits[0]]
            print(f"  Successfully connected to {dataset_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            dataset = None
            continue

    if dataset is None:
        print("\nError: Could not load Criteo dataset from any source.")
        print("Please check your internet connection and Hugging Face access.")
        print("\nAlternative: Download Criteo data manually from:")
        print("  https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/")
        sys.exit(1)

    # Take first N samples
    print(f"\nSampling {args.num_samples:,} records...")
    samples = []
    for i, sample in enumerate(dataset):
        if i >= args.num_samples:
            break
        samples.append(sample)
        if (i + 1) % 100000 == 0:
            print(f"  Downloaded {i + 1:,} samples...")

    print(f"  Total samples: {len(samples):,}")

    if len(samples) == 0:
        print("\nError: No samples downloaded. Dataset may be empty or inaccessible.")
        sys.exit(1)

    # Convert to DataFrame
    print("\nConverting to DataFrame...")
    df = pd.DataFrame(samples)

    # Rename columns to match Criteo format
    # Criteo format: label, I1-I13 (integers), C1-C26 (categories)
    column_mapping = {}

    # Check if columns need renaming
    if "label" in df.columns:
        pass  # Already correct
    elif "click" in df.columns:
        column_mapping["click"] = "label"

    # Map integer features
    for i in range(1, 14):
        old_name = f"int_{i}" if f"int_{i}" in df.columns else f"I{i}"
        if old_name in df.columns and old_name != f"I{i}":
            column_mapping[old_name] = f"I{i}"

    # Map categorical features
    for i in range(1, 27):
        old_name = f"cat_{i}" if f"cat_{i}" in df.columns else f"C{i}"
        if old_name in df.columns and old_name != f"C{i}":
            column_mapping[old_name] = f"C{i}"

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Ensure all expected columns exist
    expected_cols = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    for col in expected_cols:
        if col not in df.columns:
            if col == "label":
                df[col] = 0
            elif col.startswith("I"):
                df[col] = 0
            else:
                df[col] = ""

    # Reorder columns
    df = df[expected_cols]

    # Split into train and test
    np.random.seed(args.seed)
    n_test = int(len(df) * args.test_ratio)
    indices = np.random.permutation(len(df))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    # Save to files
    train_path = output_dir / "train_raw.parquet"
    test_path = output_dir / "test_raw.parquet"

    print(f"\nSaving data:")
    print(f"  Training set: {len(train_df):,} samples -> {train_path}")
    train_df.to_parquet(train_path, index=False)

    print(f"  Test set: {len(test_df):,} samples -> {test_path}")
    test_df.to_parquet(test_path, index=False)

    # Print data statistics
    print("\n" + "=" * 60)
    print("Data Statistics:")
    print("=" * 60)
    print(f"  Total samples: {len(df):,}")
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Positive rate (clicks): {df['label'].mean():.4f}")
    print(f"  Number of features: {len(df.columns) - 1}")
    print(f"    - Integer features (I1-I13): 13")
    print(f"    - Categorical features (C1-C26): 26")

    # File sizes
    train_size = train_path.stat().st_size / (1024 * 1024)
    test_size = test_path.stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  Training data: {train_size:.1f} MB")
    print(f"  Test data: {test_size:.1f} MB")
    print(f"  Total: {train_size + test_size:.1f} MB")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nNext step: Run preprocessing")
    print(f"  python scripts/criteo/preprocess_criteo.py")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    download_and_save(args)


if __name__ == "__main__":
    main()
