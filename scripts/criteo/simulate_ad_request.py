#!/usr/bin/env python3
"""
Simulate ad requests using Criteo test data.

Tests:
1. Direct model prediction (offline)
2. HTTP API request (online, if server is running)

Usage:
    python scripts/criteo/simulate_ad_request.py [--num-requests 100]
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
        description="Simulate ad requests using Criteo data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/criteo",
        help="Directory with test data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/criteo",
        help="Directory with trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepfm_criteo",
        help="Model name",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to simulate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL for online testing",
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Also test HTTP API (requires running server)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    return parser.parse_args()


def test_offline_prediction(args: argparse.Namespace) -> dict:
    """Test model prediction directly (offline)."""
    import numpy as np
    import pandas as pd

    from liteads.ml_engine.serving.predictor import ModelPredictor

    print("\n" + "=" * 60)
    print("Offline Model Prediction Test")
    print("=" * 60)

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # Load model
    model_path = model_dir / f"{args.model_name}.pt"
    feature_builder_path = model_dir / f"{args.model_name}_features.pkl"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run train_criteo_model.py first")
        return {}

    print(f"\nLoading model from {model_path}...")
    predictor = ModelPredictor(
        model_path=model_path,
        feature_builder_path=feature_builder_path,
        device="cpu",
    )
    predictor.load()
    print("  Model loaded successfully!")

    # Load test data
    test_path = data_dir / "test.parquet"
    if not test_path.exists():
        test_path = data_dir / "val.parquet"

    if not test_path.exists():
        print(f"Error: No test data found in {data_dir}")
        return {}

    print(f"\nLoading test data from {test_path}...")
    test_df = pd.read_parquet(test_path)
    print(f"  Test samples available: {len(test_df):,}")

    # Sample test data
    np.random.seed(args.seed)
    n_samples = min(args.num_requests, len(test_df))
    sample_indices = np.random.choice(len(test_df), n_samples, replace=False)
    samples_df = test_df.iloc[sample_indices].reset_index(drop=True)

    print(f"  Samples selected: {n_samples}")

    # Convert to feature dictionaries
    print(f"\nPreparing features...")
    feature_records = samples_df.drop(columns=["click"]).to_dict("records")
    labels = samples_df["click"].values

    # Run predictions
    print(f"\nRunning predictions...")
    start_time = time.time()

    predictions = []
    latencies = []

    for i in range(0, len(feature_records), args.batch_size):
        batch = feature_records[i:i + args.batch_size]
        batch_start = time.time()
        results = predictor.predict_batch(batch)
        batch_latency = (time.time() - batch_start) * 1000

        predictions.extend([r.pctr for r in results])
        latencies.append(batch_latency)

        if (i + args.batch_size) % 100 == 0 or i + args.batch_size >= len(feature_records):
            print(f"  Processed {min(i + args.batch_size, len(feature_records)):,} samples...")

    total_time = time.time() - start_time
    predictions = np.array(predictions)

    # Calculate metrics
    from sklearn.metrics import roc_auc_score, log_loss

    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.5

    try:
        logloss = log_loss(labels, predictions)
    except ValueError:
        logloss = float("inf")

    # Statistics
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = len(predictions) / total_time

    print(f"\n" + "-" * 40)
    print("Prediction Results:")
    print("-" * 40)
    print(f"  Total samples: {len(predictions):,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print(f"\nLatency (per batch of {args.batch_size}):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")
    print(f"\nModel Quality:")
    print(f"  AUC: {auc:.4f}")
    print(f"  LogLoss: {logloss:.4f}")
    print(f"\nPrediction Distribution:")
    print(f"  Min pCTR: {predictions.min():.4f}")
    print(f"  Max pCTR: {predictions.max():.4f}")
    print(f"  Mean pCTR: {predictions.mean():.4f}")
    print(f"  Actual CTR: {labels.mean():.4f}")

    # Show sample predictions
    print(f"\nSample Predictions (first 10):")
    print(f"  {'pCTR':>8}  {'Label':>6}")
    print(f"  {'-'*8}  {'-'*6}")
    for i in range(min(10, len(predictions))):
        print(f"  {predictions[i]:>8.4f}  {labels[i]:>6}")

    return {
        "num_samples": len(predictions),
        "total_time_seconds": total_time,
        "throughput": throughput,
        "avg_latency_ms": avg_latency,
        "p99_latency_ms": p99_latency,
        "auc": auc,
        "logloss": logloss,
        "mean_pctr": float(predictions.mean()),
        "actual_ctr": float(labels.mean()),
    }


def test_api_request(args: argparse.Namespace) -> dict:
    """Test HTTP API requests (online)."""
    import numpy as np
    import pandas as pd

    try:
        import httpx
    except ImportError:
        print("\nNote: httpx not installed, skipping API test")
        print("Install with: pip install httpx")
        return {}

    print("\n" + "=" * 60)
    print("Online API Request Test")
    print("=" * 60)

    # Check if server is running
    try:
        response = httpx.get(f"{args.api_url}/health", timeout=5.0)
        if response.status_code != 200:
            print(f"\nServer not healthy at {args.api_url}")
            return {}
    except Exception as e:
        print(f"\nServer not reachable at {args.api_url}: {e}")
        print("Please start the server first:")
        print("  python -m liteads.ad_server.main")
        return {}

    print(f"\nServer is running at {args.api_url}")

    # Load test data
    data_dir = Path(args.data_dir)
    test_path = data_dir / "test.parquet"
    if not test_path.exists():
        test_path = data_dir / "val.parquet"

    if not test_path.exists():
        print(f"Error: No test data found")
        return {}

    test_df = pd.read_parquet(test_path)

    # Sample and convert to API request format
    np.random.seed(args.seed + 1)  # Different seed for different samples
    n_samples = min(args.num_requests, len(test_df))
    sample_indices = np.random.choice(len(test_df), n_samples, replace=False)
    samples_df = test_df.iloc[sample_indices]

    print(f"\nSending {n_samples} requests to API...")

    latencies = []
    success_count = 0
    error_count = 0

    for idx, (_, row) in enumerate(samples_df.iterrows()):
        # Convert Criteo features to AdRequest format
        # Note: This is a simplified mapping for demonstration
        request_data = {
            "slot_id": "criteo_test_slot",
            "user_id": f"criteo_user_{idx}",
            "device": {
                "os": "android",
                "os_version": "13.0",
                "model": "Test Device",
            },
            "geo": {
                "country": "US",
            },
            "user_features": {
                "custom": {
                    # Map Criteo features to custom features
                    **{f"I{i}": float(row.get(f"I{i}", 0)) for i in range(1, 14)},
                    **{f"C{i}": int(row.get(f"C{i}", 0)) for i in range(1, 27)},
                }
            },
            "num_ads": 1,
        }

        try:
            start_time = time.time()
            response = httpx.post(
                f"{args.api_url}/api/v1/ad",
                json=request_data,
                timeout=10.0,
            )
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            print(f"  Request {idx} failed: {e}")

        if (idx + 1) % 20 == 0:
            print(f"  Sent {idx + 1} requests...")

    # Statistics
    if latencies:
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
    else:
        avg_latency = p50_latency = p99_latency = max_latency = 0

    print(f"\n" + "-" * 40)
    print("API Test Results:")
    print("-" * 40)
    print(f"  Total requests: {n_samples}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"\nLatency:")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  P50: {p50_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")

    return {
        "total_requests": n_samples,
        "success_count": success_count,
        "error_count": error_count,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p99_latency_ms": p99_latency,
        "max_latency_ms": max_latency,
    }


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Criteo Ad Request Simulation")
    print("=" * 60)

    results = {}

    # Test offline prediction
    offline_results = test_offline_prediction(args)
    if offline_results:
        results["offline"] = offline_results

    # Test API if requested
    if args.test_api:
        api_results = test_api_request(args)
        if api_results:
            results["api"] = api_results

    # Save results
    if results:
        output_dir = Path(args.model_dir)
        results_path = output_dir / "simulation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
