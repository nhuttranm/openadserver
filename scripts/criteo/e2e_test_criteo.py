#!/usr/bin/env python3
"""
End-to-end test for Criteo Click Logs integration.

Runs the complete pipeline:
1. Download data (or use existing)
2. Preprocess data
3. Train model
4. Run predictions
5. Generate report

Usage:
    python scripts/criteo/e2e_test_criteo.py [--skip-download] [--epochs 3]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end Criteo integration test"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download if files exist",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing if files exist",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training if model exists",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500000,
        help="Number of samples to download",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of test requests",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/criteo",
        help="Data directory",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/criteo",
        help="Model directory",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str], check: bool = True) -> tuple[bool, float, str]:
    """
    Run a pipeline step.

    Returns:
        Tuple of (success, duration_seconds, output)
    """
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print("-" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        duration = time.time() - start_time

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0 and check:
            print(f"\nStep FAILED with return code {result.returncode}")
            return False, duration, result.stderr

        print(f"\nStep completed in {duration:.1f} seconds")
        return True, duration, result.stdout

    except Exception as e:
        duration = time.time() - start_time
        print(f"\nStep FAILED with exception: {e}")
        return False, duration, str(e)


def check_files_exist(paths: list[Path]) -> bool:
    """Check if all files exist."""
    return all(p.exists() for p in paths)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Criteo End-to-End Integration Test")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Number of samples: {args.num_samples:,}")
    print(f"  Training epochs: {args.epochs}")

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    results = {
        "start_time": datetime.now().isoformat(),
        "config": vars(args),
        "steps": {},
    }

    # Step 1: Download data
    raw_files = [data_dir / "train_raw.parquet", data_dir / "test_raw.parquet"]
    if args.skip_download and check_files_exist(raw_files):
        print(f"\n[SKIP] Data download - files already exist")
        results["steps"]["download"] = {"skipped": True}
    else:
        success, duration, output = run_step(
            "Download Criteo Data",
            [
                sys.executable,
                "scripts/criteo/download_criteo.py",
                "--num-samples", str(args.num_samples),
                "--output-dir", args.data_dir,
            ],
        )
        results["steps"]["download"] = {
            "success": success,
            "duration_seconds": duration,
        }
        if not success:
            print("\n[FAILED] Data download failed. Aborting.")
            sys.exit(1)

    # Step 2: Preprocess data
    processed_files = [data_dir / "train.parquet", data_dir / "val.parquet"]
    if args.skip_preprocess and check_files_exist(processed_files):
        print(f"\n[SKIP] Preprocessing - files already exist")
        results["steps"]["preprocess"] = {"skipped": True}
    else:
        success, duration, output = run_step(
            "Preprocess Data",
            [
                sys.executable,
                "scripts/criteo/preprocess_criteo.py",
                "--input-dir", args.data_dir,
                "--output-dir", args.data_dir,
            ],
        )
        results["steps"]["preprocess"] = {
            "success": success,
            "duration_seconds": duration,
        }
        if not success:
            print("\n[FAILED] Preprocessing failed. Aborting.")
            sys.exit(1)

    # Step 3: Train model
    model_files = [
        model_dir / "deepfm_criteo.pt",
        model_dir / "deepfm_criteo_features.pkl",
    ]
    if args.skip_training and check_files_exist(model_files):
        print(f"\n[SKIP] Training - model already exists")
        results["steps"]["training"] = {"skipped": True}
    else:
        success, duration, output = run_step(
            "Train DeepFM Model",
            [
                sys.executable,
                "scripts/criteo/train_criteo_model.py",
                "--data-dir", args.data_dir,
                "--output-dir", args.model_dir,
                "--epochs", str(args.epochs),
                "--batch-size", "512",
            ],
        )
        results["steps"]["training"] = {
            "success": success,
            "duration_seconds": duration,
        }
        if not success:
            print("\n[FAILED] Training failed. Aborting.")
            sys.exit(1)

    # Step 4: Run simulation
    success, duration, output = run_step(
        "Simulate Ad Requests",
        [
            sys.executable,
            "scripts/criteo/simulate_ad_request.py",
            "--data-dir", args.data_dir,
            "--model-dir", args.model_dir,
            "--num-requests", str(args.num_requests),
        ],
    )
    results["steps"]["simulation"] = {
        "success": success,
        "duration_seconds": duration,
    }

    # Load simulation results if available
    simulation_results_path = model_dir / "simulation_results.json"
    if simulation_results_path.exists():
        with open(simulation_results_path) as f:
            simulation_data = json.load(f)
            results["steps"]["simulation"]["metrics"] = simulation_data

    # Load training metrics if available
    training_metrics_path = model_dir / "training_metrics.json"
    if training_metrics_path.exists():
        with open(training_metrics_path) as f:
            training_data = json.load(f)
            results["training_metrics"] = training_data

    # Calculate totals
    total_duration = sum(
        step.get("duration_seconds", 0)
        for step in results["steps"].values()
        if not step.get("skipped", False)
    )

    results["end_time"] = datetime.now().isoformat()
    results["total_duration_seconds"] = total_duration

    # Save results
    report_path = model_dir / "e2e_test_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("End-to-End Test Summary")
    print("=" * 60)

    print("\nStep Results:")
    for step_name, step_result in results["steps"].items():
        if step_result.get("skipped"):
            status = "[SKIPPED]"
            duration = "-"
        elif step_result.get("success"):
            status = "[SUCCESS]"
            duration = f"{step_result['duration_seconds']:.1f}s"
        else:
            status = "[FAILED]"
            duration = f"{step_result.get('duration_seconds', 0):.1f}s"
        print(f"  {step_name:15} {status:10} {duration}")

    print(f"\nTotal Duration: {total_duration:.1f} seconds")

    if "training_metrics" in results:
        tm = results["training_metrics"]
        print(f"\nModel Performance:")
        print(f"  Best AUC: {tm.get('best_val_auc', 'N/A'):.4f}")
        print(f"  Best Loss: {tm.get('best_val_loss', 'N/A'):.4f}")

    if "simulation" in results["steps"]:
        sim = results["steps"]["simulation"]
        if "metrics" in sim and "offline" in sim["metrics"]:
            offline = sim["metrics"]["offline"]
            print(f"\nPrediction Performance:")
            print(f"  Test AUC: {offline.get('auc', 'N/A'):.4f}")
            print(f"  Throughput: {offline.get('throughput', 'N/A'):.0f} samples/sec")
            print(f"  Avg Latency: {offline.get('avg_latency_ms', 'N/A'):.2f}ms")

    print(f"\nReport saved to: {report_path}")

    # Print model files
    print(f"\nModel Files:")
    for f in model_dir.glob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("End-to-End Test Complete!")
    print("=" * 60)

    # Usage instructions
    print("\n" + "-" * 60)
    print("How to use the trained model:")
    print("-" * 60)
    print("""
from liteads.ml_engine.serving.predictor import ModelPredictor

# Load model
predictor = ModelPredictor(
    model_path="models/criteo/deepfm_criteo.pt",
    feature_builder_path="models/criteo/deepfm_criteo_features.pkl"
)
predictor.load()

# Make prediction
features = {
    "I1": 5, "I2": 10, ...,  # Integer features
    "C1": 12345, "C2": 67890, ...,  # Categorical features (hashed)
}
result = predictor.predict(features)
print(f"pCTR: {result.pctr:.4f}")
""")

    # Check if all steps succeeded
    all_success = all(
        step.get("success", True) or step.get("skipped", False)
        for step in results["steps"].values()
    )

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
