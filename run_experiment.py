"""Command-line entry point for the complete experiment."""

from __future__ import annotations

import argparse
import json

from hvac_anomaly.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to RTU_CLEAN_faultandnofault.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold-quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_quantile=args.threshold_quantile,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
