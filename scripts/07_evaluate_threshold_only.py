"""Entry point: stage 7 threshold-only baseline evaluation.

Evaluates the stage-6 threshold sweep frame-by-frame against stage-5 truth:
overall and per-day operating curves, range-bin detection probability,
clutter-by-range, SNR distributions of written detections, and measurement-
error sanity checks -- as CSV tables, matplotlib plots, and a Markdown
report. NO tracking, data association, or ML happens here (that's stage 8).

Usage:
    python scripts/07_evaluate_threshold_only.py --overwrite
    python scripts/07_evaluate_threshold_only.py --self-test
"""

import argparse
import os
import sys

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.threshold_eval import (
    ThresholdEvalConfig,
    parse_range_bins,
    run_evaluation,
    self_test,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Threshold-only baseline evaluation of stage-6 detections (no tracking).")
    parser.add_argument("--truth-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    parser.add_argument("--detections-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage07_threshold_only"))
    parser.add_argument("--coverage-range-m", type=float, default=100_000.0,
                        help="Stage-6 clutter-support radius, referenced in the report (default: 100000).")
    parser.add_argument("--range-bins-m", type=str, default="0,50000,100000,200000,inf",
                        help="Comma-separated bin edges in metres; 'inf' allowed (default: "
                             "0,50000,100000,200000,inf).")
    parser.add_argument("--chunksize", type=int, default=1_000_000,
                        help="Rows per chunk for the large CSV scans (default: 1000000).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate outputs that already exist (default: refuse).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG generation (tables and report only).")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = ThresholdEvalConfig(
        truth_dir=args.truth_dir,
        detections_dir=args.detections_dir,
        output_dir=args.output_dir,
        coverage_range_m=args.coverage_range_m,
        range_bins_m=parse_range_bins(args.range_bins_m),
        chunksize=args.chunksize,
        overwrite=args.overwrite,
        no_plots=args.no_plots,
    )
    run_evaluation(cfg)
    print("\n07_evaluate_threshold_only completed successfully.")


if __name__ == "__main__":
    main()
