"""Entry point: convert stage 4's uniformly-sampled trajectories into
noiseless radar-coordinate ground truth relative to a fixed radar site.

No radar noise, clutter, detection thresholds, range-Doppler imaging, ML
dataset construction, or train/test splitting happens here -- see
utils/radar_truth.py for exactly what this stage does and does not do.
Stage 6 will simulate radar detections/noise/clutter on top of this truth.

Usage:
    python scripts/05_make_radar_truth.py --radar-lat 45.05 --radar-lon 7.35
    python scripts/05_make_radar_truth.py --self-test
"""

import argparse
import os
import sys

import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_radar_truth_dir, get_radar_truth_summary_path, get_trajectories_dir
from utils.radar_truth import (
    RadarTruthConfig,
    discover_input_files,
    make_radar_truth_for_day,
    run_validation_gate,
    self_test,
    summarize_day,
)

SUMMARY_COLUMNS = [
    "date", "status", "input_rows", "output_rows",
    "unique_trajectories_in", "unique_trajectories_out",
    "rows_dropped_min_range", "rows_dropped_max_range", "rows_dropped_below_horizon",
    "output_file",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Express stage-4 trajectories in radar coordinates (noiseless ground truth).")
    parser.add_argument("--radar-lat", type=float, default=None,
                         help="Radar site latitude in degrees (REQUIRED -- no default is invented).")
    parser.add_argument("--radar-lon", type=float, default=None,
                         help="Radar site longitude in degrees (REQUIRED -- no default is invented).")
    parser.add_argument("--radar-alt-m", type=float, default=0.0,
                         help="Radar site altitude in metres (default: 0).")
    parser.add_argument("--radar-name", type=str, default="radar",
                         help="Radar site name recorded in the output (default: 'radar').")
    parser.add_argument("--input-dir", type=str, default=None,
                         help="Directory containing stage-4 trajectory CSVs "
                              "(default: this project's data/active/trajectories_10s).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Directory for the radar-truth CSVs and summary "
                              "(default: this project's data/active/radar_truth).")
    parser.add_argument("--position-source", choices=["smooth", "interp"], default="smooth",
                         help="Which stage-4 position triple to use (default: smooth).")
    parser.add_argument("--min-range-m", type=float, default=0.0,
                         help="Drop rows with range below this (default: 0, i.e. keep all).")
    parser.add_argument("--max-range-m", type=float, default=None,
                         help="If set, drop rows with range above this (default: keep all).")
    parser.add_argument("--drop-below-horizon", action="store_true",
                         help="Drop rows with elevation below 0 rad (default: keep them).")
    parser.add_argument("--overwrite", action="store_true",
                         help="Regenerate outputs that already exist (default: skip them).")
    parser.add_argument("--self-test", action="store_true",
                         help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    args = parser.parse_args()

    if not args.self_test and (args.radar_lat is None or args.radar_lon is None):
        parser.error("--radar-lat and --radar-lon are required (the radar location is never invented)")
    return args


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = RadarTruthConfig(
        radar_lat=args.radar_lat,
        radar_lon=args.radar_lon,
        radar_alt_m=args.radar_alt_m,
        radar_name=args.radar_name,
        position_source=args.position_source,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        drop_below_horizon=args.drop_below_horizon,
        overwrite=args.overwrite,
    )

    input_dir = args.input_dir or get_trajectories_dir()
    output_dir = args.output_dir or get_radar_truth_dir()
    os.makedirs(output_dir, exist_ok=True)

    day_files = discover_input_files(input_dir)
    if not day_files:
        print(f"No {'states_*' + '_conventionalGA_trajectories_10s.csv'} files found in {input_dir}")
        return

    day_results = []
    for date, input_path in day_files:
        result = make_radar_truth_for_day(date, input_path, output_dir, cfg)
        day_results.append(result)

        if result["status"] == "skipped":
            continue
        dropped = (result["rows_dropped_min_range"] + result["rows_dropped_max_range"]
                   + result["rows_dropped_below_horizon"])
        print(f"\n--- {result['date']} ---")
        print(f"input rows:                  {result['input_rows']}")
        print(f"output rows:                 {result['output_rows']}")
        print(f"unique trajectories in:      {result['unique_trajectories_in']}")
        print(f"unique trajectories out:     {result['unique_trajectories_out']}")
        print(f"rows dropped (range/horizon):{dropped:>9} "
              f"(min-range {result['rows_dropped_min_range']}, "
              f"max-range {result['rows_dropped_max_range']}, "
              f"horizon {result['rows_dropped_below_horizon']})")
        print(f"output path:                 {result['output_file']}")

    summary_df = pd.DataFrame([summarize_day(r) for r in day_results], columns=SUMMARY_COLUMNS)
    summary_path = get_radar_truth_summary_path(output_dir)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")

    run_validation_gate(day_results)

    print("\n05_make_radar_truth completed successfully.")


if __name__ == "__main__":
    main()
