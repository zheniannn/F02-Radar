"""Entry point: simulate thresholded radar point detections from stage 5's
noiseless radar-coordinate truth.

Point-detection level only: SNR draw -> logistic Pd vs threshold ->
Bernoulli detection with Gaussian measurement noise, plus Poisson clutter
false alarms per frame. No raw RF/IQ, pulse compression, range-Doppler
imaging, tracking, ML training, or train/test splitting happens here -- see
utils/radar_sim.py for exactly what this stage does and does not do.

Usage:
    python scripts/06_simulate_radar_detections.py
    python scripts/06_simulate_radar_detections.py --threshold-db -5 0 3 6 9 12 --overwrite
    python scripts/06_simulate_radar_detections.py --self-test
"""

import argparse
import os
import sys

import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_radar_truth_dir, get_sim_detections_dir, get_sim_detection_summary_path
from utils.radar_sim import (
    SUMMARY_COLUMNS,
    RadarSimConfig,
    discover_radar_truth_files,
    run_validation_gate,
    self_test,
    simulate_day,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate thresholded radar point detections from radar-coordinate truth.")

    parser.add_argument("--input-dir", type=str, default=None,
                         help="Directory containing radar_truth_YYYY-MM-DD.csv files "
                              "(default: this repo's data/active/radar_truth).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Directory for detection CSVs and summary "
                              "(default: this repo's data/active/sim_detections).")
    parser.add_argument("--threshold-db", nargs="+", type=float, default=[-5.0, 0.0, 3.0, 6.0, 9.0, 12.0],
                         help="Detection threshold(s) in dB; one output file per day per threshold.")
    parser.add_argument("--scenario-id", type=str, default="default",
                         help="Scenario label stamped on every row and used in seeding (default: 'default').")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base RNG seed; per-day/per-threshold children are derived via sha256 (default: 42).")
    parser.add_argument("--overwrite", action="store_true",
                         help="Regenerate outputs that already exist (default: skip them).")
    parser.add_argument("--frame-period-s", type=float, default=10.0,
                         help="Radar scan period: a frame is floor(timestamp / this) (default: 10, "
                              "matching the stage-4 grid). Stage-4 timestamps are per-trajectory "
                              "anchored, so frames must be binned, not grouped by raw timestamp.")

    snr = parser.add_argument_group("target SNR model")
    snr.add_argument("--snr-model", choices=["constant", "range_decay"], default="range_decay")
    snr.add_argument("--target-snr-ref-db", type=float, default=10.0,
                      help="Mean SNR at the reference range (or everywhere, for 'constant').")
    snr.add_argument("--snr-ref-range-m", type=float, default=50_000.0)
    snr.add_argument("--snr-range-power", type=float, default=4.0,
                      help="Range-decay exponent: SNR falls 10*power*log10(R/Rref) dB.")
    snr.add_argument("--target-snr-std-db", type=float, default=3.0)
    snr.add_argument("--target-snr-min-db", type=float, default=-20.0)
    snr.add_argument("--target-snr-max-db", type=float, default=30.0)

    pdg = parser.add_argument_group("detection probability")
    pdg.add_argument("--pd-transition-width-db", type=float, default=2.0)
    pdg.add_argument("--pd-max", type=float, default=0.98)
    pdg.add_argument("--pd-min", type=float, default=0.01)

    noise = parser.add_argument_group("measurement noise (1-sigma)")
    noise.add_argument("--sigma-range-m", type=float, default=75.0)
    noise.add_argument("--sigma-azimuth-deg", type=float, default=0.15)
    noise.add_argument("--sigma-elevation-deg", type=float, default=0.15)
    noise.add_argument("--sigma-radial-velocity-mps", type=float, default=2.0)

    clut = parser.add_argument_group("clutter model")
    clut.add_argument("--clutter-model", choices=["poisson"], default="poisson")
    clut.add_argument("--clutter-rate-ref", type=float, default=20.0,
                       help="Expected false alarms per frame at the reference threshold.")
    clut.add_argument("--clutter-ref-threshold-db", type=float, default=0.0)
    clut.add_argument("--clutter-threshold-scale-db", type=float, default=6.0)
    clut.add_argument("--max-range-m", type=float, default=None,
                       help="Clutter range upper bound; if omitted, inferred per day from the "
                            "truth's max range rounded up to the nearest 10 km (printed clearly).")
    clut.add_argument("--min-range-m", type=float, default=0.0)
    clut.add_argument("--min-elevation-deg", type=float, default=0.0)
    clut.add_argument("--max-elevation-deg", type=float, default=20.0)
    clut.add_argument("--min-radial-velocity-mps", type=float, default=-120.0)
    clut.add_argument("--max-radial-velocity-mps", type=float, default=120.0)
    clut.add_argument("--clutter-snr-scale-db", type=float, default=6.0,
                       help="Exponential scale of above-threshold clutter SNR.")

    parser.add_argument("--self-test", action="store_true",
                         help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = RadarSimConfig(
        thresholds_db=args.threshold_db,
        scenario_id=args.scenario_id,
        seed=args.seed,
        overwrite=args.overwrite,
        frame_period_s=args.frame_period_s,
        snr_model=args.snr_model,
        target_snr_ref_db=args.target_snr_ref_db,
        snr_ref_range_m=args.snr_ref_range_m,
        snr_range_power=args.snr_range_power,
        target_snr_std_db=args.target_snr_std_db,
        target_snr_min_db=args.target_snr_min_db,
        target_snr_max_db=args.target_snr_max_db,
        pd_transition_width_db=args.pd_transition_width_db,
        pd_max=args.pd_max,
        pd_min=args.pd_min,
        sigma_range_m=args.sigma_range_m,
        sigma_azimuth_deg=args.sigma_azimuth_deg,
        sigma_elevation_deg=args.sigma_elevation_deg,
        sigma_radial_velocity_mps=args.sigma_radial_velocity_mps,
        clutter_model=args.clutter_model,
        clutter_rate_ref=args.clutter_rate_ref,
        clutter_ref_threshold_db=args.clutter_ref_threshold_db,
        clutter_threshold_scale_db=args.clutter_threshold_scale_db,
        max_range_m=args.max_range_m,
        min_range_m=args.min_range_m,
        min_elevation_deg=args.min_elevation_deg,
        max_elevation_deg=args.max_elevation_deg,
        min_radial_velocity_mps=args.min_radial_velocity_mps,
        max_radial_velocity_mps=args.max_radial_velocity_mps,
        clutter_snr_scale_db=args.clutter_snr_scale_db,
    )

    input_dir = args.input_dir or get_radar_truth_dir()
    output_dir = args.output_dir or get_sim_detections_dir()
    os.makedirs(output_dir, exist_ok=True)

    day_files = discover_radar_truth_files(input_dir)
    if not day_files:
        print(f"No radar_truth_YYYY-MM-DD.csv files found in {input_dir}")
        return

    all_results = []
    for date, input_path in day_files:
        results = simulate_day(date, input_path, output_dir, cfg)
        all_results.extend(results)

        for r in results:
            if r["status"] == "skipped":
                continue
            print(f"\n--- {r['date']}  threshold {r['threshold_db']:g} dB ---")
            print(f"truth rows:            {r['truth_rows']}")
            print(f"frames:                {r['frames']}")
            print(f"target detections:     {r['target_detections']}")
            print(f"missed targets:        {r['missed_targets']}")
            print(f"empirical Pd:          {r['empirical_pd']:.4f}")
            print(f"clutter detections:    {r['clutter_detections']}")
            print(f"total detections:      {r['total_detections']}")
            print(f"false alarms / frame:  {r['false_alarm_per_frame']:.3f}")
            print(f"output path:           {r['output_file']}")

    summary_df = pd.DataFrame(all_results, columns=SUMMARY_COLUMNS)
    summary_path = get_sim_detection_summary_path(output_dir)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")

    run_validation_gate([r["output_file"] for r in all_results], summary_df)

    print("\n06_simulate_radar_detections completed successfully.")


if __name__ == "__main__":
    main()
