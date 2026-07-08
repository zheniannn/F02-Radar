"""Read-only audit of the relocated wide-area weak-target experiment.

Summarizes existing stage-5 relocated truth and stage-6 detection outputs
into a compact Markdown report: truth-level range/elevation/anchor
statistics, detection-level counts, the threshold sweep, and a
coverage-range audit that quantifies how much of the experiment lives
beyond the clutter-support radius.

This script MODIFIES NOTHING except the report file it writes. The
experiment it describes is deliberately wide-area: relocation anchors
trajectory STARTS near the radar, full trajectories are retained, and
--max-range-m in stage 6 bounds the clutter support only -- far targets
stay in the dataset as low-SNR targets via the range-decay model.

Usage:
    python scripts/06_audit_relocated_experiment.py
    python scripts/06_audit_relocated_experiment.py --coverage-range-m 100000 \
        --output reports/relocated_experiment_audit.md
    python scripts/06_audit_relocated_experiment.py --self-test
"""

import argparse
import glob
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRUTH_PATTERN = re.compile(r"radar_truth_(\d{4}-\d{2}-\d{2})\.csv$")
DETECTION_PATTERN = re.compile(r"detections_(\d{4}-\d{2}-\d{2})_thr_.+dB\.csv$")
SUMMARY_NAME = "sim_detection_summary.csv"

ANCHOR_BAND_M = (10_000.0, 80_000.0)   # the stage-5 default relocation anchor band
CHUNK_SIZE = 1_000_000


def _discover(directory: str, pattern: re.Pattern) -> list:
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, n) for n in os.listdir(directory) if pattern.search(n))


def _percentiles(values: np.ndarray) -> tuple:
    if values.size == 0:
        return (float("nan"),) * 3
    return tuple(float(v) for v in np.percentile(values, [50, 95, 99]))


# =============================================================================
# Truth-level audit (chunked)
# =============================================================================

def audit_truth(truth_dir: str, coverage_range_m: float) -> dict:
    files = _discover(truth_dir, TRUTH_PATTERN)
    if not files:
        raise SystemExit(f"No radar_truth_YYYY-MM-DD.csv files found in {truth_dir}")

    header = list(pd.read_csv(files[0], nrows=0).columns)
    has_relocated = "relocated" in header

    usecols = ["trajectory_id", "sample_idx", "range_m", "ground_range_m", "elevation_deg"]
    usecols += ["relocated"] if has_relocated else []

    total_rows = 0
    relocated_rows = 0
    rows_beyond = 0
    all_ids: set = set()
    ids_beyond: set = set()
    range_chunks, elev_chunks, first_gr_chunks = [], [], []

    for path in files:
        for chunk in pd.read_csv(path, usecols=usecols, dtype={"trajectory_id": str},
                                 chunksize=CHUNK_SIZE):
            total_rows += len(chunk)
            if has_relocated:
                relocated_rows += int((chunk["relocated"] == 1).sum())
            rng = chunk["range_m"].to_numpy()
            rows_beyond += int((rng > coverage_range_m).sum())
            all_ids.update(chunk["trajectory_id"].unique())
            ids_beyond.update(chunk.loc[rng > coverage_range_m, "trajectory_id"].unique())
            range_chunks.append(rng)
            elev_chunks.append(chunk["elevation_deg"].to_numpy())
            # every trajectory has a sample_idx == 0 row (stage-4 grid start)
            first_gr_chunks.append(
                chunk.loc[chunk["sample_idx"] == 0, "ground_range_m"].to_numpy())

    ranges = np.concatenate(range_chunks)
    elevations = np.concatenate(elev_chunks)
    first_ground = np.concatenate(first_gr_chunks)

    lo, hi = ANCHOR_BAND_M
    in_band = float(((first_ground >= lo) & (first_ground <= hi)).mean()) if first_ground.size else float("nan")

    return {
        "n_files": len(files),
        "total_rows": total_rows,
        "n_trajectories": len(all_ids),
        "relocated_fraction": (relocated_rows / total_rows) if (has_relocated and total_rows) else float("nan"),
        "relocated_column_present": has_relocated,
        "range_p": _percentiles(ranges),
        "elevation_p": _percentiles(elevations),
        "first_ground_range_p": _percentiles(first_ground),
        "frac_rows_beyond": rows_beyond / total_rows if total_rows else float("nan"),
        "frac_traj_ever_beyond": len(ids_beyond) / len(all_ids) if all_ids else float("nan"),
        "frac_first_in_anchor_band": in_band,
    }


# =============================================================================
# Detection-level audit (chunked; these files are large)
# =============================================================================

def audit_detections(detections_dir: str, coverage_range_m: float) -> dict:
    files = _discover(detections_dir, DETECTION_PATTERN)
    if not files:
        raise SystemExit(f"No detections_YYYY-MM-DD_thr_*dB.csv files found in {detections_dir}")

    total = targets = clutter = targets_beyond = 0
    for path in files:
        for chunk in pd.read_csv(path, usecols=["is_target", "truth_range_m"],
                                 chunksize=CHUNK_SIZE):
            total += len(chunk)
            is_tgt = chunk["is_target"] == 1
            targets += int(is_tgt.sum())
            clutter += int((~is_tgt).sum())
            targets_beyond += int((chunk.loc[is_tgt, "truth_range_m"] > coverage_range_m).sum())

    summary_path = os.path.join(detections_dir, SUMMARY_NAME)
    threshold_table = None
    if os.path.exists(summary_path):
        s = pd.read_csv(summary_path)
        s = s[s["status"] == "created"] if "status" in s.columns else s
        if not s.empty:
            threshold_table = (
                s.groupby("threshold_db")
                .agg(empirical_pd=("empirical_pd", "mean"),
                     false_alarm_per_frame=("false_alarm_per_frame", "mean"),
                     target_detections=("target_detections", "sum"),
                     clutter_detections=("clutter_detections", "sum"))
                .reset_index()
                .sort_values("threshold_db")
            )

    return {
        "n_files": len(files),
        "total_detections": total,
        "target_detections": targets,
        "clutter_detections": clutter,
        "frac_targets_beyond": targets_beyond / targets if targets else float("nan"),
        "threshold_table": threshold_table,
    }


# =============================================================================
# Report
# =============================================================================

def write_report(output_path: str, truth: dict, det: dict, coverage_range_m: float,
                 truth_dir: str, detections_dir: str) -> None:
    cov_km = coverage_range_m / 1000.0
    r50, r95, r99 = truth["range_p"]
    e50, e95, e99 = truth["elevation_p"]
    g50, g95, g99 = truth["first_ground_range_p"]

    lines = [
        "# Relocated Wide-Area Weak-Target Experiment Audit",
        "",
        f"- Truth: `{truth_dir}`",
        f"- Detections: `{detections_dir}`",
        f"- Coverage range audited against: **{cov_km:.0f} km**",
        "",
        "## Interpretation",
        "",
        "This is a **wide-area** weak-target experiment, not a range-contained",
        "radar-coverage experiment:",
        "",
        "- Relocation anchors each trajectory's **start** near the radar (first",
        "  sample in the 10-80 km anchor band); after that first sample the",
        "  aircraft follows its original ADS-B-derived motion unchanged, so the",
        "  **full trajectories are retained** and long flights drift well beyond",
        "  the anchor band.",
        f"- Stage 6's `--max-range-m {coverage_range_m:.0f}` defines the spatial",
        "  **clutter support** only -- where false alarms are generated. It does",
        f"  **not** remove target truth rows beyond {cov_km:.0f} km.",
        f"- Targets beyond {cov_km:.0f} km stay in the dataset and simply become",
        "  weaker: the range-decay SNR model hands them progressively lower SNR",
        "  and therefore lower detection probability.",
        "- This setup is suited to studying threshold trade-offs and weak-target",
        "  tracking over long trajectories. A future range-contained experiment",
        "  would need stage-5 post-relocation target filtering or a stage-6",
        "  target-range gate.",
        "",
        "## Stage 05 truth summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Truth files | {truth['n_files']} |",
        f"| Truth rows | {truth['total_rows']:,} |",
        f"| Trajectories | {truth['n_trajectories']:,} |",
        f"| Relocated fraction | {truth['relocated_fraction']:.3f} |"
        if truth["relocated_column_present"] else
        "| Relocated fraction | unavailable (column missing) |",
        f"| range_m p50/p95/p99 | {r50/1000:.1f} / {r95/1000:.1f} / {r99/1000:.1f} km |",
        f"| elevation_deg p50/p95/p99 | {e50:.2f} / {e95:.2f} / {e99:.2f} deg |",
        f"| First-sample ground range p50/p95/p99 | {g50/1000:.1f} / {g95/1000:.1f} / {g99/1000:.1f} km |",
        f"| First sample inside 10-80 km anchor band | {truth['frac_first_in_anchor_band']:.4f} |",
        "",
        "## Stage 06 detection summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Detection files | {det['n_files']} |",
        f"| Total detections | {det['total_detections']:,} |",
        f"| Target detections | {det['target_detections']:,} |",
        f"| Clutter detections | {det['clutter_detections']:,} |",
        "",
    ]

    if det["threshold_table"] is not None:
        lines += [
            "Per-threshold sweep (empirical Pd and false alarms averaged across days;",
            "detections summed):",
            "",
            "| threshold (dB) | empirical Pd | false alarms/frame | target detections | clutter detections |",
            "|---:|---:|---:|---:|---:|",
        ]
        for _, row in det["threshold_table"].iterrows():
            lines.append(
                f"| {row['threshold_db']:g} | {row['empirical_pd']:.3f} "
                f"| {row['false_alarm_per_frame']:.2f} "
                f"| {int(row['target_detections']):,} | {int(row['clutter_detections']):,} |")
        lines.append("")
    else:
        lines += ["_No sim_detection_summary.csv found; threshold table omitted._", ""]

    lines += [
        "## Coverage-range audit",
        "",
        f"How much of the experiment lives beyond the {cov_km:.0f} km clutter-support",
        "radius (retained by design, at reduced SNR):",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Fraction of truth rows beyond {cov_km:.0f} km | {truth['frac_rows_beyond']:.4f} |",
        f"| Fraction of trajectories ever beyond {cov_km:.0f} km | {truth['frac_traj_ever_beyond']:.4f} |",
        f"| Fraction of target detections beyond {cov_km:.0f} km | {det['frac_targets_beyond']:.4f} |",
        "",
        "## Notes for Stage 07",
        "",
        "Stage 07 should evaluate threshold-only detection performance both",
        "overall and, optionally, split by range bin:",
        "",
        "- 0-50 km",
        "- 50-100 km",
        "- 100-200 km",
        "- > 200 km",
        "",
        "The per-bin split will make the effect of the range-decay SNR model",
        "directly visible: near bins should show high Pd at every threshold,",
        "far bins should collapse toward the Pd floor.",
        "",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_audit(truth_dir: str, detections_dir: str, coverage_range_m: float, output: str) -> None:
    truth = audit_truth(truth_dir, coverage_range_m)
    det = audit_detections(detections_dir, coverage_range_m)
    write_report(output, truth, det, coverage_range_m, truth_dir, detections_dir)

    cov_km = coverage_range_m / 1000.0
    print(f"Audit report written to: {os.path.abspath(output)}")
    print(f"  truth: {truth['n_files']} files, {truth['total_rows']:,} rows, "
          f"{truth['n_trajectories']:,} trajectories, relocated fraction "
          f"{truth['relocated_fraction']:.3f}" if truth["relocated_column_present"]
          else f"  truth: {truth['n_files']} files, relocated column missing")
    print(f"  detections: {det['n_files']} files, {det['total_detections']:,} rows "
          f"({det['target_detections']:,} target / {det['clutter_detections']:,} clutter)")
    print(f"  beyond {cov_km:.0f} km: {truth['frac_rows_beyond']:.1%} of truth rows, "
          f"{truth['frac_traj_ever_beyond']:.1%} of trajectories ever, "
          f"{det['frac_targets_beyond']:.1%} of target detections")


# =============================================================================
# Self-test (no real data required)
# =============================================================================

def self_test() -> None:
    """Tiny synthetic truth + detections -> run the audit -> assert report content."""
    with tempfile.TemporaryDirectory() as tmp:
        truth_dir = os.path.join(tmp, "truth")
        det_dir = os.path.join(tmp, "det")
        os.makedirs(truth_dir)
        os.makedirs(det_dir)

        # Trajectory A stays inside 100 km; trajectory B starts inside the
        # 10-80 km anchor band but later exceeds 100 km.
        rows = []
        for i, r in enumerate([20_000, 25_000, 30_000, 35_000, 40_000]):
            rows.append(dict(trajectory_id="aaa_r0", sample_idx=i, range_m=float(r),
                             ground_range_m=float(r), elevation_deg=2.0, relocated=1))
        for i, r in enumerate([70_000, 90_000, 110_000, 130_000, 150_000]):
            rows.append(dict(trajectory_id="bbb_r0", sample_idx=i, range_m=float(r),
                             ground_range_m=float(r), elevation_deg=1.0, relocated=1))
        pd.DataFrame(rows).to_csv(os.path.join(truth_dir, "radar_truth_2022-01-01.csv"), index=False)

        # Detections for two thresholds: targets (one beyond coverage) + clutter.
        for thr, token in [(0.0, "0p0"), (6.0, "6p0")]:
            det_rows = [
                dict(is_target=1, truth_range_m=30_000.0),
                dict(is_target=1, truth_range_m=130_000.0),   # beyond coverage
                dict(is_target=0, truth_range_m=np.nan),
                dict(is_target=0, truth_range_m=np.nan),
            ]
            pd.DataFrame(det_rows).to_csv(
                os.path.join(det_dir, f"detections_2022-01-01_thr_{token}dB.csv"), index=False)
        pd.DataFrame([
            dict(date="2022-01-01", threshold_db=0.0, status="created", empirical_pd=0.8,
                 false_alarm_per_frame=20.0, target_detections=2, clutter_detections=2),
            dict(date="2022-01-01", threshold_db=6.0, status="created", empirical_pd=0.5,
                 false_alarm_per_frame=7.0, target_detections=2, clutter_detections=2),
        ]).to_csv(os.path.join(det_dir, SUMMARY_NAME), index=False)

        report = os.path.join(tmp, "reports", "audit.md")
        run_audit(truth_dir, det_dir, 100_000.0, report)

        assert os.path.exists(report), "report file was not written"
        text = open(report).read()
        for needle in ["wide-area", "clutter support", "Fraction of truth rows", "threshold"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        # sanity of the numbers on this hand-built fixture
        assert "| Trajectories | 2 |" in text
        assert "| Fraction of trajectories ever beyond 100 km | 0.5000 |" in text
        assert "| Fraction of truth rows beyond 100 km | 0.3000 |" in text          # 3 of 10
        assert "| Fraction of target detections beyond 100 km | 0.5000 |" in text   # 2 of 4

    print("\nRelocated experiment audit self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the relocated wide-area weak-target experiment.")
    parser.add_argument("--truth-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    parser.add_argument("--detections-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"))
    parser.add_argument("--coverage-range-m", type=float, default=100_000.0)
    parser.add_argument("--output", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "relocated_experiment_audit.md"))
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_audit(args.truth_dir, args.detections_dir, args.coverage_range_m, args.output)


if __name__ == "__main__":
    main()
