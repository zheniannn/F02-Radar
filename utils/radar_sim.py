"""Simulate thresholded radar point detections from stage 5's noiseless
radar-coordinate truth.

Stage 6 operates at the POINT-DETECTION level only: per truth row it draws a
target SNR, converts it to a detection probability against each threshold,
and emits either a noisy measurement (detected) or nothing (missed); per
frame it adds Poisson-distributed clutter false alarms. It does NOT simulate
raw RF/IQ data, pulse compression, or range-Doppler images, and it does not
run trackers, evaluate tracking, train models, or split train/test -- those
belong to later stages.

Reproducibility: every (day, threshold) pair gets its own child RNG whose
seed is derived deterministically from (base seed, date, threshold,
scenario_id) via sha256 -- never Python's hash(), which is not stable across
processes. Re-running any subset of days/thresholds reproduces identical
output regardless of order.
"""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

INPUT_PREFIX = "radar_truth_"
INPUT_SUFFIX = ".csv"
OUTPUT_PREFIX = "detections_"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

REQUIRED_INPUT_COLUMNS = [
    "date", "radar_name", "icao24", "trajectory_id", "source_segment_id",
    "sample_idx", "timestamp",
    "range_m", "azimuth_rad", "azimuth_deg", "elevation_rad", "elevation_deg",
    "ground_range_m", "radial_velocity_mps", "east_m", "north_m", "up_m",
    "speed_enu_mps", "speed_stage4_mps", "is_interpolated",
    "trajectory_start_time", "trajectory_end_time", "trajectory_duration_s", "n_samples",
]

# Stage-5 radar-site columns, carried through when present.
RADAR_SITE_COLUMNS = ["radar_lat_deg", "radar_lon_deg", "radar_alt_m"]

OUTPUT_COLUMNS = [
    "date", "scenario_id", "threshold_db", "frame_id", "timestamp",
    "detection_id", "is_target",
    "icao24", "trajectory_id", "source_segment_id", "sample_idx",
    "truth_range_m", "truth_azimuth_rad", "truth_elevation_rad", "truth_radial_velocity_mps",
    "meas_range_m", "meas_azimuth_rad", "meas_azimuth_deg",
    "meas_elevation_rad", "meas_elevation_deg", "meas_radial_velocity_mps",
    "range_error_m", "azimuth_error_rad", "elevation_error_rad", "radial_velocity_error_mps",
    "snr_db", "pd", "clutter_density_per_frame",
    "radar_name", "radar_lat_deg", "radar_lon_deg", "radar_alt_m",
]

SUMMARY_COLUMNS = [
    "date", "scenario_id", "threshold_db", "status",
    "truth_rows", "frames",
    "target_detections", "missed_targets", "empirical_pd",
    "clutter_detections", "total_detections", "false_alarm_per_frame",
    "median_snr_target_db", "p10_snr_target_db", "p90_snr_target_db",
    "output_file",
]


@dataclass
class RadarSimConfig:
    """All stage-6 tunables in one place (populated from the CLI)."""
    thresholds_db: List[float] = field(default_factory=lambda: [-5.0, 0.0, 3.0, 6.0, 9.0, 12.0])
    scenario_id: str = "default"
    seed: int = 42
    overwrite: bool = False

    # Target SNR model
    snr_model: str = "range_decay"           # or "constant"
    target_snr_ref_db: float = 10.0
    snr_ref_range_m: float = 50_000.0
    snr_range_power: float = 4.0             # 10*power*log10 range decay (4 ~= radar equation R^-4)
    target_snr_std_db: float = 3.0
    target_snr_min_db: float = -20.0
    target_snr_max_db: float = 30.0

    # Detection probability (logistic in SNR-vs-threshold)
    pd_transition_width_db: float = 2.0
    pd_max: float = 0.98
    pd_min: float = 0.01

    # Measurement noise (1-sigma)
    sigma_range_m: float = 75.0
    sigma_azimuth_deg: float = 0.15
    sigma_elevation_deg: float = 0.15
    sigma_radial_velocity_mps: float = 2.0

    # Clutter model
    clutter_model: str = "poisson"
    clutter_rate_ref: float = 20.0           # expected false alarms/frame at the reference threshold
    clutter_ref_threshold_db: float = 0.0
    clutter_threshold_scale_db: float = 6.0
    max_range_m: Optional[float] = None      # None -> inferred from the day's truth (printed)
    min_range_m: float = 0.0
    min_elevation_deg: float = 0.0
    max_elevation_deg: float = 20.0
    min_radial_velocity_mps: float = -120.0
    max_radial_velocity_mps: float = 120.0
    clutter_snr_scale_db: float = 6.0        # Exponential scale for above-threshold clutter SNR


# =============================================================================
# Discovery / naming / seeding
# =============================================================================

def discover_radar_truth_files(input_dir: str) -> List[Tuple[str, str]]:
    """Return sorted (date, path) pairs for every stage-5 truth CSV in input_dir."""
    results = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith(INPUT_PREFIX) and name.endswith(INPUT_SUFFIX)):
            continue
        match = DATE_PATTERN.search(name)
        if not match:
            print(f"WARNING: no date pattern found in filename '{name}'; skipping.")
            continue
        results.append((match.group(1), os.path.join(input_dir, name)))
    return results


def threshold_to_token(threshold_db: float) -> str:
    """Filename-safe threshold token: 6.0 -> '6p0', -5.0 -> 'm5p0'."""
    return f"{threshold_db:.1f}".replace("-", "m").replace(".", "p")


def derive_seed(base_seed: int, date: str, threshold_db: float, scenario_id: str) -> int:
    """Deterministic 64-bit child seed from (base seed, date, threshold,
    scenario). sha256-based so it's stable across processes and platforms."""
    key = f"{base_seed}|{date}|{threshold_db:.6f}|{scenario_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def validate_input_columns(path: str) -> bool:
    """Check the stage-5 required columns; return whether the radar-site
    lat/lon/alt columns are also present (they are carried through)."""
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Missing required stage-5 column(s) {missing} in {path}")
    return all(c in columns for c in RADAR_SITE_COLUMNS)


# =============================================================================
# SNR / detection probability
# =============================================================================

def compute_target_snr(range_m: np.ndarray, cfg: RadarSimConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-row target SNR in dB: constant or range-decay mean plus Gaussian
    scintillation, clamped to [target_snr_min_db, target_snr_max_db]."""
    noise = rng.normal(0.0, cfg.target_snr_std_db, size=len(range_m))
    if cfg.snr_model == "constant":
        snr = cfg.target_snr_ref_db + noise
    elif cfg.snr_model == "range_decay":
        r = np.clip(np.asarray(range_m, dtype=float), 1.0, None)   # guard log10(0)
        snr = (cfg.target_snr_ref_db
               - 10.0 * cfg.snr_range_power * np.log10(r / cfg.snr_ref_range_m)
               + noise)
    else:
        raise ValueError(f"Unknown snr-model '{cfg.snr_model}'")
    return np.clip(snr, cfg.target_snr_min_db, cfg.target_snr_max_db)


def compute_pd(snr_db: np.ndarray, threshold_db: float, cfg: RadarSimConfig) -> np.ndarray:
    """Logistic detection probability vs (SNR - threshold)."""
    return cfg.pd_min + (cfg.pd_max - cfg.pd_min) / (
        1.0 + np.exp(-(np.asarray(snr_db, dtype=float) - threshold_db) / cfg.pd_transition_width_db)
    )


# =============================================================================
# Simulation
# =============================================================================

def simulate_target_detections(
    df_truth: pd.DataFrame, threshold_db: float, cfg: RadarSimConfig, rng: np.random.Generator
) -> Tuple[pd.DataFrame, int]:
    """Draw SNR -> Pd -> Bernoulli detection per truth row; emit noisy
    measurements for detected rows. Returns (detections_df, n_missed).

    df_truth must already carry a 'frame_id' column.
    """
    n = len(df_truth)
    snr = compute_target_snr(df_truth["range_m"].to_numpy(), cfg, rng)
    pd_vals = compute_pd(snr, threshold_db, cfg)
    detected = rng.random(n) < pd_vals
    n_detected = int(detected.sum())

    d = df_truth[detected]
    truth_range = d["range_m"].to_numpy(dtype=float)
    truth_az = d["azimuth_rad"].to_numpy(dtype=float)
    truth_el = d["elevation_rad"].to_numpy(dtype=float)
    truth_rv = d["radial_velocity_mps"].to_numpy(dtype=float)

    range_noise = rng.normal(0.0, cfg.sigma_range_m, n_detected)
    az_noise = rng.normal(0.0, np.radians(cfg.sigma_azimuth_deg), n_detected)
    el_noise = rng.normal(0.0, np.radians(cfg.sigma_elevation_deg), n_detected)
    rv_noise = rng.normal(0.0, cfg.sigma_radial_velocity_mps, n_detected)

    meas_range = np.maximum(truth_range + range_noise, 0.0)
    meas_az = np.mod(truth_az + az_noise, 2.0 * np.pi)
    meas_el = np.clip(truth_el + el_noise, -np.pi / 2, np.pi / 2)
    meas_rv = truth_rv + rv_noise

    # Errors are meas - truth AFTER postprocessing (so clipping shows up);
    # azimuth uses a wrap-aware difference so a wrapped measurement doesn't
    # report a ~2*pi error.
    az_error = np.mod(meas_az - truth_az + np.pi, 2.0 * np.pi) - np.pi

    out = pd.DataFrame({
        "frame_id": d["frame_id"].to_numpy(),
        "timestamp": d["timestamp"].to_numpy(),
        "is_target": 1,
        "icao24": d["icao24"].to_numpy(),
        "trajectory_id": d["trajectory_id"].to_numpy(),
        "source_segment_id": d["source_segment_id"].to_numpy(),
        "sample_idx": d["sample_idx"].to_numpy(),
        "truth_range_m": truth_range,
        "truth_azimuth_rad": truth_az,
        "truth_elevation_rad": truth_el,
        "truth_radial_velocity_mps": truth_rv,
        "meas_range_m": meas_range,
        "meas_azimuth_rad": meas_az,
        "meas_elevation_rad": meas_el,
        "meas_radial_velocity_mps": meas_rv,
        "range_error_m": meas_range - truth_range,
        "azimuth_error_rad": az_error,
        "elevation_error_rad": meas_el - truth_el,
        "radial_velocity_error_mps": rv_noise,
        "snr_db": snr[detected],
        "pd": pd_vals[detected],
    })
    return out, n - n_detected


def simulate_clutter_for_day(
    frames: pd.DataFrame, threshold_db: float, cfg: RadarSimConfig,
    rng: np.random.Generator, max_range_m: float
) -> Tuple[pd.DataFrame, float]:
    """Poisson false alarms per frame, uniform in range/azimuth/elevation/
    radial velocity, with above-threshold exponential SNR (only clutter that
    passed the threshold is ever written). Returns (clutter_df, lambda)."""
    lam = cfg.clutter_rate_ref * np.exp(
        -(threshold_db - cfg.clutter_ref_threshold_db) / cfg.clutter_threshold_scale_db
    )
    counts = rng.poisson(lam, size=len(frames))
    total = int(counts.sum())
    frame_pos = np.repeat(np.arange(len(frames)), counts)

    meas_az = rng.uniform(0.0, 2.0 * np.pi, total)
    meas_el = np.radians(rng.uniform(cfg.min_elevation_deg, cfg.max_elevation_deg, total))

    out = pd.DataFrame({
        "frame_id": frames["frame_id"].to_numpy()[frame_pos],
        "timestamp": frames["timestamp"].to_numpy()[frame_pos],
        "is_target": 0,
        "icao24": np.nan,
        "trajectory_id": np.nan,
        "source_segment_id": np.nan,
        "sample_idx": np.nan,
        "truth_range_m": np.nan,
        "truth_azimuth_rad": np.nan,
        "truth_elevation_rad": np.nan,
        "truth_radial_velocity_mps": np.nan,
        "meas_range_m": rng.uniform(cfg.min_range_m, max_range_m, total),
        "meas_azimuth_rad": meas_az,
        "meas_elevation_rad": meas_el,
        "meas_radial_velocity_mps": rng.uniform(
            cfg.min_radial_velocity_mps, cfg.max_radial_velocity_mps, total),
        "range_error_m": np.nan,
        "azimuth_error_rad": np.nan,
        "elevation_error_rad": np.nan,
        "radial_velocity_error_mps": np.nan,
        "snr_db": threshold_db + rng.exponential(cfg.clutter_snr_scale_db, total),
        "pd": np.nan,
    })
    return out, float(lam)


def simulate_day(date: str, input_path: str, output_dir: str, cfg: RadarSimConfig) -> List[Dict]:
    """Simulate every configured threshold for one day's truth file.
    Returns one summary dict per threshold (status 'created' or 'skipped')."""
    has_site = validate_input_columns(input_path)

    usecols = ["icao24", "trajectory_id", "source_segment_id", "sample_idx", "timestamp",
               "range_m", "azimuth_rad", "elevation_rad", "radial_velocity_mps", "radar_name"]
    usecols += RADAR_SITE_COLUMNS if has_site else []
    df_truth = pd.read_csv(input_path, usecols=usecols,
                           dtype={c: str for c in ["icao24", "trajectory_id", "source_segment_id"]},
                           low_memory=False)

    # A frame is a unique timestamp within the day; frame_id is its stable
    # index after ascending sort. Multiple aircraft share a frame.
    unique_ts = np.sort(df_truth["timestamp"].unique())
    frames = pd.DataFrame({"frame_id": np.arange(len(unique_ts)), "timestamp": unique_ts})
    df_truth = df_truth.assign(frame_id=np.searchsorted(unique_ts, df_truth["timestamp"].to_numpy()))

    if cfg.max_range_m is not None:
        max_range_m = float(cfg.max_range_m)
    else:
        max_range_m = float(np.ceil(df_truth["range_m"].max() / 10_000.0) * 10_000.0)
        print(f"[{date}] --max-range-m not provided: inferred {max_range_m:.0f} m "
              f"from the day's truth (max range rounded up to nearest 10 km)")

    radar_name = df_truth["radar_name"].iloc[0]
    radar_lat = float(df_truth["radar_lat_deg"].iloc[0]) if has_site else np.nan
    radar_lon = float(df_truth["radar_lon_deg"].iloc[0]) if has_site else np.nan
    radar_alt = float(df_truth["radar_alt_m"].iloc[0]) if has_site else np.nan

    results = []
    for threshold_db in cfg.thresholds_db:
        output_path = os.path.join(
            output_dir, f"{OUTPUT_PREFIX}{date}_thr_{threshold_to_token(threshold_db)}dB.csv")

        if os.path.exists(output_path) and not cfg.overwrite:
            print(f"[{date} thr={threshold_db:g}dB] output exists, skipping "
                  f"(pass --overwrite to regenerate): {output_path}")
            results.append({"date": date, "scenario_id": cfg.scenario_id,
                            "threshold_db": threshold_db, "status": "skipped",
                            "truth_rows": None, "frames": None,
                            "target_detections": None, "missed_targets": None,
                            "empirical_pd": None, "clutter_detections": None,
                            "total_detections": None, "false_alarm_per_frame": None,
                            "median_snr_target_db": None, "p10_snr_target_db": None,
                            "p90_snr_target_db": None,
                            "output_file": os.path.abspath(output_path)})
            continue

        rng = np.random.default_rng(derive_seed(cfg.seed, date, threshold_db, cfg.scenario_id))

        targets, n_missed = simulate_target_detections(df_truth, threshold_db, cfg, rng)
        clutter, lam = simulate_clutter_for_day(frames, threshold_db, cfg, rng, max_range_m)

        out = pd.concat([targets, clutter], ignore_index=True)
        out = out.sort_values(["frame_id", "is_target"], ascending=[True, False],
                              kind="mergesort").reset_index(drop=True)
        out.insert(0, "date", date)
        out.insert(1, "scenario_id", cfg.scenario_id)
        out.insert(2, "threshold_db", threshold_db)
        out["detection_id"] = np.arange(len(out))
        out["meas_azimuth_deg"] = np.degrees(out["meas_azimuth_rad"])
        out["meas_elevation_deg"] = np.degrees(out["meas_elevation_rad"])
        out["clutter_density_per_frame"] = lam
        out["radar_name"] = radar_name
        out["radar_lat_deg"] = radar_lat
        out["radar_lon_deg"] = radar_lon
        out["radar_alt_m"] = radar_alt
        out = out[OUTPUT_COLUMNS]

        out.to_csv(output_path, index=False)

        target_snr = out.loc[out["is_target"] == 1, "snr_db"].to_numpy()
        n_targets = len(target_snr)
        results.append({
            "date": date,
            "scenario_id": cfg.scenario_id,
            "threshold_db": threshold_db,
            "status": "created",
            "truth_rows": len(df_truth),
            "frames": len(frames),
            "target_detections": n_targets,
            "missed_targets": n_missed,
            "empirical_pd": n_targets / len(df_truth) if len(df_truth) else float("nan"),
            "clutter_detections": len(clutter),
            "total_detections": len(out),
            "false_alarm_per_frame": len(clutter) / len(frames) if len(frames) else float("nan"),
            "median_snr_target_db": float(np.median(target_snr)) if n_targets else float("nan"),
            "p10_snr_target_db": float(np.percentile(target_snr, 10)) if n_targets else float("nan"),
            "p90_snr_target_db": float(np.percentile(target_snr, 90)) if n_targets else float("nan"),
            "output_file": os.path.abspath(output_path),
        })
    return results


# =============================================================================
# Validation gate
# =============================================================================

_GATE_USECOLS = [
    "is_target", "trajectory_id",
    "truth_range_m", "meas_range_m", "meas_azimuth_rad", "meas_elevation_rad", "snr_db",
    "range_error_m", "azimuth_error_rad", "elevation_error_rad", "radial_velocity_error_mps",
]


def run_validation_gate(output_paths: List[str], summary_df: pd.DataFrame) -> None:
    """Post-run checks on the written outputs; raises ValueError on failure."""

    def fail(message: str) -> None:
        raise ValueError(f"Stage 06 validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    n_created = int((summary_df["status"] == "created").sum())
    n_skipped = int((summary_df["status"] == "skipped").sum())
    if n_created + n_skipped == 0:
        fail("no output file was created or skipped as existing")
    print(f"  outputs: {n_created} created, {n_skipped} skipped (already existed)")

    missing_summary = [c for c in SUMMARY_COLUMNS if c not in summary_df.columns]
    if missing_summary:
        fail(f"summary is missing column(s): {missing_summary}")
    print("  summary columns present: OK")

    for path in output_paths:
        if not os.path.exists(path):
            fail(f"expected output file does not exist: {path}")
        header = list(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in OUTPUT_COLUMNS if c not in header]
        if missing:
            fail(f"{os.path.basename(path)}: missing required column(s) {missing}")

        df = pd.read_csv(path, usecols=_GATE_USECOLS, low_memory=False)
        name = os.path.basename(path)

        if not df["is_target"].isin([0, 1]).all():
            fail(f"{name}: is_target contains values other than 0/1")
        r = df["meas_range_m"].to_numpy()
        if not (np.isfinite(r).all() and (r >= 0).all()):
            fail(f"{name}: meas_range_m non-finite or negative")
        az = df["meas_azimuth_rad"].to_numpy()
        if not ((az >= 0).all() and (az < 2 * np.pi).all()):
            fail(f"{name}: meas_azimuth_rad outside [0, 2*pi)")
        el = df["meas_elevation_rad"].to_numpy()
        if not ((el >= -np.pi / 2).all() and (el <= np.pi / 2).all()):
            fail(f"{name}: meas_elevation_rad outside [-pi/2, pi/2]")
        if not np.isfinite(df["snr_db"].to_numpy()).all():
            fail(f"{name}: snr_db contains non-finite values")

        is_target = df["is_target"] == 1
        tgt = df[is_target]
        if len(tgt):
            traj = tgt["trajectory_id"]
            if not (traj.notna() & (traj.astype(str).str.strip() != "")).all():
                fail(f"{name}: target rows with blank trajectory_id")
            if not np.isfinite(tgt["truth_range_m"].to_numpy()).all():
                fail(f"{name}: target rows with non-finite truth_range_m")
            for col in ["range_error_m", "azimuth_error_rad",
                        "elevation_error_rad", "radial_velocity_error_mps"]:
                if not np.isfinite(tgt[col].to_numpy()).all():
                    fail(f"{name}: target rows with non-finite {col}")
        clu = df[~is_target]
        if len(clu):
            if not clu["trajectory_id"].isna().all():
                fail(f"{name}: clutter rows with a non-blank trajectory_id")
            if not clu["truth_range_m"].isna().all():
                fail(f"{name}: clutter rows with finite truth_range_m")
    print(f"  per-file label/bounds/finiteness checks over {len(output_paths)} file(s): OK")

    # Report-only monotonic-trend check (finite random samples may wiggle).
    created = summary_df[summary_df["status"] == "created"]
    if not created.empty:
        print("\n  threshold trends per day (report-only; expect both to generally decrease):")
        for date, group in created.groupby("date"):
            g = group.sort_values("threshold_db")
            pd_str = " -> ".join(f"{v:.3f}" for v in g["empirical_pd"])
            fa_str = " -> ".join(f"{v:.2f}" for v in g["false_alarm_per_frame"])
            thr_str = ", ".join(f"{v:g}" for v in g["threshold_db"])
            print(f"    [{date}] thresholds dB: {thr_str}")
            print(f"      empirical_pd:         {pd_str}")
            print(f"      false_alarm_per_frame: {fa_str}")


# =============================================================================
# Self-test (no real data required)
# =============================================================================

def self_test() -> None:
    """End-to-end check on a tiny synthetic stage-5-like truth file: two
    trajectories x five timestamps, thresholds [-5, 6, 12], fixed seed."""
    date = "2022-01-01"
    t = 1_000_000.0 + 10.0 * np.arange(5)

    rows = []
    for k, (traj, rng0, az, el, rv) in enumerate([
        ("aaa111_1000000_r0", 20_000.0, 0.5, 0.05, 60.0),
        ("bbb222_1000000_r0", 40_000.0, 2.0, 0.10, -60.0),
    ]):
        for i in range(5):
            rows.append({
                "date": date, "radar_name": "selftest",
                "radar_lat_deg": 45.0, "radar_lon_deg": 7.0, "radar_alt_m": 200.0,
                "icao24": f"{'ab'[k]*6}", "trajectory_id": traj,
                "source_segment_id": traj.rsplit("_", 1)[0], "sample_idx": i,
                "timestamp": t[i],
                "range_m": rng0 + 100.0 * i, "azimuth_rad": az,
                "azimuth_deg": np.degrees(az), "elevation_rad": el,
                "elevation_deg": np.degrees(el), "ground_range_m": rng0,
                "radial_velocity_mps": rv, "east_m": 1000.0, "north_m": 1000.0, "up_m": 500.0,
                "speed_enu_mps": 60.0, "speed_stage4_mps": 60.0, "is_interpolated": False,
                "trajectory_start_time": t[0], "trajectory_end_time": t[-1],
                "trajectory_duration_s": 40.0, "n_samples": 5,
            })

    cfg = RadarSimConfig(
        thresholds_db=[-5.0, 6.0, 12.0], seed=123,
        snr_model="constant", target_snr_ref_db=10.0, target_snr_std_db=3.0,
        clutter_rate_ref=10.0, max_range_m=60_000.0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        in_dir, out_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        pd.DataFrame(rows).to_csv(os.path.join(in_dir, f"{INPUT_PREFIX}{date}{INPUT_SUFFIX}"), index=False)

        (found_date, path), = discover_radar_truth_files(in_dir)
        results = simulate_day(found_date, path, out_dir, cfg)
        summary = pd.DataFrame(results, columns=SUMMARY_COLUMNS)
        by_thr = {r["threshold_db"]: r for r in results}

        for r in results:
            assert os.path.exists(r["output_file"]), f"missing output {r['output_file']}"
        assert by_thr[-5.0]["empirical_pd"] >= by_thr[12.0]["empirical_pd"], \
            "empirical Pd catastrophically inverted across thresholds"
        assert by_thr[-5.0]["clutter_detections"] >= by_thr[12.0]["clutter_detections"], \
            "clutter should not increase with threshold"

        out = pd.read_csv(by_thr[-5.0]["output_file"])
        missing = [c for c in OUTPUT_COLUMNS if c not in out.columns]
        assert not missing, f"missing output columns: {missing}"
        assert out["is_target"].isin([0, 1]).all(), "invalid is_target labels"
        assert (out["meas_range_m"] >= 0).all(), "negative measured range"
        assert ((out["meas_azimuth_rad"] >= 0) & (out["meas_azimuth_rad"] < 2 * np.pi)).all(), \
            "measured azimuth out of bounds"
        assert out["meas_elevation_rad"].between(-np.pi / 2, np.pi / 2).all(), \
            "measured elevation out of bounds"
        tgt = out[out["is_target"] == 1]
        assert tgt["trajectory_id"].notna().all(), "target row lost its trajectory_id"
        assert out.loc[out["is_target"] == 0, "truth_range_m"].isna().all(), \
            "clutter row has truth range"

        run_validation_gate([r["output_file"] for r in results], summary)

    print("\nStage 06 self-test passed.")
