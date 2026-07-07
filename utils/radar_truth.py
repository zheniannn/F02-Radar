"""Convert stage 4's uniformly-sampled trajectories into noiseless
radar-coordinate ground truth relative to a fixed, user-supplied radar site.

This stage does NOT simulate radar noise, inject clutter, apply detection
thresholds, generate range-Doppler images, or build ML datasets -- those
belong to later stages. It only re-expresses each trajectory point as
east/north/up + range/azimuth/elevation (+ ENU and radial velocities) with
respect to the radar origin, using exact WGS84 geometry (see utils/geo.py).

The radar location is never invented: it must arrive via the CLI.
"""

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.geo import geodetic_to_enu, wrap_angle_2pi

INPUT_PREFIX = "states_"
INPUT_SUFFIX = "_conventionalGA_trajectories_10s.csv"
OUTPUT_PREFIX = "radar_truth_"
OUTPUT_SUFFIX = ".csv"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Stage-4 columns stage 5 depends on, beyond the position triple.
BASE_REQUIRED_COLUMNS = [
    "icao24", "trajectory_id", "source_segment_id", "sample_idx",
    "timestamp", "dt_s", "is_interpolated",
    "speed_mps", "accel_mps2", "turn_rate_deg_s",
    "trajectory_start_time", "trajectory_end_time", "trajectory_duration_s",
    "n_samples",
]

POSITION_COLUMNS = {
    "smooth": ("lat_smooth", "lon_smooth", "alt_smooth"),
    "interp": ("lat_interp", "lon_interp", "alt_interp"),
}

# Hard-fail threshold for the stage4-vs-ENU speed consistency check.
MAX_MEDIAN_SPEED_DIFF_MPS = 20.0


@dataclass
class RadarTruthConfig:
    """All stage-5 tunables in one place (populated from the CLI)."""
    radar_lat: float
    radar_lon: float
    radar_alt_m: float = 0.0
    radar_name: str = "radar"
    position_source: str = "smooth"      # which stage-4 position triple to use
    min_range_m: float = 0.0
    max_range_m: Optional[float] = None
    drop_below_horizon: bool = False
    overwrite: bool = False


# =============================================================================
# Discovery / validation
# =============================================================================

def discover_input_files(input_dir: str) -> List[Tuple[str, str]]:
    """Return sorted (date, path) pairs for every stage-4 trajectories CSV in input_dir."""
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


def validate_columns(path: str, position_source: str) -> Tuple[Tuple[str, str, str], bool]:
    """Peek a CSV's header; return (position_columns, has_callsign).

    Fails clearly if the requested position source's columns, or any other
    required stage-4 column, are missing.
    """
    if position_source not in POSITION_COLUMNS:
        raise ValueError(f"Unknown position source '{position_source}' (choices: smooth, interp)")

    columns = list(pd.read_csv(path, nrows=0).columns)

    pos_cols = POSITION_COLUMNS[position_source]
    missing_pos = [c for c in pos_cols if c not in columns]
    if missing_pos:
        raise ValueError(
            f"--position-source {position_source} requested but column(s) {missing_pos} "
            f"are missing from {path}"
        )

    missing = [c for c in BASE_REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Missing required stage-4 column(s) {missing} in {path}")

    return pos_cols, "callsign" in columns


# =============================================================================
# Numerics
# =============================================================================

def compute_trajectory_velocities(df: pd.DataFrame, east, north, up):
    """Finite-difference ENU velocities per trajectory via np.gradient over
    each trajectory's timestamps (centered differences in the interior,
    one-sided at the ends). Trajectories with fewer than 2 samples get NaN
    (stage 4's min-points filter makes that all but impossible).
    """
    t = df["timestamp"].to_numpy(dtype=float)
    ve = np.full(len(df), np.nan)
    vn = np.full(len(df), np.nan)
    vu = np.full(len(df), np.nan)

    for _, positions in df.groupby("trajectory_id", sort=False).indices.items():
        if len(positions) < 2:
            continue
        tt = t[positions]
        ve[positions] = np.gradient(east[positions], tt)
        vn[positions] = np.gradient(north[positions], tt)
        vu[positions] = np.gradient(up[positions], tt)

    return ve, vn, vu


# =============================================================================
# Per-day orchestration
# =============================================================================

def make_radar_truth_for_day(date: str, input_path: str, output_dir: str, cfg: RadarTruthConfig) -> Dict:
    """Convert one day's trajectories to radar truth. Returns the summary
    dict for the day; status is 'created' or 'skipped' (existing output and
    no --overwrite)."""
    output_path = os.path.join(output_dir, f"{OUTPUT_PREFIX}{date}{OUTPUT_SUFFIX}")

    if os.path.exists(output_path) and not cfg.overwrite:
        print(f"[{date}] output already exists, skipping (pass --overwrite to regenerate): {output_path}")
        return {"date": date, "status": "skipped", "input_rows": None, "output_rows": None,
                "unique_trajectories_in": None, "unique_trajectories_out": None,
                "rows_dropped_min_range": None, "rows_dropped_max_range": None,
                "rows_dropped_below_horizon": None,
                "output_file": os.path.abspath(output_path), "_final_df": None}

    pos_cols, has_callsign = validate_columns(input_path, cfg.position_source)

    read_cols = BASE_REQUIRED_COLUMNS + list(pos_cols) + (["callsign"] if has_callsign else [])
    string_cols = {c: str for c in ["icao24", "trajectory_id", "source_segment_id"]
                   + (["callsign"] if has_callsign else [])}
    df = pd.read_csv(input_path, usecols=read_cols, dtype=string_cols, low_memory=False)

    input_rows = len(df)
    unique_in = int(df["trajectory_id"].nunique())

    lat = df[pos_cols[0]].to_numpy(dtype=float)
    lon = df[pos_cols[1]].to_numpy(dtype=float)
    alt = df[pos_cols[2]].to_numpy(dtype=float)

    # Exact WGS84 geometry (no flat-earth approximation).
    east, north, up = geodetic_to_enu(lat, lon, alt, cfg.radar_lat, cfg.radar_lon, cfg.radar_alt_m)

    ground_range = np.hypot(east, north)
    rng = np.sqrt(east**2 + north**2 + up**2)
    azimuth_rad = wrap_angle_2pi(np.arctan2(east, north))   # 0 = north, pi/2 = east
    elevation_rad = np.arctan2(up, ground_range)

    # Velocities are computed on FULL trajectories, before any row filtering,
    # so a filtered-out neighbor can't corrupt a kept row's velocity.
    ve, vn, vu = compute_trajectory_velocities(df, east, north, up)
    speed_enu = np.sqrt(ve**2 + vn**2 + vu**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        # dot(v, line-of-sight unit vector); positive = range increasing.
        radial = np.where(rng > 0, (east * ve + north * vn + up * vu) / rng, np.nan)

    # Row-level filtering (never trajectory-level).
    drop_min = rng < cfg.min_range_m
    drop_max = rng > cfg.max_range_m if cfg.max_range_m is not None else np.zeros(len(df), dtype=bool)
    drop_horizon = elevation_rad < 0.0 if cfg.drop_below_horizon else np.zeros(len(df), dtype=bool)
    keep = ~(drop_min | drop_max | drop_horizon)

    out = pd.DataFrame({
        "date": date,
        "radar_name": cfg.radar_name,
        "radar_lat_deg": cfg.radar_lat,
        "radar_lon_deg": cfg.radar_lon,
        "radar_alt_m": cfg.radar_alt_m,
        "icao24": df["icao24"],
        **({"callsign": df["callsign"]} if has_callsign else {}),
        "trajectory_id": df["trajectory_id"],
        "source_segment_id": df["source_segment_id"],
        "sample_idx": df["sample_idx"],
        "timestamp": df["timestamp"],
        "dt_s": df["dt_s"],
        "lat_deg": lat,
        "lon_deg": lon,
        "alt_m": alt,
        "east_m": east,
        "north_m": north,
        "up_m": up,
        "range_m": rng,
        "azimuth_rad": azimuth_rad,
        "azimuth_deg": np.degrees(azimuth_rad),
        "elevation_rad": elevation_rad,
        "elevation_deg": np.degrees(elevation_rad),
        "ground_range_m": ground_range,
        "ve_mps": ve,
        "vn_mps": vn,
        "vu_mps": vu,
        "radial_velocity_mps": radial,
        "speed_enu_mps": speed_enu,
        "speed_stage4_mps": df["speed_mps"],
        "accel_stage4_mps2": df["accel_mps2"],
        "turn_rate_stage4_deg_s": df["turn_rate_deg_s"],
        "is_interpolated": df["is_interpolated"],
        "trajectory_start_time": df["trajectory_start_time"],
        "trajectory_end_time": df["trajectory_end_time"],
        "trajectory_duration_s": df["trajectory_duration_s"],
        "n_samples": df["n_samples"],
    })[keep].reset_index(drop=True)

    out.to_csv(output_path, index=False)

    return {
        "date": date,
        "status": "created",
        "input_rows": input_rows,
        "output_rows": len(out),
        "unique_trajectories_in": unique_in,
        "unique_trajectories_out": int(out["trajectory_id"].nunique()) if not out.empty else 0,
        "rows_dropped_min_range": int(drop_min.sum()),
        "rows_dropped_max_range": int(drop_max.sum()),
        "rows_dropped_below_horizon": int(drop_horizon.sum()),
        "output_file": os.path.abspath(output_path),
        # kept only for the validation gate, not written to the summary CSV
        "_final_df": out,
    }


def summarize_day(day_result: Dict) -> Dict:
    """The summary-CSV row for one day (everything except private fields)."""
    return {k: v for k, v in day_result.items() if not k.startswith("_")}


# =============================================================================
# Validation gate
# =============================================================================

def run_validation_gate(day_results: List[Dict]) -> None:
    """Post-run checks; raises ValueError with a clear message on failure."""

    def fail(message: str) -> None:
        raise ValueError(f"Stage 05 validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    created = [r for r in day_results if r["status"] == "created"]
    skipped = [r for r in day_results if r["status"] == "skipped"]
    if not created and not skipped:
        fail("no output file was created or skipped as existing")
    print(f"  outputs: {len(created)} created, {len(skipped)} skipped (already existed)")

    frames = [r["_final_df"] for r in created if r["_final_df"] is not None and not r["_final_df"].empty]
    if not frames:
        print("  no newly created rows to validate (all outputs skipped or empty).")
        return

    required_out = [
        "date", "radar_name", "radar_lat_deg", "radar_lon_deg", "radar_alt_m",
        "icao24", "trajectory_id", "source_segment_id", "sample_idx", "timestamp", "dt_s",
        "lat_deg", "lon_deg", "alt_m", "east_m", "north_m", "up_m",
        "range_m", "azimuth_rad", "azimuth_deg", "elevation_rad", "elevation_deg",
        "ground_range_m", "ve_mps", "vn_mps", "vu_mps", "radial_velocity_mps",
        "speed_enu_mps", "speed_stage4_mps", "accel_stage4_mps2", "turn_rate_stage4_deg_s",
        "is_interpolated", "trajectory_start_time", "trajectory_end_time",
        "trajectory_duration_s", "n_samples",
    ]
    for df in frames:
        missing = [c for c in required_out if c not in df.columns]
        if missing:
            fail(f"output is missing required column(s): {missing}")
    print("  required output columns present: OK")

    for r in created:
        df = r["_final_df"]
        if df is None or df.empty:
            continue
        rng = df["range_m"].to_numpy()
        if not (np.isfinite(rng).all() and (rng >= 0).all()):
            fail(f"{r['date']}: range_m contains non-finite or negative values")
        az = df["azimuth_rad"].to_numpy()
        if not ((az >= 0).all() and (az < 2 * np.pi).all()):
            fail(f"{r['date']}: azimuth_rad outside [0, 2*pi)")
        el = df["elevation_rad"].to_numpy()
        if not ((el >= -np.pi / 2).all() and (el <= np.pi / 2).all()):
            fail(f"{r['date']}: elevation_rad outside [-pi/2, pi/2]")
        same_traj = df["trajectory_id"] == df["trajectory_id"].shift(1)
        if not (df["timestamp"].diff()[same_traj] > 0).all():
            fail(f"{r['date']}: timestamps not monotonic increasing within a trajectory")
    print("  range finite/nonnegative, azimuth in [0, 2pi), elevation in [-pi/2, pi/2],")
    print("  per-trajectory timestamps monotonic: OK")

    # Stage4-vs-ENU speed consistency: report-only unless wildly off.
    speed_enu = np.concatenate([f["speed_enu_mps"].to_numpy() for f in frames])
    speed_s4 = np.concatenate([pd.to_numeric(f["speed_stage4_mps"]).to_numpy() for f in frames])
    both = np.isfinite(speed_enu) & np.isfinite(speed_s4)
    median_diff = float(np.median(np.abs(speed_enu[both] - speed_s4[both]))) if both.any() else float("nan")
    print(f"  median |speed_enu - speed_stage4|: {median_diff:.3f} m/s (report-only; "
          f"hard limit {MAX_MEDIAN_SPEED_DIFF_MPS})")
    if np.isfinite(median_diff) and median_diff > MAX_MEDIAN_SPEED_DIFF_MPS:
        fail(f"median |speed_enu - speed_stage4| = {median_diff:.2f} m/s exceeds "
             f"{MAX_MEDIAN_SPEED_DIFF_MPS} m/s -- ENU velocity computation is likely broken")

    channels = {
        "range_m": np.concatenate([f["range_m"].to_numpy() for f in frames]),
        "ground_range_m": np.concatenate([f["ground_range_m"].to_numpy() for f in frames]),
        "|radial_velocity| (m/s)": np.abs(np.concatenate([f["radial_velocity_mps"].to_numpy() for f in frames])),
        "speed_enu (m/s)": speed_enu,
        "elevation (deg)": np.concatenate([f["elevation_deg"].to_numpy() for f in frames]),
    }
    print("\n  combined statistics (report-only):")
    print(f"  {'channel':>24} | {'p50':>12} | {'p95':>12} | {'p99':>12}")
    for name, values in channels.items():
        p50, p95, p99 = np.nanpercentile(values, [50, 95, 99])
        print(f"  {name:>24} | {p50:>12.3f} | {p95:>12.3f} | {p99:>12.3f}")


# =============================================================================
# Self-test (no real data required)
# =============================================================================

def self_test() -> None:
    """End-to-end check on a tiny synthetic stage-4-like file: one 5-point
    trajectory flying east at ~50 m/s near a radar at (45N, 7E, 200 m)."""
    radar_lat, radar_lon, radar_alt = 45.0, 7.0, 200.0
    n = 5
    # ~50 m/s east: 500 m per 10 s step; metres-per-degree-longitude at 45N
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(radar_lat))
    lons = 7.01 + 500.0 * np.arange(n) / m_per_deg_lon
    t = 1_000_000.0 + 10.0 * np.arange(n)

    rows = pd.DataFrame({
        "icao24": "abc123", "callsign": "TEST1",
        "trajectory_id": "abc123_1000000_r0", "source_segment_id": "abc123_1000000",
        "sample_idx": np.arange(n), "timestamp": t, "dt_s": 10.0,
        "lat_interp": 45.0005, "lon_interp": lons, "alt_interp": 1000.0,
        "lat_smooth": 45.0005, "lon_smooth": lons, "alt_smooth": 1000.0,
        "is_interpolated": False,
        "speed_mps": 50.0, "accel_mps2": 0.0, "accel_vector_mps2": 0.0, "turn_rate_deg_s": 0.0,
        "trajectory_start_time": t[0], "trajectory_end_time": t[-1],
        "trajectory_duration_s": t[-1] - t[0], "n_samples": n,
    })

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        rows.to_csv(os.path.join(in_dir, f"{INPUT_PREFIX}2022-01-01{INPUT_SUFFIX}"), index=False)

        cfg = RadarTruthConfig(radar_lat=radar_lat, radar_lon=radar_lon, radar_alt_m=radar_alt)
        (date, path), = discover_input_files(in_dir)
        result = make_radar_truth_for_day(date, path, out_dir, cfg)
        out = result["_final_df"]

        assert os.path.exists(result["output_file"]), "output file was not written"
        assert (out["range_m"] >= 0).all() and np.isfinite(out["range_m"]).all(), "bad range_m"
        assert ((out["azimuth_rad"] >= 0) & (out["azimuth_rad"] < 2 * np.pi)).all(), "azimuth out of bounds"
        assert (out["timestamp"].diff().iloc[1:] > 0).all(), "timestamps not monotonic"
        assert np.isfinite(out["radial_velocity_mps"]).mean() > 0.9, "radial velocity mostly non-finite"
        # physics: eastbound at 50 m/s => speed_enu ~= 50 and, from a radar
        # roughly south-west of the track, range should be increasing overall
        assert np.allclose(out["speed_enu_mps"], 50.0, atol=1.0), \
            f"speed_enu {out['speed_enu_mps'].tolist()} != ~50"
        assert out["range_m"].iloc[-1] > out["range_m"].iloc[0], "range should grow flying away"

        run_validation_gate([result])

    print("\nStage 05 self-test passed.")
