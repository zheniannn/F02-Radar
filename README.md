# F02-RADAR

Radar-side stages of the light-GA motion-prior pipeline. **F02-RADAR starts
from F01-PREPROCESSING stage 4 outputs**: the uniformly-sampled (10 s grid)
trajectory CSVs. Stage 5 re-expresses those trajectories relative to a
fixed, user-supplied radar site.

**Stage 5 produces noiseless radar-coordinate truth only.** It does not
simulate anything. **Stage 6 will simulate radar measurements, noise,
clutter, detection thresholds, and missed detections** on top of this truth.

## Structure

```
F02-RADAR/
├── scripts/
│   └── 05_make_radar_truth.py   # stage 5: 10s trajectories -> radar-coordinate truth
├── utils/
│   ├── io.py                     # path resolution (repo-root relative)
│   ├── geo.py                    # WGS84 geodetic -> ECEF -> ENU (numpy, no pymap3d)
│   └── radar_truth.py            # stage 5 rules: discovery, conversion, filters, gate
├── data/
│   └── active/
│       ├── trajectories_10s/     # INPUT: stage-4 CSVs copied from F01 (git-ignored)
│       └── radar_truth/          # OUTPUT: per-day truth CSVs + summary (git-ignored)
└── reports/
```

## Input contract from F01-PREPROCESSING

Required input files (one per day), produced by F01's stage 4:

```
states_YYYY-MM-DD_conventionalGA_trajectories_10s.csv
```

Each file must contain at least these stage-4 columns:

| Group | Columns |
|---|---|
| identity / provenance | `icao24`, `trajectory_id`, `source_segment_id`, `sample_idx`, `is_interpolated` (+ `callsign` if present, carried through) |
| grid | `timestamp`, `dt_s` |
| positions | `lat_smooth`, `lon_smooth`, `alt_smooth` (default source) and/or `lat_interp`, `lon_interp`, `alt_interp` (`--position-source interp`) |
| stage-4 kinematics | `speed_mps`, `accel_mps2`, `turn_rate_deg_s` |
| trajectory metadata | `trajectory_start_time`, `trajectory_end_time`, `trajectory_duration_s`, `n_samples` |

Stage 5 fails with a clear error if any required column (or the requested
position-source triple) is missing.

## How to copy inputs from F01

```bash
cp ../F01-PREPROCESSING/data/active/trajectories_10s/states_*_trajectories_10s.csv data/active/trajectories_10s/
```

(Input CSVs are git-ignored; only the directory structure is committed.)

## Usage

The radar location is **never invented** — `--radar-lat` and `--radar-lon`
are required:

```bash
python scripts/05_make_radar_truth.py --radar-lat 45.05 --radar-lon 7.35
python scripts/05_make_radar_truth.py --radar-lat 45.05 --radar-lon 7.35 \
    --radar-alt-m 250 --radar-name torino --max-range-m 120000 --drop-below-horizon --overwrite
python scripts/05_make_radar_truth.py --self-test   # tiny synthetic check, no real data needed
```

Can be run from any working directory — paths are resolved relative to the
repo root, not the shell's cwd.

CLI flags: `--radar-lat` / `--radar-lon` (required), `--radar-alt-m`
(default 0), `--radar-name` (default `radar`), `--position-source`
`smooth|interp` (default `smooth`), `--min-range-m` (default 0),
`--max-range-m` (default: no limit), `--drop-below-horizon` (off by
default), `--overwrite` (existing outputs are skipped with a clear message
unless passed), `--input-dir` (default: `data/active/trajectories_10s`),
`--output-dir` (default: `data/active/radar_truth`).

- **Output:** `data/active/radar_truth/radar_truth_YYYY-MM-DD.csv`, one per
  day, plus `radar_truth_summary.csv` (one row per day, reporting rows
  before and after filtering). Input files are never modified.

## Method

For each day (`utils/radar_truth.py::make_radar_truth_for_day`):

1. **Skip-or-overwrite** — if the day's output already exists and
   `--overwrite` was not passed, it is skipped with a clear message.

2. **Positions** — the chosen stage-4 position triple is written out as
   `lat_deg`/`lon_deg`/`alt_m` and converted to local **East/North/Up
   metres relative to the radar origin using exact WGS84
   geodetic→ECEF→ENU** transforms (`utils/geo.py`, vectorized numpy — no
   flat-earth approximation, no pymap3d dependency).

3. **Radar geometry** — per sample:
   `range_m = √(E²+N²+U²)`, `ground_range_m = √(E²+N²)`,
   `azimuth_rad = atan2(E, N)` wrapped to [0, 2π) (0 = north, π/2 = east),
   `elevation_rad = atan2(U, ground_range)`; degree versions of both.

4. **Velocities** — `ve/vn/vu_mps` are per-trajectory finite differences of
   the ENU positions via `np.gradient` over each trajectory's timestamps
   (centered in the interior, one-sided at the ends; NaN for <2-sample
   trajectories, which stage 4 makes near-impossible).
   `radial_velocity_mps` is the dot product of the ENU velocity with the
   line-of-sight unit vector — **positive means range increasing** — and
   `speed_enu_mps = √(ve²+vn²+vu²)`. Stage-4 kinematics are carried through
   unchanged as `speed_stage4_mps`, `accel_stage4_mps2`,
   `turn_rate_stage4_deg_s` for cross-validation.

5. **Row-level filtering** (never trajectory-level): `range_m <
   --min-range-m`, `range_m > --max-range-m` (if set), and, with
   `--drop-below-horizon`, `elevation_rad < 0`. Velocities are computed on
   the full trajectory *before* filtering, so dropping a row never corrupts
   its neighbors' velocities.

6. **Save** — identity/provenance columns are preserved and radar-site
   columns (`radar_name`, `radar_lat_deg`, `radar_lon_deg`, `radar_alt_m`)
   are stamped on every row.

After all days, a **validation gate** runs and raises a clear error on
failure: at least one output created or skipped; required output columns
present; `range_m` finite and ≥ 0; azimuth in [0, 2π); elevation in
[−π/2, π/2]; per-trajectory timestamps monotonic increasing. The median
|`speed_enu_mps` − `speed_stage4_mps`| is printed report-only (hard failure
only above 20 m/s), followed by a report-only p50/p95/p99 table for range,
ground range, |radial velocity|, ENU speed, and elevation.

## Self-test

```bash
python scripts/05_make_radar_truth.py --self-test
```

Builds a tiny synthetic stage-4-like file (one 5-point trajectory flying
east at ~50 m/s near a radar at 45°N 7°E), runs the full stage into a
temporary directory, and asserts the geometry (nonnegative finite range,
bounded azimuth, monotonic timestamps, finite radial velocity, ENU speed
≈ 50 m/s, growing range when flying away). No real data is touched.

## Extending

Stage-5 rules live entirely in `utils/radar_truth.py`; the WGS84 math is
isolated in `utils/geo.py`. `io.py` and the entry-point script only need to
change if the underlying file/folder layout changes.
