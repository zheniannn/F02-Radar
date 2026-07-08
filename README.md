# F02-Radar

> **Evaluation moved to F03-Evaluation.** F02-Radar ends at stage 6
> synthetic radar detections. Use **F03-Evaluation** for stage 7 threshold
> evaluation and stage 8 tracking baselines (the stage-7 code and reports
> in this repo are retained for history but maintained in F03).

Radar-side stages of the light-GA motion-prior pipeline. **F02-Radar starts
from F01-Preprocessing stage 4 outputs**: the uniformly-sampled (10 s grid)
trajectory CSVs.

Pipeline:

```
stage 4 trajectories (from F01)
        │
    05_make_radar_truth.py          radar-coordinate truth (noiseless)
        │
    06_simulate_radar_detections.py synthetic point detections under
        ▼                           SNR / noise / clutter / thresholds
data/active/sim_detections/detections_<date>_thr_<T>dB.csv
```

**Stage 5 produces noiseless radar-coordinate truth only.** It does not
simulate anything. **Stage 6 simulates radar measurements at the
point-detection level**: SNR-dependent missed detections, Gaussian
measurement noise, and Poisson clutter under configurable thresholds — it
is a synthetic point-detection simulation, **not** a raw radar simulation
(no RF/IQ, pulse compression, or range-Doppler imaging; no tracking, model
training, or train/test splitting — those belong to later stages).

## Structure

```
F02-Radar/
├── scripts/
│   ├── 05_make_radar_truth.py            # stage 5: 10s trajectories -> radar-coordinate truth
│   └── 06_simulate_radar_detections.py   # stage 6: truth -> thresholded point detections
├── utils/
│   ├── io.py                     # path resolution (repo-root relative)
│   ├── geo.py                    # WGS84 geodetic -> ECEF -> ENU (numpy, no pymap3d)
│   ├── radar_truth.py            # stage 5 rules: discovery, conversion, filters, gate
│   └── radar_sim.py              # stage 6 rules: SNR/Pd model, noise, clutter, gate
├── data/
│   └── active/
│       ├── trajectories_10s/     # INPUT: stage-4 CSVs copied from F01 (git-ignored)
│       ├── radar_truth/          # stage-5 output: per-day truth CSVs (git-ignored)
│       └── sim_detections/       # stage-6 output: detection CSVs (git-ignored)
└── reports/
```

## Input contract from F01-Preprocessing

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
cp ../F01-Preprocessing/data/active/trajectories_10s/states_*_trajectories_10s.csv data/active/trajectories_10s/
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
`--output-dir` (default: `data/active/radar_truth`), plus the optional
`--relocate-*` flags (off by default; see "Synthetic relocation near
radar" below).

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
ground range, |radial velocity|, ENU speed, and elevation. When relocation
is enabled, the gate additionally verifies `relocated == 1`, finite
original geography and anchors, and that every trajectory's first-point
ground range lies within the configured anchor bounds (checked via the
per-trajectory anchor columns, so row filtering can't hide a violation),
plus a report-only first-point ground-range percentile line.

## Synthetic relocation near radar

By default, stage 5 uses **true ADS-B geography**. With
`--relocate-near-radar`, each trajectory's **motion/shape is preserved but
its absolute location is moved near the radar**: a deterministic anchor is
drawn per trajectory (ground range ~ U[`--relocate-min-ground-range-m`,
`--relocate-max-ground-range-m`], bearing ~ U[`--relocate-min-bearing-deg`,
`--relocate-max-bearing-deg`], seeded via sha256 of
(`--relocate-seed`, date, trajectory_id, radar name) — stable across runs
and processes), and the trajectory's ENU offsets relative to its own first
point are re-planted on that anchor. This is intended for **controlled
radar simulation and target injection**; the resulting `lat_deg` /
`lon_deg` / `alt_m` are *synthetic* coordinates derived from the relocated
ENU and **must not be interpreted as real aircraft geography**. The true
positions are always preserved in `original_lat_deg` / `original_lon_deg` /
`original_alt_m`, alongside `relocated` (0/1) and per-trajectory
`relocation_anchor_*` / `relocation_delta_*` provenance columns (NaN when
relocation is off).

Altitude handling (`--relocate-anchor-altitude-mode`): `preserve` (default)
keeps each trajectory's original MSL altitude profile relative to the radar
altitude (`up = original_alt − radar_alt`) — realistic GA
cruise/climb/descent without needing terrain; `fixed_up` puts the first
point at `--relocate-fixed-up-m` above the radar and preserves the original
vertical displacement profile from there.

Velocities, range/azimuth/elevation, and radial velocity are all computed
from the **relocated** ENU coordinates, so downstream consumers see a fully
consistent radar frame. **Caution:** `--min-range-m`, `--max-range-m`, and
`--drop-below-horizon` are applied **after** relocation. Stage 6 should use
relocated stage-5 outputs if the experiment assumes all aircraft are within
radar coverage.

```bash
python scripts/05_make_radar_truth.py \
  --radar-lat <LAT> \
  --radar-lon <LON> \
  --radar-alt-m <ALT> \
  --relocate-near-radar \
  --relocate-min-ground-range-m 10000 \
  --relocate-max-ground-range-m 80000 \
  --overwrite
```

## Self-test

```bash
python scripts/05_make_radar_truth.py --self-test
```

Builds a tiny synthetic stage-4-like file (one 5-point trajectory flying
east at ~50 m/s near a radar at 45°N 7°E), runs the full stage into a
temporary directory, and asserts the geometry (nonnegative finite range,
bounded azimuth, monotonic timestamps, finite radial velocity, ENU speed
≈ 50 m/s, growing range when flying away). A second branch then moves the
same trajectory ~7,000 km away, confirms it really is that far without
relocation, and re-runs with `--relocate-near-radar` bounds of 20–30 km —
asserting the relocated first point lands inside those bounds, the motion
is still ~50 m/s, the original geography is preserved in `original_*`, and
the synthetic coordinates sit near the radar. No real data is touched.

---

# Stage 6 — `06_simulate_radar_detections.py`

Simulates **thresholded radar point detections** from stage 5's noiseless
truth. Each output row is one detection after thresholding — either a noisy
measurement of a real aircraft (`is_target = 1`, with truth metadata and
per-component measurement errors) or a clutter false alarm
(`is_target = 0`, with truth/identity fields NaN/blank). This is a
**synthetic point-detection simulation, not a raw radar simulation** — no
RF/IQ, pulse compression, or range-Doppler images.

## Input contract from Stage 5

One file per day, `radar_truth_YYYY-MM-DD.csv`, containing the stage-5
output columns (identity/provenance, `timestamp`, `range_m`,
`azimuth_rad`/`_deg`, `elevation_rad`/`_deg`, `ground_range_m`,
`radial_velocity_mps`, ENU positions/velocities, and trajectory metadata).
Stage 6 fails clearly if any required column is missing.

## Outputs

One file per day **per threshold**:

```
data/active/sim_detections/detections_YYYY-MM-DD_thr_<THRESHOLD>dB.csv
    e.g. detections_2022-06-06_thr_6p0dB.csv   (-5 dB -> thr_m5p0dB)
```

plus `sim_detection_summary.csv` with one row per (day, threshold):
truth rows, frames, target detections, missed targets, empirical Pd,
clutter detections, total detections, false alarms per frame, and
p10/median/p90 target SNR. Existing outputs are skipped with a clear
message unless `--overwrite` is passed.

A **frame** is one radar scan: `floor(timestamp / --frame-period-s)`
(default 10 s, matching the stage-4 grid), with `frame_id` the stable index
of the sorted unique frames and the bin start as the frame's representative
timestamp. Binning is essential — stage-4 timestamps are anchored at each
trajectory's own first fix, so raw timestamps are *not* aligned across
aircraft; grouping them directly would put almost every detection in its
own "frame". Multiple aircraft share a frame — no one-target-per-frame
assumption anywhere.

## Method

Per truth row and threshold (`utils/radar_sim.py`):

1. **SNR draw** — `range_decay` (default):
   `snr = ref − 10·power·log10(range/ref_range) + N(0, std)` (the radar
   equation in dB: with `power = 4`, SNR falls 12 dB per range doubling),
   clamped to `[min, max]`, range clipped to ≥ 1 m before the log.
   `constant`: `snr = ref + N(0, std)`.
2. **Detection probability** — logistic in SNR vs threshold:
   `pd = pd_min + (pd_max − pd_min) / (1 + exp(−(snr − thr)/width))`,
   then a Bernoulli draw. Missed targets write no row.
3. **Measurement noise** — Gaussian per component
   (`--sigma-range-m 75`, `--sigma-azimuth-deg 0.15`,
   `--sigma-elevation-deg 0.15`, `--sigma-radial-velocity-mps 2`); measured
   range floored at 0, azimuth wrapped to [0, 2π), elevation clipped to
   [−π/2, π/2]; per-component error columns recorded (azimuth error is
   wrap-aware).
4. **Clutter** — per frame, `n ~ Poisson(λ)` with
   `λ = rate_ref · exp(−(thr − ref_thr)/scale)` (lower thresholds → more
   false alarms); uniform in range/azimuth/elevation/radial velocity within
   the configured bounds; SNR = threshold + Exponential(scale), since only
   threshold-passing clutter is ever written. If `--max-range-m` is
   omitted it is inferred per day from the truth's max range rounded up to
   the nearest 10 km (printed clearly).
5. **Reproducibility** — every (day, threshold) pair gets a child
   `numpy.random.default_rng` seeded via sha256 of
   (seed, date, threshold, scenario-id) — stable across processes and
   run order (never Python's `hash()`).

After all days, a **validation gate** checks: outputs created/skipped;
required columns; `is_target` ∈ {0, 1}; measured range finite ≥ 0; azimuth
in [0, 2π); elevation in [−π/2, π/2]; SNR finite; target rows have
trajectory ids, finite truth range, and finite error columns; clutter rows
have blank ids and NaN truth. Threshold trends (empirical Pd and false
alarms per frame should generally decrease with threshold) are printed
**report-only** — finite random samples may wiggle.

## Using relocated Stage 5 truth

Stage 6 can consume either **true-geography** or **relocated** stage-5 radar
truth. For the controlled low-SNR radar experiment, relocated truth is
recommended — every trajectory is then genuinely inside radar coverage.
Point stage 6 at the relocated directory with
`--input-dir data/active/radar_truth_relocated`, and pass an **explicit
`--max-range-m`** (e.g. 100000) so the clutter support is consistent across
days and thresholds instead of being inferred per day.

Stage 6 reports the input's relocated state per day (`relocated truth
fraction: 1.000`, or `unavailable (column missing)` for pre-relocation
truth files) and prints a warning — never a failure — when the input does
not look fully relocated. The summary CSV records
`relocated_column_present`, `relocated_truth_fraction`, and
`max_range_m_used` per (day, threshold).

Target detections **preserve key relocation metadata** (`relocated`,
`original_lat_deg`/`original_lon_deg`/`original_alt_m`,
`relocation_anchor_east/north/up_m`) copied from the stage-5 truth row, so
later tracking/evaluation can trace any detection back to the original
ADS-B trajectory. Inputs without relocation columns get `relocated = 0` and
NaN metadata. Clutter rows always carry `relocated = 0` (a false alarm is
never a relocated aircraft) with all original/anchor columns NaN.
`relocation_delta_*` columns are deliberately not carried through.

Recommended real-data command:

```bash
python scripts/06_simulate_radar_detections.py \
  --input-dir data/active/radar_truth_relocated \
  --threshold-db -5 0 3 6 9 12 \
  --snr-model range_decay \
  --target-snr-ref-db 8 \
  --snr-ref-range-m 50000 \
  --target-snr-std-db 3 \
  --sigma-range-m 75 \
  --sigma-azimuth-deg 0.15 \
  --sigma-elevation-deg 0.15 \
  --sigma-radial-velocity-mps 2 \
  --clutter-rate-ref 20 \
  --max-range-m 100000 \
  --overwrite
```

## Usage

```bash
python scripts/06_simulate_radar_detections.py                      # defaults, thresholds -5..12
python scripts/06_simulate_radar_detections.py --threshold-db 0 6 --scenario-id lowclutter \
    --clutter-rate-ref 5 --overwrite
python scripts/06_simulate_radar_detections.py --self-test          # tiny synthetic check
```

## Self-test

`--self-test` builds a tiny synthetic truth file (two trajectories × five
timestamps), simulates thresholds [−5, 6, 12] with a fixed seed and
moderate clutter into a temporary directory, and asserts output existence,
sane Pd/clutter ordering across thresholds, column completeness, valid
labels, and bounded measurements. No real data is touched.

---

# Relocated wide-area weak-target experiment

The current main experiment is a **wide-area weak-target experiment — not a
range-contained radar-coverage experiment**. Understanding that distinction
is essential to interpreting the outputs:

- Aircraft trajectories are synthetically **anchored near the radar at
  their first sample** (ground range uniform in 10–80 km, bearing uniform).
- After the first sample, each aircraft follows its **original
  ADS-B-derived motion** unchanged — so long trajectories drift well beyond
  the anchor band (p95 sample range ≈ 194 km, p99 ≈ 359 km in the current
  dataset).
- Stage 6's `--max-range-m 100000` controls the **spatial support of
  clutter generation only**. It does **not** remove target truth rows
  outside 100 km.
- Farther targets remain in the dataset and naturally receive **lower SNR
  through the range-decay model**, becoming weak targets rather than being
  cut. This is intentional.
- This setup is suitable for studying **threshold trade-offs and
  weak-target tracking over long trajectories**. A future range-contained
  experiment would require stage-5 post-relocation target filtering or a
  stage-6 target-range gate — neither exists today, by design.

Commands used for the current dataset:

```bash
# Stage 05 relocated truth
python scripts/05_make_radar_truth.py \
  --radar-lat 38.966 \
  --radar-lon -86.999 \
  --radar-alt-m 200 \
  --radar-name centroid \
  --relocate-near-radar \
  --relocate-min-ground-range-m 10000 \
  --relocate-max-ground-range-m 80000 \
  --output-dir data/active/radar_truth_relocated \
  --overwrite

# Stage 06 relocated detections
python scripts/06_simulate_radar_detections.py \
  --input-dir data/active/radar_truth_relocated \
  --output-dir data/active/sim_detections_relocated \
  --scenario-id relocated \
  --threshold-db -5 0 3 6 9 12 \
  --snr-model range_decay \
  --target-snr-ref-db 8 \
  --snr-ref-range-m 50000 \
  --target-snr-std-db 3 \
  --sigma-range-m 75 \
  --sigma-azimuth-deg 0.15 \
  --sigma-elevation-deg 0.15 \
  --sigma-radial-velocity-mps 2 \
  --clutter-rate-ref 20 \
  --max-range-m 100000 \
  --overwrite
```

A read-only audit of this experiment (truth statistics, threshold sweep,
coverage-range fractions, stage-7 notes) can be regenerated any time with:

```bash
python scripts/06_audit_relocated_experiment.py
```

which writes `reports/relocated_experiment_audit.md`
(`--self-test` available; see the script docstring for flags).

---

# Stage 7 — Threshold-only baseline evaluation

Evaluates the detection trade-off produced by the stage-6 threshold sweep:
low thresholds recover more targets but admit more clutter; high thresholds
suppress clutter but miss weak targets. **This is not tracking** — no data
association, Kalman filtering, or trajectory smoothing happens here;
detections are scored frame-by-frame against stage-5 truth. **Stage 8 will
add tracking** (a low-threshold detect-then-track baseline with a
constant-velocity Kalman filter and gating, before any ML model).

- **Inputs:** stage-5 truth (`radar_truth_YYYY-MM-DD.csv`), stage-6
  detections (`detections_YYYY-MM-DD_thr_*dB.csv`), and the stage-6
  summary when present (authoritative frame counts).
- **Outputs** (`reports/stage07_threshold_only/`): per-day and overall
  operating-curve tables, Pd by truth range bin, clutter by measured range
  bin, SNR summaries of written detections (not Pd-by-SNR — stage 6 writes
  no missed-target rows), measurement-error sanity checks,
  `threshold_only_report.md`, and four matplotlib plots. Counts/Pd/false
  alarms are exact; only SNR/error quantiles use a capped deterministic
  sample (documented in the report).

```bash
python scripts/07_evaluate_threshold_only.py \
  --truth-dir data/active/radar_truth_relocated \
  --detections-dir data/active/sim_detections_relocated \
  --output-dir reports/stage07_threshold_only \
  --coverage-range-m 100000 \
  --range-bins-m 0,50000,100000,200000,inf \
  --chunksize 1000000 \
  --overwrite
```

`--self-test` runs a tiny synthetic end-to-end check; `--no-plots` skips
PNG generation. All reads are chunked — the 17 GB detection set is never
loaded into memory.

## Extending

Stage-5 rules live entirely in `utils/radar_truth.py`, stage-6 rules in
`utils/radar_sim.py`, stage-7 evaluation in `utils/threshold_eval.py`; the
WGS84 math is isolated in `utils/geo.py`. `io.py` and the entry-point
scripts only need to change if the underlying file/folder layout changes.
