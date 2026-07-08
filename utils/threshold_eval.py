"""Stage 7: threshold-only baseline evaluation of stage-6 detections.

Quantifies the detection trade-off produced by the stage-6 threshold sweep
-- low thresholds recover more targets but admit more clutter; high
thresholds suppress clutter but miss weak targets -- as tables, plots, and
a Markdown report. No tracking, data association, or smoothing happens
here; that is stage 8.

Memory: every truth/detection file is read in chunks and reduced to counts
immediately -- nothing close to the full 17 GB is ever resident. Counts,
Pd, and false-alarm rates are EXACT. Only the SNR / measurement-error
quantiles are computed from a capped deterministic sample per group
(QUANTILE_SAMPLE_CAP rows, seed SAMPLE_SEED), which the report discloses.
"""

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRUTH_PATTERN = re.compile(r"radar_truth_(\d{4}-\d{2}-\d{2})\.csv$")
DETECTION_PATTERN = re.compile(r"detections_(\d{4}-\d{2}-\d{2})_thr_(.+)dB\.csv$")
SUMMARY_NAME = "sim_detection_summary.csv"

QUANTILE_SAMPLE_CAP = 200_000
SAMPLE_SEED = 123

ERROR_COLUMNS = ["range_error_m", "azimuth_error_rad",
                 "elevation_error_rad", "radial_velocity_error_mps"]

DETECTION_USECOLS = ["is_target", "truth_range_m", "meas_range_m", "snr_db"] + ERROR_COLUMNS


@dataclass
class ThresholdEvalConfig:
    """All stage-7 tunables in one place (populated from the CLI)."""
    truth_dir: str
    detections_dir: str
    output_dir: str
    coverage_range_m: float = 100_000.0
    range_bins_m: List[float] = field(default_factory=lambda: [0.0, 50_000.0, 100_000.0, 200_000.0, float("inf")])
    chunksize: int = 1_000_000
    overwrite: bool = False
    no_plots: bool = False


# =============================================================================
# Parsing / discovery
# =============================================================================

def parse_range_bins(range_bins_m: str) -> List[float]:
    """'0,50000,100000,200000,inf' -> [0.0, 50000.0, 100000.0, 200000.0, inf]."""
    edges = [float(p) for p in range_bins_m.split(",")]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"range bins must be strictly increasing with >= 2 edges: {edges}")
    return edges


def format_range_bin_labels(edges: List[float]) -> List[str]:
    """[0, 50k, 100k, inf] -> ['0-50 km', '50-100 km', '>100 km']."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append(f">{lo / 1000:.0f} km")
        else:
            labels.append(f"{lo / 1000:.0f}-{hi / 1000:.0f} km")
    return labels


def discover_truth_files(truth_dir: str) -> List[Tuple[str, str]]:
    """Sorted (date, path) pairs for radar_truth_YYYY-MM-DD.csv files."""
    out = []
    for name in sorted(os.listdir(truth_dir)):
        m = TRUTH_PATTERN.search(name)
        if m:
            out.append((m.group(1), os.path.join(truth_dir, name)))
    return out


def parse_detection_filename(path: str) -> Optional[Tuple[str, float]]:
    """detections_2022-06-06_thr_m5p0dB.csv -> ('2022-06-06', -5.0)."""
    m = DETECTION_PATTERN.search(os.path.basename(path))
    if not m:
        return None
    return m.group(1), float(m.group(2).replace("m", "-").replace("p", "."))


def discover_detection_files(detections_dir: str) -> List[Tuple[str, float, str]]:
    """Sorted (date, threshold_db, path) triples for stage-6 detection files."""
    out = []
    for name in sorted(os.listdir(detections_dir)):
        parsed = parse_detection_filename(name)
        if parsed:
            out.append((parsed[0], parsed[1], os.path.join(detections_dir, name)))
    return sorted(out)


def load_detection_summary_if_available(detections_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(detections_dir, SUMMARY_NAME)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "created"]
    return df if not df.empty else None


# =============================================================================
# Capped deterministic sampling for quantiles (counts stay exact elsewhere)
# =============================================================================

class SampleBuffer:
    """Collects up to `cap` values; overflow chunks are deterministically
    subsampled (fixed seed + call counter), then further input is dropped.
    `truncated` records whether the quantiles come from a capped sample."""

    def __init__(self, cap: int = QUANTILE_SAMPLE_CAP):
        self.cap = cap
        self.parts: List[np.ndarray] = []
        self.size = 0
        self.calls = 0
        self.truncated = False

    def add(self, values: np.ndarray) -> None:
        self.calls += 1
        values = values[np.isfinite(values)]
        if not values.size:
            return
        remaining = self.cap - self.size
        if remaining <= 0:
            self.truncated = True
            return
        if len(values) > remaining:
            rng = np.random.default_rng(SAMPLE_SEED + self.calls)
            values = rng.choice(values, size=remaining, replace=False)
            self.truncated = True
        self.parts.append(np.asarray(values, dtype=float))
        self.size += len(values)

    def values(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.array([])


# =============================================================================
# Pass 1: truth denominators (chunked)
# =============================================================================

def compute_truth_denominators(truth_files: List[Tuple[str, str]], cfg: ThresholdEvalConfig,
                               need_frames_fallback: bool) -> Dict[str, Dict]:
    """Per day: exact truth-row count, per-range-bin truth counts, and (only
    when no stage-6 summary exists) the unique-timestamp frame fallback."""
    edges = np.array(cfg.range_bins_m)
    per_day: Dict[str, Dict] = {}

    for date, path in truth_files:
        usecols = ["range_m"] + (["timestamp"] if need_frames_fallback else [])
        rows = 0
        bin_counts = np.zeros(len(edges) - 1, dtype=np.int64)
        ts_parts: List[np.ndarray] = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=cfg.chunksize):
            rows += len(chunk)
            rng_m = chunk["range_m"].to_numpy()
            idx = np.digitize(rng_m, edges[1:-1])          # right-open bins
            bin_counts += np.bincount(idx, minlength=len(edges) - 1)
            if need_frames_fallback:
                ts_parts.append(chunk["timestamp"].unique())
        frames = int(np.unique(np.concatenate(ts_parts)).size) if ts_parts else None
        per_day[date] = {"truth_rows": rows, "bin_counts": bin_counts, "frames_fallback": frames}
    return per_day


# =============================================================================
# Pass 2: detection metrics (chunked; the large pass)
# =============================================================================

def compute_detection_metrics(detection_files: List[Tuple[str, float, str]],
                              cfg: ThresholdEvalConfig) -> Dict:
    """Exact counts per (day, threshold) and per range bin, plus capped
    samples for SNR / error quantiles and exact accumulators for the
    range-error mean/std."""
    edges = np.array(cfg.range_bins_m)
    n_bins = len(edges) - 1

    records = []
    snr_buffers: Dict[Tuple[float, int], SampleBuffer] = {}
    err_buffers: Dict[Tuple[float, str], SampleBuffer] = {}
    err_acc: Dict[float, Dict[str, float]] = {}
    relocated_seen = False

    for date, threshold_db, path in detection_files:
        header = list(pd.read_csv(path, nrows=0).columns)
        relocated_seen = relocated_seen or ("relocated" in header)

        targets = clutter = 0
        tgt_bins = np.zeros(n_bins, dtype=np.int64)
        clu_bins = np.zeros(n_bins, dtype=np.int64)

        for chunk in pd.read_csv(path, usecols=DETECTION_USECOLS, chunksize=cfg.chunksize):
            is_tgt = (chunk["is_target"] == 1).to_numpy()
            tgt = chunk[is_tgt]
            clu = chunk[~is_tgt]
            targets += len(tgt)
            clutter += len(clu)

            if len(tgt):
                idx = np.digitize(tgt["truth_range_m"].to_numpy(), edges[1:-1])
                tgt_bins += np.bincount(idx, minlength=n_bins)
                snr_buffers.setdefault((threshold_db, 1), SampleBuffer()).add(tgt["snr_db"].to_numpy())
                acc = err_acc.setdefault(threshold_db, {"n": 0, "sum": 0.0, "sumsq": 0.0})
                re_vals = tgt["range_error_m"].to_numpy(dtype=float)
                re_vals = re_vals[np.isfinite(re_vals)]
                acc["n"] += len(re_vals)
                acc["sum"] += float(re_vals.sum())
                acc["sumsq"] += float((re_vals**2).sum())
                for col in ERROR_COLUMNS:
                    err_buffers.setdefault((threshold_db, col), SampleBuffer()).add(
                        tgt[col].to_numpy(dtype=float))
            if len(clu):
                idx = np.digitize(clu["meas_range_m"].to_numpy(), edges[1:-1])
                clu_bins += np.bincount(idx, minlength=n_bins)
                snr_buffers.setdefault((threshold_db, 0), SampleBuffer()).add(clu["snr_db"].to_numpy())

        records.append({"date": date, "threshold_db": threshold_db,
                        "target_detections": targets, "clutter_detections": clutter,
                        "target_bins": tgt_bins, "clutter_bins": clu_bins})

    return {"records": records, "snr_buffers": snr_buffers,
            "err_buffers": err_buffers, "err_acc": err_acc,
            "relocated_seen": relocated_seen}


# =============================================================================
# Tables
# =============================================================================

def build_overall_tables(truth_den: Dict[str, Dict], det: Dict,
                         summary: Optional[pd.DataFrame], cfg: ThresholdEvalConfig) -> Dict[str, pd.DataFrame]:
    edges = cfg.range_bins_m
    labels = format_range_bin_labels(edges)

    # Authoritative per-day frame counts: stage-6 summary when available,
    # else unique truth timestamps.
    frames_by_day: Dict[str, int] = {}
    if summary is not None and "frames" in summary.columns:
        for date, grp in summary.groupby("date"):
            frames_by_day[str(date)] = int(grp["frames"].iloc[0])
    for date, d in truth_den.items():
        frames_by_day.setdefault(date, d["frames_fallback"] or 0)

    by_day_rows, bin_rows, clu_bin_rows = [], [], []
    for rec in det["records"]:
        date, thr = rec["date"], rec["threshold_db"]
        truth_rows = truth_den[date]["truth_rows"]
        frames = frames_by_day[date]
        targets, clutter = rec["target_detections"], rec["clutter_detections"]
        total = targets + clutter
        by_day_rows.append({
            "date": date, "threshold_db": thr,
            "truth_rows": truth_rows, "frames": frames,
            "target_detections": targets, "clutter_detections": clutter,
            "missed_targets": truth_rows - targets,
            "empirical_pd": targets / truth_rows if truth_rows else np.nan,
            "false_alarm_per_frame": clutter / frames if frames else np.nan,
            "total_detections": total,
            "target_fraction": targets / total if total else np.nan,
            "clutter_fraction": clutter / total if total else np.nan,
        })
        for b, label in enumerate(labels):
            truth_bin = int(truth_den[date]["bin_counts"][b])
            tgt_bin = int(rec["target_bins"][b])
            bin_rows.append({
                "date": date, "threshold_db": thr, "range_bin": label,
                "bin_lo_m": edges[b], "bin_hi_m": edges[b + 1],
                "truth_rows_bin": truth_bin, "target_detections_bin": tgt_bin,
                "missed_targets_bin": truth_bin - tgt_bin,
                "empirical_pd_bin": tgt_bin / truth_bin if truth_bin else np.nan,
            })
            clu_bin_rows.append({
                "date": date, "threshold_db": thr, "range_bin": label,
                "bin_lo_m": edges[b], "bin_hi_m": edges[b + 1],
                "clutter_detections_bin": int(rec["clutter_bins"][b]),
                "false_alarm_per_frame_bin": int(rec["clutter_bins"][b]) / frames if frames else np.nan,
            })

    by_day = pd.DataFrame(by_day_rows).sort_values(["date", "threshold_db"]).reset_index(drop=True)
    by_bin = pd.DataFrame(bin_rows).sort_values(["date", "threshold_db", "bin_lo_m"]).reset_index(drop=True)
    clu_by_bin = pd.DataFrame(clu_bin_rows).sort_values(["date", "threshold_db", "bin_lo_m"]).reset_index(drop=True)

    overall = (
        by_day.groupby("threshold_db")
        .agg(truth_rows=("truth_rows", "sum"), frames=("frames", "sum"),
             target_detections=("target_detections", "sum"),
             clutter_detections=("clutter_detections", "sum"))
        .reset_index()
    )
    overall["missed_targets"] = overall["truth_rows"] - overall["target_detections"]
    overall["empirical_pd"] = overall["target_detections"] / overall["truth_rows"]
    overall["false_alarm_per_frame"] = overall["clutter_detections"] / overall["frames"]
    overall["total_detections"] = overall["target_detections"] + overall["clutter_detections"]
    overall["target_fraction"] = overall["target_detections"] / overall["total_detections"]
    overall["clutter_fraction"] = overall["clutter_detections"] / overall["total_detections"]

    overall_by_bin = (
        by_bin.groupby(["threshold_db", "range_bin", "bin_lo_m"])
        .agg(truth_rows_bin=("truth_rows_bin", "sum"),
             target_detections_bin=("target_detections_bin", "sum"))
        .reset_index()
        .sort_values(["threshold_db", "bin_lo_m"])
    )
    overall_by_bin["empirical_pd_bin"] = np.where(
        overall_by_bin["truth_rows_bin"] > 0,
        overall_by_bin["target_detections_bin"] / overall_by_bin["truth_rows_bin"], np.nan)

    # SNR summary (exact counts; quantiles from the capped samples)
    snr_rows = []
    for (thr, is_target), buf in sorted(det["snr_buffers"].items()):
        vals = buf.values()
        snr_rows.append({
            "threshold_db": thr, "is_target": is_target,
            "count": int(sum((r["target_detections"] if is_target else r["clutter_detections"])
                             for r in det["records"] if r["threshold_db"] == thr)),
            "snr_p10_db": float(np.percentile(vals, 10)) if vals.size else np.nan,
            "snr_p50_db": float(np.percentile(vals, 50)) if vals.size else np.nan,
            "snr_p90_db": float(np.percentile(vals, 90)) if vals.size else np.nan,
            "snr_mean_db": float(vals.mean()) if vals.size else np.nan,
            "sampled_for_quantiles": buf.truncated,
        })
    snr_summary = pd.DataFrame(snr_rows)

    # Measurement-error summary (exact count/mean/std; percentiles from samples)
    err_rows = []
    for thr in sorted({r["threshold_db"] for r in det["records"]}):
        acc = det["err_acc"].get(thr, {"n": 0, "sum": 0.0, "sumsq": 0.0})
        n = acc["n"]
        mean = acc["sum"] / n if n else np.nan
        var = acc["sumsq"] / n - mean**2 if n else np.nan
        bufs = {col: det["err_buffers"].get((thr, col), SampleBuffer()) for col in ERROR_COLUMNS}
        vals = {col: np.abs(b.values()) for col, b in bufs.items()}

        def p(col, q):
            return float(np.percentile(vals[col], q)) if vals[col].size else np.nan

        err_rows.append({
            "threshold_db": thr, "count": n,
            "range_error_mean_m": mean,
            "range_error_std_m": float(np.sqrt(max(var, 0.0))) if n else np.nan,
            "range_abs_error_p50_m": p("range_error_m", 50),
            "range_abs_error_p95_m": p("range_error_m", 95),
            "azimuth_abs_error_p50_rad": p("azimuth_error_rad", 50),
            "azimuth_abs_error_p95_rad": p("azimuth_error_rad", 95),
            "elevation_abs_error_p50_rad": p("elevation_error_rad", 50),
            "elevation_abs_error_p95_rad": p("elevation_error_rad", 95),
            "rv_abs_error_p50_mps": p("radial_velocity_error_mps", 50),
            "rv_abs_error_p95_mps": p("radial_velocity_error_mps", 95),
            "sampled_for_quantiles": any(b.truncated for b in bufs.values()),
        })
    error_summary = pd.DataFrame(err_rows)

    return {"by_day": by_day, "overall": overall, "by_bin": by_bin,
            "clutter_by_bin": clu_by_bin, "overall_by_bin": overall_by_bin,
            "snr_summary": snr_summary, "error_summary": error_summary}


# =============================================================================
# Plots (matplotlib only, default colors, one figure per plot)
# =============================================================================

def make_plots(tables: Dict[str, pd.DataFrame], plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    overall = tables["overall"].sort_values("threshold_db")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(overall["threshold_db"], overall["empirical_pd"], marker="o")
    ax.set_xlabel("threshold (dB)")
    ax.set_ylabel("empirical Pd")
    ax.set_title("Empirical detection probability vs threshold (overall)")
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pd_vs_threshold.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(overall["threshold_db"], overall["false_alarm_per_frame"], marker="o")
    if (overall["false_alarm_per_frame"] > 0).all():
        ax.set_yscale("log")
    ax.set_xlabel("threshold (dB)")
    ax.set_ylabel("false alarms per frame")
    ax.set_title("Clutter false alarms vs threshold (overall)")
    ax.grid(True, linewidth=0.5, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "false_alarm_vs_threshold.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    obb = tables["overall_by_bin"]
    for label in obb.sort_values("bin_lo_m")["range_bin"].unique():
        sub = obb[obb["range_bin"] == label].sort_values("threshold_db")
        ax.plot(sub["threshold_db"], sub["empirical_pd_bin"], marker="o", label=label)
    ax.set_xlabel("threshold (dB)")
    ax.set_ylabel("empirical Pd")
    ax.set_title("Detection probability vs threshold, by truth range bin")
    ax.grid(True, linewidth=0.5)
    ax.legend(title="range bin")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pd_by_range_bin.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(overall["threshold_db"], overall["target_detections"], marker="o", label="target detections")
    ax.plot(overall["threshold_db"], overall["clutter_detections"], marker="o", label="clutter detections")
    ax.set_xlabel("threshold (dB)")
    ax.set_ylabel("detections")
    ax.set_title("Detections vs threshold (overall)")
    ax.grid(True, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "detections_by_threshold.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================

def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> List[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---:" for _ in cols]) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and not float(v).is_integer():
                cells.append(float_fmt.format(v))
            elif isinstance(v, float):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(output_dir: str, tables: Dict[str, pd.DataFrame], cfg: ThresholdEvalConfig,
                 relocated_seen: bool, summary: Optional[pd.DataFrame]) -> str:
    overall = tables["overall"].sort_values("threshold_db")
    by_day = tables["by_day"]
    obb = tables["overall_by_bin"]

    reloc_note = ""
    if summary is not None and "relocated_truth_fraction" in summary.columns:
        frac = summary["relocated_truth_fraction"].dropna()
        if len(frac):
            reloc_note = (f"The stage-6 summary reports a relocated truth fraction of "
                          f"{float(frac.mean()):.3f}; ")
    if relocated_seen:
        reloc_note += "detection files carry relocation metadata columns."

    lines = [
        "# Stage 07 Threshold-Only Baseline Evaluation",
        "",
        "## Experiment definition",
        "",
        "- This is a **threshold-only** point-detection baseline: each stage-6",
        "  detection stream is evaluated as-is, frame by frame, against the",
        "  stage-5 truth denominators.",
        "- **No tracking**, data association, or trajectory smoothing is used.",
        "- The current dataset is the **relocated wide-area weak-target",
        "  experiment**: trajectory starts are anchored 10-80 km from the radar,",
        "  full trajectories are retained, and long flights drift far beyond the",
        "  anchor band.",
        f"- Stage 6's `--max-range-m {cfg.coverage_range_m:.0f}` controlled **clutter",
        "  support**, not target containment: far targets remain and receive",
        "  lower SNR through the range-decay model.",
        f"- {reloc_note}" if reloc_note else "- No relocation metadata detected in inputs.",
        "",
        "## Overall operating curve",
        "",
    ]
    lines += _md_table(overall[["threshold_db", "truth_rows", "target_detections",
                                "clutter_detections", "missed_targets", "empirical_pd",
                                "false_alarm_per_frame", "target_fraction"]])
    lines += [
        "",
        "## Per-day consistency",
        "",
        "Empirical Pd per day and threshold (columns = threshold dB):",
        "",
    ]
    pd_pivot = by_day.pivot_table(index="date", columns="threshold_db",
                                  values="empirical_pd").round(4).reset_index()
    pd_pivot.columns = [str(c) for c in pd_pivot.columns]
    lines += _md_table(pd_pivot)
    lines += [
        "",
        "## Range-bin detection probability",
        "",
        "Empirical Pd by truth range bin (rows) and threshold (columns), summed",
        "across days:",
        "",
    ]
    bin_pivot = obb.pivot_table(index="range_bin", columns="threshold_db",
                                values="empirical_pd_bin").round(4)
    bin_pivot = bin_pivot.reindex(obb.sort_values("bin_lo_m")["range_bin"].unique()).reset_index()
    bin_pivot.columns = [str(c) for c in bin_pivot.columns]
    lines += _md_table(bin_pivot)

    clu = tables["clutter_by_bin"].groupby(["threshold_db", "range_bin", "bin_lo_m"]).agg(
        clutter_detections_bin=("clutter_detections_bin", "sum")).reset_index()
    clu_pivot = clu.pivot_table(index="range_bin", columns="threshold_db",
                                values="clutter_detections_bin")
    clu_pivot = clu_pivot.reindex(clu.sort_values("bin_lo_m")["range_bin"].unique()).reset_index()
    clu_pivot.columns = [str(c) for c in clu_pivot.columns]
    lines += [
        "",
        "## Clutter by measured range bin",
        "",
        "Clutter false alarms by *measured* range bin (counts, summed across",
        "days). Clutter is uniform in range within the configured support, so",
        "wider bins collect proportionally more:",
        "",
    ]
    lines += _md_table(clu_pivot)
    lines += [
        "",
        "## SNR distribution of written detections",
        "",
        "SNR is summarized **only for written detections**. Stage 6 does not",
        "write missed-target rows, so missed-target SNR values are not",
        "available -- **this is not Pd-by-SNR**. Quantiles are computed from a",
        f"capped deterministic sample of up to {QUANTILE_SAMPLE_CAP:,} values per",
        "group (seed 123); counts are exact.",
        "",
    ]
    lines += _md_table(tables["snr_summary"], float_fmt="{:.2f}")
    lines += [
        "",
        "## Measurement noise sanity check",
        "",
        "Target measurement errors (mean/std exact; percentiles from the capped",
        "sample). These should match the stage-6 noise configuration",
        "(sigma_range 75 m, sigma_az/el 0.15 deg = 2.62e-3 rad, sigma_rv 2 m/s):",
        "",
    ]
    lines += _md_table(tables["error_summary"], float_fmt="{:.4g}")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Lower thresholds recover substantially more targets but admit",
        "  proportionally more clutter; higher thresholds suppress clutter but",
        "  miss weak targets.",
        "- Range decay makes farther targets systematically less detectable:",
        "  the range-bin table shows Pd collapsing with distance at every",
        "  threshold.",
        "- A threshold alone cannot separate weak far targets from clutter --",
        "  which motivates Stage 08 tracking / physics-guided path scoring on",
        "  the low-threshold detection stream.",
        "",
        "## Recommended next stage",
        "",
        "Stage 08 should implement a **low-threshold detect-then-track",
        "baseline** -- starting with a constant-velocity Kalman filter and",
        "gating -- before any ML model.",
        "",
    ]

    path = os.path.join(output_dir, "threshold_only_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate
# =============================================================================

def run_validation_gate(output_dir: str, tables: Dict[str, pd.DataFrame]) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"Stage 07 validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    for name in ["threshold_by_day.csv", "threshold_overall.csv", "threshold_by_range_bin.csv"]:
        path = os.path.join(output_dir, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            fail(f"{name} missing or empty")
    print("  required output tables exist and are nonempty: OK")

    overall = tables["overall"]
    if not overall["empirical_pd"].between(0.0, 1.0).all():
        fail("empirical_pd outside [0, 1]")
    fa = overall["false_alarm_per_frame"].to_numpy()
    if not (np.isfinite(fa).all() and (fa >= 0).all()):
        fail("false_alarm_per_frame non-finite or negative")
    if not (overall["total_detections"]
            == overall["target_detections"] + overall["clutter_detections"]).all():
        fail("total_detections != target + clutter")
    print("  Pd in [0,1], false alarms finite/nonnegative, totals consistent: OK")

    bad_bins = tables["overall_by_bin"]
    zero_den = bad_bins[bad_bins["truth_rows_bin"] == 0]["range_bin"].unique()
    if len(zero_den):
        print(f"  note: range bin(s) with zero truth denominator: {list(zero_den)}")
    valid = bad_bins[bad_bins["truth_rows_bin"] > 0]
    if not valid["empirical_pd_bin"].between(0.0, 1.0).all():
        fail("range-bin empirical_pd outside [0, 1] where denominator > 0")
    print("  range-bin Pd valid where denominator > 0: OK")

    o = overall.sort_values("threshold_db")
    print("\n  threshold trend (report-only; expect both to generally decrease):")
    print("    empirical_pd:          " + " -> ".join(f"{v:.3f}" for v in o["empirical_pd"]))
    print("    false_alarm_per_frame: " + " -> ".join(f"{v:.2f}" for v in o["false_alarm_per_frame"]))


# =============================================================================
# Orchestration
# =============================================================================

def run_evaluation(cfg: ThresholdEvalConfig) -> Dict[str, pd.DataFrame]:
    key_output = os.path.join(cfg.output_dir, "threshold_overall.csv")
    if os.path.exists(key_output) and not cfg.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key_output}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    truth_files = discover_truth_files(cfg.truth_dir)
    if not truth_files:
        raise SystemExit(f"No radar_truth_YYYY-MM-DD.csv files found in {cfg.truth_dir}")
    detection_files = discover_detection_files(cfg.detections_dir)
    if not detection_files:
        raise SystemExit(f"No detections_*_thr_*dB.csv files found in {cfg.detections_dir}")

    summary = load_detection_summary_if_available(cfg.detections_dir)
    if summary is None:
        print("note: no sim_detection_summary.csv found; frame counts fall back to "
              "unique truth timestamps.")

    print(f"truth: {len(truth_files)} file(s); detections: {len(detection_files)} file(s)")
    truth_den = compute_truth_denominators(truth_files, cfg, need_frames_fallback=summary is None)
    det = compute_detection_metrics(detection_files, cfg)
    tables = build_overall_tables(truth_den, det, summary, cfg)

    tables["by_day"].to_csv(os.path.join(cfg.output_dir, "threshold_by_day.csv"), index=False)
    tables["overall"].to_csv(os.path.join(cfg.output_dir, "threshold_overall.csv"), index=False)
    tables["by_bin"].to_csv(os.path.join(cfg.output_dir, "threshold_by_range_bin.csv"), index=False)
    tables["clutter_by_bin"].to_csv(os.path.join(cfg.output_dir, "clutter_by_range_bin.csv"), index=False)
    tables["snr_summary"].to_csv(os.path.join(cfg.output_dir, "detected_snr_summary.csv"), index=False)
    tables["error_summary"].to_csv(os.path.join(cfg.output_dir, "measurement_error_summary.csv"), index=False)

    if not cfg.no_plots:
        make_plots(tables, os.path.join(cfg.output_dir, "plots"))

    report_path = write_report(cfg.output_dir, tables, cfg, det["relocated_seen"], summary)
    run_validation_gate(cfg.output_dir, tables)

    o = tables["overall"].sort_values("threshold_db")
    print(f"\nreport: {os.path.abspath(report_path)}")
    print(f"tables + plots in: {os.path.abspath(cfg.output_dir)}")
    print("\noverall operating curve:")
    print(o[["threshold_db", "empirical_pd", "false_alarm_per_frame",
             "target_detections", "clutter_detections"]].to_string(index=False))
    return tables


# =============================================================================
# Self-test (no real data required)
# =============================================================================

def self_test() -> None:
    """Tiny synthetic truth + detections -> full evaluation -> assertions."""
    date = "2022-01-01"
    t = 1_000.0 + 10.0 * np.arange(5)

    with tempfile.TemporaryDirectory() as tmp:
        truth_dir = os.path.join(tmp, "truth")
        det_dir = os.path.join(tmp, "det")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(truth_dir)
        os.makedirs(det_dir)

        # 10 truth rows across bins: 0-50 (3), 50-100 (3), 100-200 (2), >200 (2)
        truth_ranges = [20e3, 30e3, 40e3, 60e3, 80e3, 90e3, 110e3, 150e3, 250e3, 300e3]
        rows = []
        for i, r in enumerate(truth_ranges):
            rows.append(dict(trajectory_id="aaa_r0" if i < 5 else "bbb_r0",
                             sample_idx=i % 5, timestamp=t[i % 5], range_m=float(r)))
        pd.DataFrame(rows).to_csv(os.path.join(truth_dir, f"radar_truth_{date}.csv"), index=False)

        def det_rows(target_ranges, n_clutter):
            out = []
            for r in target_ranges:
                out.append(dict(is_target=1, truth_range_m=float(r), meas_range_m=float(r) + 50.0,
                                snr_db=10.0, range_error_m=50.0, azimuth_error_rad=0.001,
                                elevation_error_rad=0.001, radial_velocity_error_mps=1.0,
                                relocated=1))
            for i in range(n_clutter):
                out.append(dict(is_target=0, truth_range_m=np.nan,
                                meas_range_m=10e3 + 60e3 * i, snr_db=5.0,
                                range_error_m=np.nan, azimuth_error_rad=np.nan,
                                elevation_error_rad=np.nan, radial_velocity_error_mps=np.nan,
                                relocated=0))
            return pd.DataFrame(out)

        # thr 0: 6 targets + 4 clutter; thr 6: 3 targets + 1 clutter
        det_rows([20e3, 30e3, 60e3, 90e3, 110e3, 250e3], 4).to_csv(
            os.path.join(det_dir, f"detections_{date}_thr_0p0dB.csv"), index=False)
        det_rows([20e3, 60e3, 110e3], 1).to_csv(
            os.path.join(det_dir, f"detections_{date}_thr_6p0dB.csv"), index=False)
        pd.DataFrame([
            dict(date=date, threshold_db=0.0, status="created", frames=5,
                 empirical_pd=0.6, false_alarm_per_frame=0.8, relocated_truth_fraction=1.0),
            dict(date=date, threshold_db=6.0, status="created", frames=5,
                 empirical_pd=0.3, false_alarm_per_frame=0.2, relocated_truth_fraction=1.0),
        ]).to_csv(os.path.join(det_dir, SUMMARY_NAME), index=False)

        cfg = ThresholdEvalConfig(truth_dir=truth_dir, detections_dir=det_dir,
                                  output_dir=out_dir, chunksize=4, overwrite=True)
        tables = run_evaluation(cfg)

        assert os.path.isdir(out_dir), "output directory missing"
        for name in ["threshold_overall.csv", "threshold_by_range_bin.csv", "threshold_only_report.md"]:
            assert os.path.exists(os.path.join(out_dir, name)), f"{name} missing"

        overall = tables["overall"].set_index("threshold_db")
        assert overall.loc[0.0, "empirical_pd"] > overall.loc[6.0, "empirical_pd"], \
            "Pd should be higher at the lower threshold"
        assert overall.loc[0.0, "false_alarm_per_frame"] > overall.loc[6.0, "false_alarm_per_frame"], \
            "false alarms should be higher at the lower threshold"
        assert abs(overall.loc[0.0, "empirical_pd"] - 0.6) < 1e-9
        assert abs(overall.loc[0.0, "false_alarm_per_frame"] - 0.8) < 1e-9

        bins = set(tables["overall_by_bin"]["range_bin"])
        assert bins == {"0-50 km", "50-100 km", "100-200 km", ">200 km"}, f"unexpected bins: {bins}"
        obb = tables["overall_by_bin"].set_index(["threshold_db", "range_bin"])
        assert abs(obb.loc[(0.0, "0-50 km"), "empirical_pd_bin"] - 2 / 3) < 1e-9
        assert abs(obb.loc[(0.0, ">200 km"), "empirical_pd_bin"] - 0.5) < 1e-9

        text = open(os.path.join(out_dir, "threshold_only_report.md")).read()
        for needle in ["threshold-only", "No tracking", "range-bin", "Stage 08"]:
            assert needle in text, f"report missing expected text: {needle!r}"
        for plot in ["pd_vs_threshold.png", "false_alarm_vs_threshold.png",
                     "pd_by_range_bin.png", "detections_by_threshold.png"]:
            assert os.path.exists(os.path.join(out_dir, "plots", plot)), f"missing plot {plot}"

    print("\nStage 07 self-test passed.")
