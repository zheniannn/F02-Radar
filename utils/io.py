"""Filesystem path helpers for the radar stages.

Paths are resolved from this file's location rather than the current working
directory, so scripts behave the same no matter where they're launched from.
All data directories live inside this repo (see README: inputs are copied in
from F01-PREPROCESSING).
"""

import os


def get_repo_root() -> str:
    """Return the F02-RADAR repo root (one level above utils/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Stage 5 (05_make_radar_truth.py): 10 s trajectories -> radar-coordinate truth ---

def get_trajectories_dir() -> str:
    """Directory of the uniform-grid trajectory CSVs (stage 4's output, copied in from F01)."""
    return os.path.join(get_repo_root(), "data", "active", "trajectories_10s")


def get_radar_truth_dir() -> str:
    """Directory for the per-day radar-coordinate truth CSVs and their summary."""
    return os.path.join(get_repo_root(), "data", "active", "radar_truth")


def get_radar_truth_summary_path(output_dir: str = "") -> str:
    """Path to the cross-day radar-truth summary CSV (kept next to the truth files)."""
    return os.path.join(output_dir or get_radar_truth_dir(), "radar_truth_summary.csv")
