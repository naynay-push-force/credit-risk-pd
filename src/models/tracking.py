"""
Experiment tracking: record each run and read runs back for comparison.

Each run writes a complete, self-describing run.json (config + metrics + git
provenance) into its run directory -- the source of truth. load_runs() rebuilds
a cross-run comparison table from those records on demand, so there is no
hand-maintained schema to keep in sync.
"""

import subprocess
import json
from pathlib import Path
from dataclasses import asdict

import pandas as pd

from config import RunConfig


def git_sha() -> str:
    """
    Return the current commit's 40-char SHA (or 'unknown' if git is
    unavailable). Pinned in each run record so the exact feature/model code
    behind a result is recoverable via `git checkout <sha>`.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """
    Return True if the working tree has uncommitted changes -- i.e. the SHA
    alone does not fully reproduce this run. It does not capture the diff; it
    just flags that the SHA is not the whole story. Returns True (untrusted) if
    the check itself fails.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return True


def log_run(
    run_dir: Path,
    run_id: str,
    cfg: RunConfig,
    metrics: dict,
) -> None:
    """
    Serialize the full run record -- config, metrics, and git provenance -- to
    run.json in the run directory. asdict(cfg) walks every field, so the record
    is complete by construction: adding a config knob never touches this writer.
    """
    record = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "metrics": metrics,
        "config": asdict(cfg),
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(record, f, indent=2)


def load_runs(reports_dir: Path = Path("reports")) -> pd.DataFrame:
    """
    Load every run.json under reports/ into one DataFrame, with nested keys
    flattened to dotted columns (e.g. 'metrics.test.auc'), sorted best-AUC
    first. This is the derived comparison view over the per-run records; it
    auto-unions keys, so new config fields appear as columns for free.
    """
    records = [json.load(open(p)) for p in reports_dir.glob("*/run.json")]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records).sort_values("metrics.test.auc", ascending=False)


def main() -> None:
    """Print a compact cross-run comparison table (`python -m src.models.tracking`)."""
    df = load_runs()
    cols = ["run_id", "git_sha", "git_dirty",
            "metrics.test.auc", "metrics.cv.roc_auc_mean",
            "config.class_weight", "config.calibration"]
    cols = [c for c in cols if c in df.columns]   # tolerate missing cols on empty/early runs
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
