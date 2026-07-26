import subprocess

from pathlib import Path
from dataclasses import asdict

import json
import pandas as pd

from config import RunConfig

def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"

def git_dirty() -> str:
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, check=True,
                             ).stdout.strip()
        return bool(out)
    except Exception:
        return "unknown"

def log_run(run_dir: Path, 
            run_id: str, 
            cfg: RunConfig, 
            metrics: dict,
) -> None:
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
    records = [json.load(open(p)) for p in reports_dir.glob("*/run.json")]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records).sort_values("metrics.test.auc", ascending=False)

def main() -> None:
    df = load_runs()
    cols = ["run_id", "git_sha", "git_dirty",
            "metrics.test.auc", "metrics.cv.roc_auc_mean",
            "config.class_weight", "config.calibration"]
    cols = [c for c in cols if c in df.columns]   # tolerate missing cols on empty/early runs
    print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
