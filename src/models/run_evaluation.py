# python -m src.models.run_evaluation

from config import FEATURE_CONFIG

from pathlib import Path
import datetime as dt
import csv

import numpy as np

from src.models.metrics import ks_statistic

from src.models.evaluate import (
    EvalPaths,
    plot_roc,
    plot_pr,
    calibration_report,
    gains_lift_table,
    score_distribution_plot,
    logistic_coefficients_table,
)
from src.models.train_baseline import train_and_predict


def main() -> None:
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    paths = EvalPaths(Path(f"reports/{run_id}_{FEATURE_CONFIG["version"]}"))
    paths.ensure()

    # Train model + get predictions (val)
    model, X_val, y_val, y_val_pred = train_and_predict()

    # Curves
    auc = plot_roc(y_val, y_val_pred, paths.figures / "roc_curve.png")
    pr_auc = plot_pr(y_val, y_val_pred, paths.figures / "pr_curve.png")
    ks, ks_thresh = ks_statistic(y_val, y_val_pred)

    # Calibration + reliability table
    calibration_report(
        y_val,
        y_val_pred,
        n_bins=10,
        strategy="quantile",
        outpath_fig=paths.figures / "calibration_curve.png",
        outpath_table=paths.tables / "calibration_table.csv",
    )

    # Gains/lift
    gains_lift_table(
        y_val,
        y_val_pred,
        n_bins=10,
        outpath_table=paths.tables / "gains_lift_table.csv",
        outpath_fig=paths.figures / "gains_curve.png",
    )

    # Score distributions
    score_distribution_plot(y_val, y_val_pred, paths.figures / "score_distribution.png")

    # Feature names + coefficients
    # Extract feature names from the fitted preprocessor
    pre = model.named_steps["preprocessor"]
    feature_names = pre.get_feature_names_out().tolist()
    logistic_coefficients_table(
        model,
        feature_names,
        outpath=paths.tables / "top_coefficients.csv",
        top_k=40,
    )

    # Write experiment record
    exp_path = Path("results/experiments.csv")
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not exp_path.exists()

    with open(exp_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "version", "class_weight", "calibration", "auc", "pr_auc", "ks", "ks_thresh", "notes"
        ])
        if write_header:
            writer.writeheader() # write header only if file is new
        writer.writerow({
            "run_id": run_id,
            "version": FEATURE_CONFIG["version"],
            "class_weight": FEATURE_CONFIG.get("class_weight", "balanced"),
            "calibration": FEATURE_CONFIG.get("calibration", "none"),
            "auc": round(auc, 6),
            "pr_auc": round(pr_auc, 6),
            "ks": round(ks, 6),
            "ks_thresh": round(ks_thresh, 6),
            "notes": FEATURE_CONFIG["notes"],
        })
    
    print(f"Run {run_id} complete. Saved evaluation artifacts to {paths.root.resolve()}")
    print(f"AUC: {auc:.6f} | PR-AUC: {pr_auc:.6f} | KS: {ks:.6f} | KS_THRESH: {ks_thresh:.6f}")

if __name__ == "__main__":
    main()