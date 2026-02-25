# python -m src.models.run_evaluation

from pathlib import Path

import numpy as np

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
    paths = EvalPaths(Path("reports"))
    paths.ensure()

    # Train model + get predictions (val)
    model, X_val, y_val, y_val_pred = train_and_predict()

    # Curves
    auc = plot_roc(y_val, y_val_pred, paths.figures / "roc_curve.png")
    ap = plot_pr(y_val, y_val_pred, paths.figures / "pr_curve.png")

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
    # We need feature names from the fitted preprocessor
    pre = model.named_steps["preprocessor"]
    feature_names = pre.get_feature_names_out().tolist()
    logistic_coefficients_table(
        model,
        feature_names,
        outpath=paths.tables / "top_coefficients.csv",
        top_k=40,
    )

    print(f"Saved evaluation artifacts to {paths.root.resolve()}")
    print(f"Validation ROC AUC: {auc:.6f}")
    print(f"Validation PR-AUC: {ap:.6f}")


if __name__ == "__main__":
    main()