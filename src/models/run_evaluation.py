# python -m src.models.run_evaluation

from config import FEATURE_CONFIG

from pathlib import Path
import datetime as dt
import time
import csv

from src.models.metrics import ks_statistic

from src.models.evaluate import (
    EvalPaths,
    run_cv,
    plot_roc,
    plot_pr,
    calibration_report,
    gains_lift_table,
    score_distribution_plot,
    logistic_coefficients_table,
)
from src.models.pipeline import (
    load_data,
    make_splits,
    build_pipeline,
    train,
)
from src.features.feature_engineering import add_application_features
from src.features.preprocessing import identify_feature_types

from sklearn.calibration import CalibratedClassifierCV

def main() -> None:
    start_time = time.perf_counter()

    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    paths = EvalPaths(Path(f"reports/{run_id}_{FEATURE_CONFIG["version"]}"))
    paths.ensure()

    ROOT = Path(__file__).resolve().parent.parent.parent
    data_path = ROOT / "data" / "raw" / "application_train.csv"
    df = load_data(data_path)

    df = add_application_features(df)
    
    X_train, X_test, y_train, y_test = make_splits(df, FEATURE_CONFIG)
    numeric_cols, categorical_cols = identify_feature_types(X_train)
    model = build_pipeline(numeric_cols, categorical_cols)

    # Stratified k-fold validation
    results = run_cv(model=model, X_train=X_train, y_train=y_train)
    print(f"CV ROC AUC: {results['roc_auc_mean']:.6f} +/- {results['roc_auc_std']:.6f}")
    print(f"CV PR-AUC:  {results['pr_auc_mean']:.6f} +/- {results['pr_auc_std']:.6f}")

    # Train and predict
    model = train(model, X_train, y_train, FEATURE_CONFIG)
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Curves
    auc = plot_roc(y_test, y_test_pred, paths.figures / "roc_curve.png")
    pr_auc = plot_pr(y_test, y_test_pred, paths.figures / "pr_curve.png")
    ks, ks_thresh = ks_statistic(y_test, y_test_pred)

    # Calibration + reliability table
    calibration_report(
        y_test,
        y_test_pred,
        n_bins=10,
        strategy="quantile",
        outpath_fig=paths.figures / "calibration_curve.png",
        outpath_table=paths.tables / "calibration_table.csv",
    )

    # Gains/lift
    gains_lift_table(
        y_test,
        y_test_pred,
        n_bins=10,
        outpath_table=paths.tables / "gains_lift_table.csv",
        outpath_fig=paths.figures / "gains_curve.png",
    )

    # Score distributions
    score_distribution_plot(y_test, y_test_pred, paths.figures / "score_distribution.png")

    # Extract the first fold's fitted pipeline to allow for feature name extraction
    if isinstance(model, CalibratedClassifierCV):
        base_pipeline = model.calibrated_classifiers_[0].estimator
    else:
        base_pipeline = model

    # Feature names + coefficients
    # Extract feature names from the fitted preprocessor
    pre = base_pipeline.named_steps["preprocessor"]
    feature_names = pre.get_feature_names_out().tolist()
    logistic_coefficients_table(
        base_pipeline,
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

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Script executed in {execution_time:.2f} seconds")
if __name__ == "__main__":
    main()