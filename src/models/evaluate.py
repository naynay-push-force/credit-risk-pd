from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

@dataclass(frozen=True)
class EvalPaths:
    root: Path

    @property 
    def figures(
        self: Any
    ) -> Path:
        return self.root / "figures"
    
    @property
    def tables(
        self: Any
    ) -> Path:
        return self.root / "tables"
    
    def ensure(
            self: Any
    ) -> None:
        self.figures.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)

def _to_numpy(x) -> np.ndarray:
    return np.asarray(x)


# Plot ROC curve
def plot_roc(y_true, y_score, 
        outpath: Path,
) -> float:
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return float(auc)

# Plot Precision-Recall curve
def plot_pr(y_true, y_score,
            outpath: Path,
) -> float:
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision, label = f"PR (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return float(ap)


# Calibration curve + reliability table
def calibration_report(
    y_true,
    y_score,
    n_bins: int,
    outpath_fig: Path,
    outpath_table: Path,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Reliability / calibration curve:
    - strategy='quantile': bins by equal-sized quantiles of predicted score (good for imbalanced data)
    - returns a table with bin stats and saves a plot of the calibration curve
    """
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    frac_pos, mean_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy=strategy
    )

    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    tab = (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            avg_pred=("p", "mean"),
            obs_rate=("y", "mean"),
            p_min=("p", "min"),
            p_max=("p", "max"),
        )
        .reset_index(drop=False)
    )
    tab.to_csv(outpath_table, index=False)

    # Plot
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed default rate")
    plt.title(f"Calibration Curve ({strategy}, bins={n_bins})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_fig, dpi=200)
    plt.close()

    return tab


# Gains/lift table + curve
def gains_lift_table(
    y_true,
    y_score,
    n_bins: int,
    outpath_table: Path,
    outpath_fig: Path,
) -> pd.DataFrame:
    """
    Decile (or n-tile) analysis.
    Produces:
    - gains: cumulative % of bads captured as you move down the ranked list
    - lift: bad rate in bin / overall bad rate
    """
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    df = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p", ascending=False)
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")

    overall_bad_rate = df["y"].mean()
    total_bads = df["y"].sum()

    g = (
        df.groupby("bin", observed=True, sort=False)
        .agg(
            n=("y", "size"),
            bads=("y", "sum"),
            bad_rate=("y", "mean"),
            p_min=("p", "min"),
            p_max=("p", "max"),
        )
        .reset_index()
    )

    g["lift"] = g["bad_rate"] / overall_bad_rate
    g["cum_bads"] = g["bads"].cumsum()
    g["cum_bads_pct"] = g["cum_bads"] / total_bads

    g.to_csv(outpath_table, index=False)

    # Plot cumulative gains (x = population %, y = % bads captured)
    g["cum_pop_pct"] = g["n"].cumsum() / g["n"].sum()

    plt.figure()
    plt.plot(g["cum_pop_pct"], g["cum_bads_pct"], marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("Cumulative population (ranked by score)")
    plt.ylabel("Cumulative defaults captured")
    plt.title(f"Cumulative Gains (bins={n_bins})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_fig, dpi=200)
    plt.close()

    return g


# Score distribution by class
def score_distribution_plot(y_true, y_score, outpath: Path) -> None:
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    df = pd.DataFrame({"y": y_true, "p": y_score})
    good = df.loc[df["y"] == 0, "p"]
    bad = df.loc[df["y"] == 1, "p"]

    plt.figure()
    plt.hist(good, bins=50, alpha=0.7, label="Non-default (0)")
    plt.hist(bad, bins=50, alpha=0.7, label="Default (1)")
    plt.xlabel("Predicted PD")
    plt.ylabel("Count")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Logistic regression coefficients table
def logistic_coefficients_table(model, 
                                feature_names: list[str], 
                                outpath: Path, 
                                top_k: int = 30
) -> pd.DataFrame:
    """
    Extract coefficients from a Pipeline(preprocessor -> LogisticRegression).
    Assumes binary classification logistic regression.
    """
    # model should be the trained sklearn Pipeline
    lr = model.named_steps.get("model")
    if lr is None:
        raise ValueError("Pipeline missing named step 'model' (LogisticRegression).")

    coefs = lr.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_k)

    df.to_csv(outpath, index=False)
    return df

