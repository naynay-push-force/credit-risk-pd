"""
Single-applicant scoring from the persisted model.

Loads a saved model (no retraining) and returns a calibrated PD plus an
approve/reject decision for one applicant. This is what makes on-demand PD
scoring real; a CLI / FastAPI / Streamlit front end is a thin wrapper over
score_applicant.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import add_application_features


def score_applicant(
    features: dict,
    model_path: Path,
    threshold: float = 0.08,    # provisional; principled value comes from the EL analysis
) -> tuple[float, str]:
    """
    Score one applicant.

    `features` is a raw-field dict (as from an application form / JSON payload);
    it is feature-engineered here, run through the loaded model, and compared
    against `threshold` to yield (pd, decision).

    Note the manual add_application_features call: a future custom transformer
    inside the persisted pipeline would let the model go raw-row -> PD on its
    own and remove this step.
    """
    model = joblib.load(model_path)

    df = pd.DataFrame([features])
    df = add_application_features(df)

    pd_hat = float(model.predict_proba(df)[0, 1])

    decision = "reject" if pd_hat >= threshold else "approve"

    return (pd_hat, decision)


def main() -> None:
    """Demo: score the first row of the training data and print its PD/decision."""
    ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ -> ..)
    df = pd.read_csv(ROOT / "data" / "raw" / "application_train.csv")
    features = df.iloc[0].to_dict()

    model_path = ROOT / "reports" / "2026-07-25_14-44-10_v3" / "model.joblib"  # your latest run
    pd_hat, decision = score_applicant(features, model_path)
    print(f"PD: {pd_hat:.4f} | decision: {decision} | actual TARGET: {features.get('TARGET')}")


if __name__ == "__main__":
    main()
