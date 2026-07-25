from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import add_application_features

def score_applicant(
        features: dict,
        model_path: Path,
        threshold: float = 0.08,    # provisional; principled value comes from EL analysis
) -> tuple[float, str]:
    model = joblib.load(model_path)

    df = pd.DataFrame([features])
    df = add_application_features(df)

    pd_hat = model.predict_proba(df)[0, 1]

    decision = "reject" if pd_hat >= threshold else "approve"

    return (pd_hat, decision)

def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(ROOT / "data" / "raw" / "application_train.csv")
    features = df.iloc[0].to_dict()

    model_path = ROOT / "reports" / "2026-07-25_14-44-10_v3" / "model.joblib"  # your latest run
    pd_hat, decision = score_applicant(features, model_path)
    print(f"PD: {pd_hat:.4f} | decision: {decision} | actual TARGET: {features.get('TARGET')}")


if __name__ == "__main__":
    main()
