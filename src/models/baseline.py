from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_baseline_model(
        preprocessor
) -> Pipeline:
    """
    Build a simple baseline PD model using logistic regression.
    
    The pipeline """

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline