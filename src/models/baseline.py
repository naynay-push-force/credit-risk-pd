from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_baseline_model(
        preprocessor
) -> Pipeline:
    """
    Build a simple baseline PD model using logistic regression.
    
    The pipeline includes preprocessing + model so that:
    - trainging and inference share the same transformations
    - the entire system can be saved as on object later"""

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