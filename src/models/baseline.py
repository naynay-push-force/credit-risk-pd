from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from config import RunConfig

def build_baseline_model(
        preprocessor: ColumnTransformer,
        cfg: RunConfig,
) -> Pipeline:
    """
    Build a simple baseline PD model using logistic regression.
    
    The pipeline includes preprocessing + model so that:
    - trainging and inference share the same transformations
    - the entire system can be saved as on object later"""

    # Safely get the value from config
    cw_config = cfg.class_weight

    if cw_config.lower() == "none":
        cw_config = None
 
    model = LogisticRegression(
        max_iter=1000,
        class_weight=cw_config,
        solver="lbfgs",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline