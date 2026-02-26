"""
Experiment configuration for feature set versioning and model hyperparameters.

Version increment policy:
    - a version increment happens when feature_engineering.py or config.py changes in a way 
      that affects what the model sees.
    - Hyperparameter changes like class_weight or calibration method do not increment the version,
      they are tracked via the class_weight and calibration columns.
    - i.e., v1 -> v2: added a log transform, or dropped a column, or both.
      v1 run 1 -> v1 run 2: changed class_weight or calibration

# Note: this is a lightweight alternative to tools like MLflow or W&B.
"""

FEATURE_CONFIG = {
    "version": "v2",
    "class_weight": "balanced",
    "calibration": "platt",
    "notes": "run 5: same as run 1; dropped DAYS_BIRTH and DAYS_EMPLOYED",

    # Redundancy / multicolinearity management
    "drop_cols": ["DAYS_BIRTH", "DAYS_EMPLOYED"],
    "keep_cols": [],    # if non-empty, only these columns will be kept

    "force_categorical": [],
}

