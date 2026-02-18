import numpy as np
import pandas as pd
from typing import Tuple

def ks_statistic(
        y_true: pd.Series,
        y_score: np.ndarray
) -> Tuple[float, float]:
    """
    Compute KS statistic for binary classification.
    
    KS
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by score ascending
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # Total defaulters/non-defaulters
    n_def = np.sum(y_true_sorted == 1)
    n_ndef = np.sum(y_true_sorted == 0)

    if n_def == 0 or n_ndef == 0:
        raise ValueError("KS undefined. Need samples of both classes.")
    
    # Emprical CDFs of scores for defaulters and non-defaulters respectively
    cum_def = np.cumsum(y_true_sorted == 1) / n_def
    cum_ndef = np.cumsum(y_true_sorted == 0) / n_ndef

    # KS is max separation
    diff = cum_def - cum_ndef
    ks_idx = np.argmax(np.abs(diff))
    ks = np.abs(diff[ks_idx])
    ks_threshold = y_score_sorted[ks_idx]

    return float(ks), float(ks_threshold)