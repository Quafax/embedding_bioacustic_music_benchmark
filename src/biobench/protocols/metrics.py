from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def multilabel_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probabilities >= threshold).astype(int)
    try:
        map_value = float(average_precision_score(y_true, probabilities, average="macro"))
    except ValueError:
        map_value = float("nan")
    return {
        "map_macro": map_value,
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "threshold": float(threshold),
    }
