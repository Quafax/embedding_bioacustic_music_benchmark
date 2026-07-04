from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biobench.config import LinearProbeConfig
from biobench.protocols.metrics import multiclass_metrics, multilabel_metrics


@dataclass(frozen=True)
class LinearResult:
    selected_c: float
    selected_threshold: float | None
    validation: dict[str, float]
    test: dict[str, float]
    test_predictions: np.ndarray
    test_probabilities: np.ndarray | None


def _multiclass_model(c_value: float, max_iter: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=c_value, max_iter=max_iter)),
    ])


def _multilabel_model(c_value: float, max_iter: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", OneVsRestClassifier(LogisticRegression(C=c_value, max_iter=max_iter))),
    ])


def evaluate_linear_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    splits: np.ndarray,
    task: str,
    config: LinearProbeConfig,
) -> LinearResult:
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    if not train.any() or not val.any() or not test.any():
        raise ValueError("Need train, val and test rows for linear probing.")

    if task == "multiclass":
        best: tuple[float, Pipeline, dict[str, float]] | None = None
        for c_value in config.c_values:
            model = _multiclass_model(c_value, config.max_iter)
            model.fit(embeddings[train], targets[train])
            metrics = multiclass_metrics(targets[val], model.predict(embeddings[val]))
            if best is None or metrics["accuracy"] > best[2]["accuracy"]:
                best = (c_value, model, metrics)
        assert best is not None
        c_value, model, validation = best
        test_prediction = model.predict(embeddings[test])
        return LinearResult(
            selected_c=c_value,
            selected_threshold=None,
            validation=validation,
            test=multiclass_metrics(targets[test], test_prediction),
            test_predictions=test_prediction,
            test_probabilities=None,
        )

    if task == "multilabel":
        best: tuple[float, float, Pipeline, dict[str, float]] | None = None
        for c_value in config.c_values:
            model = _multilabel_model(c_value, config.max_iter)
            model.fit(embeddings[train], targets[train])
            probabilities = model.predict_proba(embeddings[val])
            for threshold in config.threshold_values:
                metrics = multilabel_metrics(targets[val], probabilities, threshold)
                score = metrics["map_macro"]
                if best is None or score > best[3]["map_macro"]:
                    best = (c_value, threshold, model, metrics)
        assert best is not None
        c_value, threshold, model, validation = best
        test_probabilities = model.predict_proba(embeddings[test])
        test_prediction = (test_probabilities >= threshold).astype(int)
        return LinearResult(
            selected_c=c_value,
            selected_threshold=threshold,
            validation=validation,
            test=multilabel_metrics(targets[test], test_probabilities, threshold),
            test_predictions=test_prediction,
            test_probabilities=test_probabilities,
        )

    raise ValueError(f"Unsupported task '{task}'.")
