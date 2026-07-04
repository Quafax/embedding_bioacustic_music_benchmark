from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _sample_balanced(labels: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray | None:
    selected: list[int] = []
    for class_id in np.unique(labels):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < shots:
            return None
        selected.extend(rng.choice(candidates, size=shots, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _prototype_predict(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
    classes = np.unique(train_y)
    prototypes = np.stack([train_x[train_y == class_id].mean(axis=0) for class_id in classes])
    distances = ((eval_x[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=-1)
    return classes[distances.argmin(axis=1)]


def evaluate_fewshot_prototype(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    shots: tuple[int, ...],
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    train_idx = np.flatnonzero(splits == "train")
    test_idx = np.flatnonzero(splits == "test")
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []
    for shot in shots:
        for repeat in range(repeats):
            local_indices = _sample_balanced(labels[train_idx], shot, rng)
            if local_indices is None:
                results.append({"shots": shot, "repeat": repeat, "status": "skipped_insufficient_examples"})
                continue
            selected = train_idx[local_indices]
            prediction = _prototype_predict(embeddings[selected], labels[selected], embeddings[test_idx])
            results.append({
                "shots": shot,
                "repeat": repeat,
                "status": "ok",
                "test_accuracy": float(accuracy_score(labels[test_idx], prediction)),
                "test_macro_f1": float(f1_score(labels[test_idx], prediction, average="macro", zero_division=0)),
            })
    return results
