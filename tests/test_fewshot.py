from __future__ import annotations

import numpy as np

from biobench.protocols.fewshot import evaluate_fewshot_prototype


def test_fewshot_runs() -> None:
    embeddings = np.array([[0.0], [0.1], [1.0], [1.1], [0.0], [1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 0, 1])
    splits = np.array(["train", "train", "train", "train", "test", "test"])
    result = evaluate_fewshot_prototype(embeddings, labels, splits, shots=(1,), repeats=2, seed=42)
    assert len(result) == 2
    assert all(row["status"] == "ok" for row in result)
