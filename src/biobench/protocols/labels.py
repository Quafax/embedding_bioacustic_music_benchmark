from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from biobench.manifest import classes_from_train, labels_as_lists, task_type


def encode_targets(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str], str]:
    task = task_type(metadata)
    classes = classes_from_train(metadata)
    if task == "multiclass":
        lookup = {label: index for index, label in enumerate(classes)}
        labels = metadata["labels"].astype(str).to_numpy()
        unknown = sorted(set(labels) - set(classes))
        if unknown:
            raise ValueError(f"Labels outside training classes are not supported: {unknown}")
        return np.asarray([lookup[label] for label in labels], dtype=int), classes, task
    mlb = MultiLabelBinarizer(classes=classes)
    targets = mlb.fit_transform(labels_as_lists(metadata["labels"].astype(str)))
    return targets.astype(int), classes, task
