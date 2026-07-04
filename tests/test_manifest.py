from __future__ import annotations

import pandas as pd
import pytest

from biobench.manifest import validate_manifest


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "clip_id": ["a", "b", "c"],
        "audio_path": ["a.wav", "b.wav", "c.wav"],
        "split": ["train", "val", "test"],
        "task": ["multiclass", "multiclass", "multiclass"],
        "labels": ["orca", "orca", "orca"],
    })


def test_valid_manifest_passes() -> None:
    validate_manifest(valid_frame())


def test_duplicate_clip_id_fails() -> None:
    frame = valid_frame()
    frame.loc[1, "clip_id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(frame)


def test_missing_split_fails() -> None:
    frame = valid_frame().iloc[:2].copy()
    with pytest.raises(ValueError, match="Missing"):
        validate_manifest(frame)
