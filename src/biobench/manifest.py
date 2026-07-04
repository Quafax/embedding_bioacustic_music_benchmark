from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from biobench.paths import project_root

REQUIRED_COLUMNS = ("clip_id", "audio_path", "split", "task", "labels")
VALID_SPLITS = {"train", "val", "test"}
VALID_TASKS = {"multiclass", "multilabel"}
OPTIONAL_COLUMNS = ("recording_id", "start_s", "end_s", "source_id", "notes")


def read_manifest(path: str | Path, *, check_files: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    validate_manifest(df, source=path, check_files=check_files)
    return df.reset_index(drop=True)


def validate_manifest(df: pd.DataFrame, *, source: Path | None = None, check_files: bool = False) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest contains no rows.")
    if df["clip_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every clip_id must be non-empty.")
    if df["clip_id"].duplicated().any():
        duplicates = df.loc[df["clip_id"].duplicated(), "clip_id"].head(5).tolist()
        raise ValueError(f"clip_id must be unique. Examples of duplicates: {duplicates}")
    splits = set(df["split"].astype(str))
    unknown_splits = splits - VALID_SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown split values: {sorted(unknown_splits)}")
    missing_splits = VALID_SPLITS - splits
    if missing_splits:
        raise ValueError(f"Manifest must contain train, val and test. Missing: {sorted(missing_splits)}")
    tasks = set(df["task"].astype(str))
    if len(tasks) != 1 or not tasks.issubset(VALID_TASKS):
        raise ValueError(f"'task' must be exactly one of {sorted(VALID_TASKS)} across the manifest; got {sorted(tasks)}")
    if df["labels"].astype(str).str.strip().eq("").any():
        raise ValueError("Every row needs a non-empty labels value.")
    if "start_s" in df.columns or "end_s" in df.columns:
        starts = pd.to_numeric(df.get("start_s", pd.Series([None] * len(df))), errors="coerce")
        ends = pd.to_numeric(df.get("end_s", pd.Series([None] * len(df))), errors="coerce")
        invalid = starts.notna() & ends.notna() & (ends <= starts)
        if invalid.any():
            bad = df.loc[invalid, "clip_id"].head(5).tolist()
            raise ValueError(f"end_s must be greater than start_s. Problem clips: {bad}")
    if check_files:
        root = project_root(source or Path.cwd())
        missing_files = [
            clip_id for clip_id, audio_path in zip(df["clip_id"], df["audio_path"])
            if not resolve_audio_path(audio_path, root).exists()
        ]
        if missing_files:
            preview = missing_files[:5]
            raise FileNotFoundError(f"Audio files missing for {len(missing_files)} clips. Examples: {preview}")


def resolve_audio_path(audio_path: str | Path, root: Path | None = None) -> Path:
    path = Path(audio_path)
    return path if path.is_absolute() else (root or project_root()) / path


def manifest_hash(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def task_type(df: pd.DataFrame) -> str:
    return str(df["task"].iloc[0])


def classes_from_train(df: pd.DataFrame) -> list[str]:
    train = df.loc[df["split"] == "train"]
    if task_type(df) == "multiclass":
        return sorted(train["labels"].astype(str).unique().tolist())
    labels: set[str] = set()
    for value in train["labels"].astype(str):
        labels.update(part.strip() for part in value.split(";") if part.strip())
    return sorted(labels)


def labels_as_lists(values: Iterable[str]) -> list[list[str]]:
    return [[part.strip() for part in str(value).split(";") if part.strip()] for value in values]
