from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biobench.encoders.base import EncoderSpec
from biobench.manifest import manifest_hash
from biobench.paths import artifact_dir


@dataclass(frozen=True)
class CacheLocation:
    directory: Path
    key: str

    @property
    def embeddings_path(self) -> Path:
        return self.directory / "embeddings.npz"

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.csv"

    @property
    def provenance_path(self) -> Path:
        return self.directory / "provenance.json"


def cache_location(manifest_path: Path, spec: EncoderSpec) -> CacheLocation:
    payload = {
        "manifest_sha256_16": manifest_hash(manifest_path),
        "encoder": spec.as_dict(),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    dataset_name = manifest_path.stem
    directory = artifact_dir() / "embeddings" / dataset_name / spec.encoder_id / key
    return CacheLocation(directory=directory, key=key)


def cache_exists(location: CacheLocation) -> bool:
    return all(path.exists() for path in [location.embeddings_path, location.metadata_path, location.provenance_path])


def save_cache(location: CacheLocation, embeddings: np.ndarray, metadata: pd.DataFrame, provenance: dict[str, Any]) -> None:
    location.directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(location.embeddings_path, embeddings=embeddings.astype(np.float32, copy=False))
    metadata.to_csv(location.metadata_path, index=False)
    location.provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")


def load_cache(location: CacheLocation) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if not cache_exists(location):
        raise FileNotFoundError(f"Embedding cache not found: {location.directory}")
    embeddings = np.load(location.embeddings_path)["embeddings"]
    metadata = pd.read_csv(location.metadata_path, keep_default_na=False)
    provenance = json.loads(location.provenance_path.read_text(encoding="utf-8"))
    return embeddings, metadata, provenance
