from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from biobench.audio import collate_audio, load_audio
from biobench.cache import CacheLocation, cache_exists, cache_location, save_cache
from biobench.config import ExperimentConfig
from biobench.encoders import build_encoder, encoder_spec
from biobench.manifest import read_manifest


class ManifestAudioDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, sample_rate: int) -> None:
        self.rows = manifest.to_dict(orient="records")
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        clip = load_audio(row, target_sample_rate=self.sample_rate)
        return {"waveform": clip.waveform, "row": row}


def extract_embeddings(config: ExperimentConfig, *, device: str = "auto", force: bool = False) -> CacheLocation:
    manifest = read_manifest(config.data.manifest, check_files=True)
    spec = encoder_spec(config.encoder)
    location = cache_location(config.data.manifest, spec)
    if cache_exists(location) and not force:
        print(f"Using existing cache: {location.directory}")
        return location

    encoder = build_encoder(config.encoder, device=device)
    dataset = ManifestAudioDataset(manifest, sample_rate=spec.sample_rate)
    loader = DataLoader(dataset, batch_size=spec.batch_size, shuffle=False, num_workers=0, collate_fn=collate_audio)

    pooled_batches: list[np.ndarray] = []
    ordered_rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=f"Extract {spec.encoder_id}"):
        temporal_embeddings = encoder.encode(batch["audio"], batch["attention_mask"])
        pooled = encoder.pool(
            temporal_embeddings,
            input_lengths=batch["lengths"],
            padded_input_length=batch["audio"].shape[1],
            pooling=spec.pooling,
        )
        pooled_batches.append(pooled.numpy())
        ordered_rows.extend(batch["rows"])

    embeddings = np.concatenate(pooled_batches, axis=0)
    metadata = pd.DataFrame(ordered_rows)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "manifest_path": str(config.data.manifest),
        "encoder": spec.as_dict(),
        "n_examples": int(len(metadata)),
        "embedding_dimension": int(embeddings.shape[1]),
        "pooling_padding_rule": "valid frame count estimated proportionally from unpadded waveform length",
    }
    save_cache(location, embeddings, metadata, provenance)
    print(f"Saved cache: {location.directory}")
    return location
