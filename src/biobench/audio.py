from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from biobench.manifest import resolve_audio_path
from biobench.paths import project_root


@dataclass(frozen=True)
class AudioClip:
    waveform: torch.Tensor
    sample_rate: int


def load_audio(row: dict[str, Any], target_sample_rate: int) -> AudioClip:
    """Load mono audio, optionally crop by seconds, then resample for the selected encoder."""
    audio_path = resolve_audio_path(str(row["audio_path"]), project_root())
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file missing for {row['clip_id']}: {audio_path}")

    info = sf.info(audio_path)
    start_s = _optional_float(row.get("start_s"))
    end_s = _optional_float(row.get("end_s"))
    start_frame = max(0, int(math.floor(start_s * info.samplerate))) if start_s is not None else 0
    stop_frame = int(math.ceil(end_s * info.samplerate)) if end_s is not None else None
    samples, source_sr = sf.read(audio_path, start=start_frame, stop=stop_frame, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError(f"Clip is empty after cropping: {row['clip_id']}")
    mono = samples.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(mono, dtype=np.float32))
    if source_sr != target_sample_rate:
        waveform = resample_linear(waveform, source_sr, target_sample_rate)
    return AudioClip(waveform=waveform.contiguous(), sample_rate=target_sample_rate)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def resample_linear(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    """Dependency-light waveform resampling. Native model sample rate is recorded in metadata."""
    if source_sr == target_sr:
        return waveform
    target_length = max(1, round(waveform.numel() * target_sr / source_sr))
    return F.interpolate(
        waveform.reshape(1, 1, -1), size=target_length, mode="linear", align_corners=False
    ).reshape(-1)


def collate_audio(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["waveform"].numel() for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    audio = torch.zeros((len(batch), max_length), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for index, item in enumerate(batch):
        waveform = item["waveform"]
        audio[index, : waveform.numel()] = waveform
        attention_mask[index, : waveform.numel()] = 1
    return {
        "audio": audio,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "rows": [item["row"] for item in batch],
    }
