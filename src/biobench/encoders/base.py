from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Literal

import torch

Pooling = Literal["mean", "max"]


@dataclass(frozen=True)
class EncoderSpec:
    encoder_id: str
    implementation: str
    model_id: str | None
    sample_rate: int
    batch_size: int
    layer: int | None = None
    pooling: Pooling = "mean"
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BaseEncoder(ABC):
    def __init__(self, spec: EncoderSpec, device: str = "auto") -> None:
        self.spec = spec
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else (device if device != "auto" else "cpu")
        )

    @abstractmethod
    def encode(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return [batch, frames, dim] or [batch, dim]."""

    @staticmethod
    def pool(
        embeddings: torch.Tensor,
        *,
        input_lengths: torch.Tensor,
        padded_input_length: int,
        pooling: Pooling,
    ) -> torch.Tensor:
        if embeddings.ndim == 2:
            return embeddings.detach().cpu()
        if embeddings.ndim != 3:
            raise ValueError(f"Expected [batch, frames, dim] or [batch, dim], got {tuple(embeddings.shape)}")
        batch, frames, _ = embeddings.shape
        if batch != len(input_lengths):
            raise ValueError("Embedding batch size does not match input lengths.")
        valid_frames = torch.ceil(frames * input_lengths.float() / max(1, padded_input_length)).long()
        valid_frames = valid_frames.clamp(min=1, max=frames)
        frame_index = torch.arange(frames, device=embeddings.device).unsqueeze(0)
        mask = frame_index < valid_frames.to(embeddings.device).unsqueeze(1)
        if pooling == "mean":
            summed = (embeddings * mask.unsqueeze(-1)).sum(dim=1)
            return (summed / valid_frames.to(embeddings.device).unsqueeze(1)).detach().cpu()
        if pooling == "max":
            masked = embeddings.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return masked.max(dim=1).values.detach().cpu()
        raise ValueError(f"Unsupported pooling '{pooling}'.")
