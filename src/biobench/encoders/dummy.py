from __future__ import annotations

import torch

from biobench.encoders.base import BaseEncoder, EncoderSpec


class DummyEncoder(BaseEncoder):
    """A deterministic test encoder. Never report it as a scientific baseline."""

    def __init__(self, spec: EncoderSpec, device: str = "auto") -> None:
        super().__init__(spec, device)
        self.embedding_dim = 24

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        audio = audio.float()
        batches, samples = audio.shape
        chunks = 12
        padded = torch.nn.functional.pad(audio, (0, (-samples) % chunks))
        frames = padded.reshape(batches, chunks, -1)
        mean = frames.mean(dim=-1, keepdim=True)
        std = frames.std(dim=-1, keepdim=True).clamp_min(1e-6)
        energy = frames.square().mean(dim=-1, keepdim=True)
        base = torch.cat([mean, std, energy], dim=-1)
        repeats = (self.embedding_dim + base.shape[-1] - 1) // base.shape[-1]
        return base.repeat(1, 1, repeats)[..., : self.embedding_dim]
