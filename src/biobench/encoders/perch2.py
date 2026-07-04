from __future__ import annotations

import torch

from biobench.encoders.base import BaseEncoder, EncoderSpec


class Perch2Encoder(BaseEncoder):
    """Single integration point for the exact official Perch 2.0 export you choose.

    The benchmark intentionally keeps TensorFlow/Kaggle/Perch-Hoplite specifics here.
    Do not change the dataset loader, cache, or evaluation code to add Perch.
    """

    def __init__(self, spec: EncoderSpec, device: str = "auto") -> None:
        super().__init__(spec, device)
        raise NotImplementedError(
            "Perch 2.0 needs one verified checkpoint-specific wrapper. Read docs/05_ENCODERS.md. "
            "Implement only Perch2Encoder.__init__ and .encode in this file; all other pipeline code is ready."
        )

    def encode(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
