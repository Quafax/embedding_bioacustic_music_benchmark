from __future__ import annotations

import torch

from biobench.encoders import build_encoder


def test_dummy_encoder_shape() -> None:
    encoder = build_encoder("dummy", device="cpu")
    output = encoder.encode(torch.randn(2, 120), torch.ones(2, 120, dtype=torch.long))
    pooled = encoder.pool(output, input_lengths=torch.tensor([120, 60]), padded_input_length=120, pooling="mean")
    assert output.ndim == 3
    assert pooled.shape == (2, 24)
