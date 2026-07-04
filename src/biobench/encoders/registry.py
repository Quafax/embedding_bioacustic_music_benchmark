from __future__ import annotations

from biobench.encoders.base import BaseEncoder, EncoderSpec
from biobench.encoders.dummy import DummyEncoder
from biobench.encoders.mert import MERTEncoder
from biobench.encoders.muq import MuQEncoder
from biobench.encoders.perch2 import Perch2Encoder


# This is deliberately the single source of truth for technical model settings.
# Experiment YAMLs name an ID; they do not repeat sample rate/checkpoint/batch size.
SPECS: dict[str, EncoderSpec] = {
    "dummy": EncoderSpec(
        encoder_id="dummy",
        implementation="dummy",
        model_id=None,
        sample_rate=24_000,
        batch_size=8,
        notes="Pipeline smoke test only; not scientific.",
    ),
    "mert_95m": EncoderSpec(
        encoder_id="mert_95m",
        implementation="mert",
        model_id="m-a-p/MERT-v1-95M",
        sample_rate=24_000,
        batch_size=4,
        layer=-1,
        pooling="mean",
        notes="Official MERT v1 95M checkpoint; last hidden layer.",
    ),
    "muq_large": EncoderSpec(
        encoder_id="muq_large",
        implementation="muq",
        model_id="OpenMuQ/MuQ-large-msd-iter",
        sample_rate=24_000,
        batch_size=2,
        layer=-1,
        pooling="mean",
        notes="Public MuQ checkpoint; use fp32 inference.",
    ),
    "perch2": EncoderSpec(
        encoder_id="perch2",
        implementation="perch2",
        model_id="TO_BE_LOCKED_BEFORE_FIRST_PERCH_RUN",
        sample_rate=32_000,
        batch_size=1,
        layer=None,
        pooling="mean",
        notes="Checkpoint/API must be locked and verified in encoders/perch2.py before use.",
    ),
}

BUILDERS = {
    "dummy": DummyEncoder,
    "mert": MERTEncoder,
    "muq": MuQEncoder,
    "perch2": Perch2Encoder,
}


def available_encoders() -> list[str]:
    return sorted(SPECS)


def encoder_spec(encoder_id: str) -> EncoderSpec:
    try:
        return SPECS[encoder_id]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder '{encoder_id}'. Available: {available_encoders()}") from exc


def build_encoder(encoder_id: str, device: str = "auto") -> BaseEncoder:
    spec = encoder_spec(encoder_id)
    return BUILDERS[spec.implementation](spec, device=device)
