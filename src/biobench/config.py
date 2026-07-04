from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from biobench.paths import resolve_from_root


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    manifest: Path


@dataclass(frozen=True)
class LinearProbeConfig:
    kind: Literal["linear_probe"]
    c_values: tuple[float, ...]
    threshold_values: tuple[float, ...]
    max_iter: int


@dataclass(frozen=True)
class FewShotConfig:
    kind: Literal["fewshot_prototype"]
    shots: tuple[int, ...]
    repeats: int


ProtocolConfig = LinearProbeConfig | FewShotConfig


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    run: RunConfig
    data: DataConfig
    encoder: str
    protocol: ProtocolConfig


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping in the YAML config.")
    return value


def _as_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{name}' must be a positive integer, got {value!r}.")
    return value


def _as_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty list.")
    try:
        converted = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must contain numbers.") from exc
    if any(item <= 0 for item in converted):
        raise ValueError(f"'{name}' must contain positive numbers.")
    return converted


def load_experiment(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw = _expect_mapping(raw, "root")

    run_raw = _expect_mapping(raw.get("run"), "run")
    run_name = run_raw.get("name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("'run.name' must be a non-empty string.")
    run = RunConfig(name=run_name.strip(), seed=_as_positive_int(run_raw.get("seed"), "run.seed"))

    data_raw = _expect_mapping(raw.get("data"), "data")
    manifest_value = data_raw.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("'data.manifest' must be a non-empty path string.")
    data = DataConfig(manifest=resolve_from_root(manifest_value))

    encoder = raw.get("encoder")
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError("'encoder' must be a registered encoder ID string.")

    protocol_raw = _expect_mapping(raw.get("protocol"), "protocol")
    kind = protocol_raw.get("kind")
    if kind == "linear_probe":
        thresholds = protocol_raw.get("threshold_values", [0.5])
        if not isinstance(protocol_raw.get("max_iter"), int):
            raise ValueError("'protocol.max_iter' must be an integer for linear_probe.")
        protocol: ProtocolConfig = LinearProbeConfig(
            kind="linear_probe",
            c_values=_as_float_tuple(protocol_raw.get("c_values"), "protocol.c_values"),
            threshold_values=_as_float_tuple(thresholds, "protocol.threshold_values"),
            max_iter=_as_positive_int(protocol_raw.get("max_iter"), "protocol.max_iter"),
        )
    elif kind == "fewshot_prototype":
        shots_raw = protocol_raw.get("shots")
        if not isinstance(shots_raw, list) or not shots_raw:
            raise ValueError("'protocol.shots' must be a non-empty list for fewshot_prototype.")
        protocol = FewShotConfig(
            kind="fewshot_prototype",
            shots=tuple(_as_positive_int(value, "protocol.shots item") for value in shots_raw),
            repeats=_as_positive_int(protocol_raw.get("repeats"), "protocol.repeats"),
        )
    else:
        raise ValueError("'protocol.kind' must be 'linear_probe' or 'fewshot_prototype'.")

    return ExperimentConfig(
        source_path=source_path,
        run=run,
        data=data,
        encoder=encoder.strip(),
        protocol=protocol,
    )
