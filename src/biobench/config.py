import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any

#Here some Error definitions for clarity
class ConfigError(ValueError):
    """Error in configuration."""




#Classes from the yaml files
@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    manifest: Path


@dataclass(frozen=True)
class ProtocolConfig:
    kind: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    encoder: str
    protocol: ProtocolConfig