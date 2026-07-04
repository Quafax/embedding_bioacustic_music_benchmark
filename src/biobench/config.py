import yaml
from pathlib import Path
from dataclasses import dataclass

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