from __future__ import annotations

from pathlib import Path


def project_root(anchor: Path | None = None) -> Path:
    """Find the folder containing pyproject.toml, starting at anchor or cwd."""
    current = (anchor or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find project root (pyproject.toml). Run from inside the project.")


def resolve_from_root(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root or project_root()) / path


def artifact_dir(root: Path | None = None) -> Path:
    path = (root or project_root()) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path
