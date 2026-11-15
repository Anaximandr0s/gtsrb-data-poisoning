"""Load the YAML experiment configuration."""

from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the repository root, so relative paths work from scripts and notebooks alike."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    """Read a YAML config file (relative paths are taken from the project root) into a dict."""
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    """Return ``cfg['paths'][key]`` as an absolute path anchored at the project root."""
    return project_root() / cfg["paths"][key]
