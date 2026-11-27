"""Fetch the released checkpoints and build the GTSRB cache: ``python -m src.download`` (``make models``)."""

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

from src.config import load_config, resolve_path
from src.data import load_gtsrb, load_train_groups

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoints(cfg: dict) -> list[Path]:
    """Download every checkpoint listed in ``release.checkpoints`` whose local copy is missing or differs.

    Each file is written next to its target and moved into place only once its hash matches,
    so an interrupted or tampered download never leaves a checkpoint that ``torch.load`` would read.
    Returns the files that were downloaded.
    """
    models_dir = resolve_path(cfg, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, expected in cfg["release"]["checkpoints"].items():
        target = models_dir / name
        if target.exists() and sha256(target) == expected:
            continue
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(f"{cfg['release']['url']}/{name}") as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        if (actual := sha256(partial)) != expected:
            partial.unlink()
            raise ValueError(f"{name}: sha256 {actual} does not match the release ({expected})")
        partial.replace(target)
        downloaded.append(target)
        logger.info("Downloaded %s", target)
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    download_checkpoints(config)
    load_gtsrb(config)
    load_train_groups(config)
