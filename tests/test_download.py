import hashlib

import pytest

from src.config import load_config, resolve_path
from src.download import download_checkpoints, sha256


@pytest.fixture
def release(tmp_path):
    """A fake release folder served through file:// URLs, and a config pointing at it."""
    assets = tmp_path / "release"
    assets.mkdir()
    (assets / "a.pt").write_bytes(b"weights-a")
    (assets / "b.pt").write_bytes(b"weights-b")
    cfg = load_config()
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["release"] = {
        "url": assets.as_uri(),
        "checkpoints": {name: hashlib.sha256(data).hexdigest() for name, data in
                        (("a.pt", b"weights-a"), ("b.pt", b"weights-b"))},
    }
    return cfg


def test_downloads_missing_checkpoints_then_skips_them(release):
    first = download_checkpoints(release)
    models_dir = resolve_path(release, "models_dir")
    assert [p.name for p in first] == ["a.pt", "b.pt"] and (models_dir / "a.pt").read_bytes() == b"weights-a"
    assert download_checkpoints(release) == []


def test_replaces_a_local_file_with_the_wrong_hash(release):
    models_dir = resolve_path(release, "models_dir")
    models_dir.mkdir(parents=True)
    (models_dir / "a.pt").write_bytes(b"stale")
    assert [p.name for p in download_checkpoints(release)] == ["a.pt", "b.pt"]
    assert sha256(models_dir / "a.pt") == release["release"]["checkpoints"]["a.pt"]


def test_rejects_a_download_that_does_not_match_the_release(release):
    release["release"]["checkpoints"]["b.pt"] = "0" * 64
    with pytest.raises(ValueError, match="b.pt"):
        download_checkpoints(release)
    models_dir = resolve_path(release, "models_dir")
    assert not (models_dir / "b.pt").exists() and not (models_dir / "b.part").exists()
