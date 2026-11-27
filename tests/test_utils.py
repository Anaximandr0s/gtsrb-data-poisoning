from src.utils import load_json, save_env_info


def test_env_info_records_versions_and_device(tmp_path):
    path = tmp_path / "env" / "info.json"
    info = save_env_info(path)
    assert load_json(path) == info
    assert {"torch", "numpy"} <= info["packages"].keys() and info["device"] in {"cpu", "cuda", "mps"}
