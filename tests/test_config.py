from src.config import load_config, project_root, resolve_path


def test_base_config_has_expected_sections():
    cfg = load_config()
    assert {"seed", "paths", "data", "backdoor"} <= cfg.keys()
    assert cfg["backdoor"]["source_class"] == 14
    assert cfg["backdoor"]["target_class"] == 5


def test_paths_resolve_from_project_root():
    cfg = load_config()
    assert resolve_path(cfg, "processed_dir") == project_root() / "data" / "processed"
