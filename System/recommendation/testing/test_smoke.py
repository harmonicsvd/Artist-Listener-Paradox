import importlib
from pathlib import Path


def test_configs_load():
    base_path = Path(__file__).resolve().parents[2] / "recommendation" / "configs"
    for cfg_name in ["base_local.yml", "base_local_from_rec.yml"]:
        cfg_path = base_path / cfg_name
        assert cfg_path.exists(), f"Config missing: {cfg_path}"


def test_main_imports():
    # Ensure entrypoint imports without executing main
    mod = importlib.import_module("System.recommendation.MainSystem")
    assert hasattr(mod, "main")
