"""
Tiny helper to load configuration files in YAML, JSON or TOML.

Usage:
    cfg = load_config('configs/base_local.yml')
"""
from pathlib import Path
from typing import Dict, Union
import json
import tomli as tomllib  # type: ignore
import yaml

def _suffix(path: Union[str, Path]) -> str:
    return Path(path).suffix.lower()

def load_config(path: Union[str, Path]) -> Dict:
    """Return the configuration dictionary stored in *path*."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    
    ext = _suffix(path)
    if ext in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if ext in {".toml", ".tml"}:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    raise ValueError(f"Unsupported config format: {ext}")
