"""Tiny config loader shared by the DVC pipeline stages."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG_PATH = "config/config.yaml"

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)