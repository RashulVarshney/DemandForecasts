"""Config loading helper shared across the pipeline."""
from __future__ import annotations

import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(path: str | pathlib.Path = "configs/config.yaml") -> dict:
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)
