"""Loads assumptions.yaml: the single place for every placeholder number."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_assumptions(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "assumptions.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
