"""Experiment configuration loader.

Loads YAML config from config/default.yaml with overrides via env vars.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment config from YAML file.

    Args:
        path: Path to YAML config file.  Falls back to CONFIG_PATH env var,
              then to config/default.yaml in the rtsa package directory.
    """
    import yaml

    if path is None:
        path = os.environ.get("RTSA_CONFIG", "")

    if not path:
        path = str(Path(__file__).parent / "default.yaml")

    with open(path, encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Simple env-var override for key settings
    if os.environ.get("RTSA_EXTRACTORS"):
        cfg["pipeline"]["extractors"] = os.environ["RTSA_EXTRACTORS"].split(",")

    return cfg
