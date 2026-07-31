"""Tests for the configuration loader (config/__init__.py)."""

import os
import json
import tempfile
import pytest
from pathlib import Path
from config import load_config


def _write_yaml(path: str, data: dict):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


class TestLoadConfigDefaults:
    def test_load_default_yaml(self):
        cfg = load_config()
        assert "pipeline" in cfg
        assert cfg["pipeline"]["extractors"] == ["rbe", "sbe", "rbe_rand"]
        assert cfg["pipeline"]["cot_samples"] == 100
        assert cfg["pipeline"]["seed"] == 42

    def test_gcp_defaults(self):
        cfg = load_config()
        gcp = cfg["gcp"]
        assert gcp["n_bootstrap"] == 2000
        assert gcp["pass_mean"] == 0.80
        assert gcp["pass_min"] == 0.60

    def test_iaa_defaults(self):
        cfg = load_config()
        assert cfg["iaa"]["n_samples"] == 100

    def test_power_analysis_defaults(self):
        cfg = load_config()
        pa = cfg["power_analysis"]
        assert pa["target_auc_improvement"] == 0.05
        assert pa["alpha"] == 0.05
        assert pa["power"] == 0.80


class TestLoadConfigFromPath:
    def test_load_from_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.yaml"
            _write_yaml(path, {"pipeline": {"extractors": ["llm"], "cot_samples": 50}})
            cfg = load_config(str(path))
            assert cfg["pipeline"]["extractors"] == ["llm"]
            assert cfg["pipeline"]["cot_samples"] == 50

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(r"C:\non_existent\config.yaml")

    def test_empty_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            cfg = load_config(str(path))
            assert cfg == {}


class TestLoadConfigEnvOverrides:
    RTSA_CONFIG_VAR = "RTSA_CONFIG"
    RTSA_EXTRACTORS_VAR = "RTSA_EXTRACTORS"

    def test_env_var_path_override(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.yaml"
            _write_yaml(path, {"pipeline": {"cot_samples": 999}})
            monkeypatch.setenv(self.RTSA_CONFIG_VAR, str(path))
            cfg = load_config()
            assert cfg["pipeline"]["cot_samples"] == 999

    def test_env_extractors_override(self, monkeypatch):
        monkeypatch.setenv(self.RTSA_EXTRACTORS_VAR, "llm_e4,llm_e5")
        cfg = load_config()
        assert cfg["pipeline"]["extractors"] == ["llm_e4", "llm_e5"]

    def test_env_extractors_override_single(self, monkeypatch):
        monkeypatch.setenv(self.RTSA_EXTRACTORS_VAR, "gpt4")
        cfg = load_config()
        assert cfg["pipeline"]["extractors"] == ["gpt4"]

    def test_env_extractors_empty_does_not_override(self, monkeypatch):
        monkeypatch.setenv(self.RTSA_EXTRACTORS_VAR, "")
        cfg = load_config()
        assert cfg["pipeline"]["extractors"] == ["rbe", "sbe", "rbe_rand"]
