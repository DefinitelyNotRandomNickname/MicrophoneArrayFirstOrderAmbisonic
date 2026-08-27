from pathlib import Path

import pytest

from models import MODELS
from utils.configs import deep_merge, load_and_merge_configs


def test_deep_merge_overrides_nested_values_without_mutating_sources():
    base = {"data": {"sr": 16000, "normalize": True}, "tags": ["base"]}
    override = {"data": {"sr": 32000}, "tags": ["override"]}

    merged = deep_merge(base, override)

    assert merged == {"data": {"sr": 32000, "normalize": True}, "tags": ["override"]}
    assert base["data"]["sr"] == 16000
    assert override["data"]["sr"] == 32000


def test_project_configs_load_and_parse_scientific_notation():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz.yaml",
        root / "configs" / "models" / "UNet.yaml",
        root / "configs" / "training" / "base.yaml",
    )

    assert config["data"]["sr"] == 32000
    assert config["model"]["model_name"] == "UNet"
    assert config["training"]["scheduler"]["min_lr"] == pytest.approx(1e-6)


def test_tfgridnet_config_loads_and_model_is_registered():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz.yaml",
        root / "configs" / "models" / "TFGridNet.yaml",
        root / "configs" / "training" / "base_mapping.yaml",
    )

    assert config["model"]["model_name"] == "TFGridNet"
    assert config["model"]["emb_hs"] == 1
    assert config["model"]["model_name"] in MODELS
    assert config["training"]["batch_size"] == 16
    assert config["training"]["precision"] == 32
    assert config["training"]["masking"] is None
    assert "accumulate_grad_batches" not in config["training"]
