from pathlib import Path

import pytest

from models import MODELS
from projects.train_module import TrainingModule
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
        root / "configs" / "models" / "unet" / "UNet.yaml",
        root / "configs" / "training" / "base.yaml",
    )

    assert config["data"]["sr"] == 32000
    assert config["model"]["model_name"] == "UNet"
    assert config["training"]["scheduler"]["min_lr"] == pytest.approx(1e-6)


def test_tfgridnet_config_loads_and_model_is_registered():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz.yaml",
        root / "configs" / "models" / "tfgridnet" / "TFGridNet.yaml",
        root / "configs" / "training" / "base_mapping.yaml",
    )

    assert config["model"]["model_name"] == "TFGridNet"
    assert config["model"]["emb_hs"] == 1
    assert config["model"]["model_name"] in MODELS
    assert config["training"]["batch_size"] == 16
    assert config["training"]["precision"] == 32
    assert config["training"]["masking"] is None
    assert "accumulate_grad_batches" not in config["training"]


def test_mamba_config_loads_and_model_is_registered():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz.yaml",
        root / "configs" / "models" / "mamba" / "Mamba.yaml",
        root / "configs" / "training" / "base_mapping.yaml",
    )

    assert config["model"]["model_name"] == "Mamba"
    assert config["model"]["dt_rank"] == "auto"
    assert config["model"]["eps"] == pytest.approx(1e-5)
    assert config["model"]["model_name"] in MODELS
    assert config["training"]["masking"] is None


def test_spatialnet_config_loads_and_model_is_registered():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz.yaml",
        root / "configs" / "models" / "spatialnet" / "SpatialNet.yaml",
        root / "configs" / "training" / "base_mapping.yaml",
    )

    assert config["model"]["model_name"] == "SpatialNet"
    assert config["model"]["share_full_band"] is True
    assert config["model"]["dim_squeeze"] == 8
    assert config["model"]["eps"] == pytest.approx(1e-5)
    assert config["model"]["model_name"] in MODELS
    assert config["training"]["masking"] is None


def test_conformer_foa_prior_experiment_configs_are_consistent():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz_foa_prior.yaml",
        root / "configs" / "models" / "conformer" / "Conformer_foa_prior.yaml",
        root / "configs" / "training" / "base_prior_residual.yaml",
    )

    prior = config["data"]["features"]["atf_foa_prior"]

    assert prior["enabled"] is True
    assert prior["geometry_source"] == "actual"
    assert prior["reg_eps"] == pytest.approx(1e-3)
    assert prior["max_wng_db"] == pytest.approx(20.0)

    assert config["training"]["prior_combine"] == "residual"
    assert config["training"]["masking"] is None

    # 4 microphones plus the 4 prior channels, real and imaginary.
    assert config["model"]["in_channels"] == 16
    assert config["model"]["model_name"] in MODELS

    TrainingModule(config)


def test_unet_foa_prior_experiment_configs_are_consistent():
    root = Path(__file__).parents[1]

    config = load_and_merge_configs(
        root / "configs" / "data" / "32khz_foa_prior.yaml",
        root / "configs" / "models" / "unet" / "UNet_foa_prior.yaml",
        root / "configs" / "training" / "base_prior_residual.yaml",
    )

    # UNet takes its input width from the first channel pair.
    assert config["model"]["channels"][0][0] == 16
    assert config["model"]["output_init_std"] == pytest.approx(1e-3)
    assert config["training"]["prior_combine"] == "residual"

    module = TrainingModule(config)
    assert module.model.in_channels == 16
