"""Shared pytest fixtures."""

import pytest

from config import load_config
from model.train import train


@pytest.fixture(scope="session")
def trained_model():
    """Ensure a model artifact exists at the default path (fast budget if missing)."""
    cfg = load_config()
    if not cfg.artifact_path.exists():
        cfg.model.time_budget_s = 5
        train(cfg)
    return cfg
