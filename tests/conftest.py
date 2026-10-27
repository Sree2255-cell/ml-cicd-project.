import pytest

from src import config, train


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Train once into a temp dir so tests never touch the real models/ folder."""
    model_dir = tmp_path_factory.mktemp("models")
    config.MODEL_DIR = model_dir
    config.MODEL_PATH = model_dir / "model.joblib"
    config.METRICS_PATH = model_dir / "metrics.json"
    metrics = train.train()
    return metrics
