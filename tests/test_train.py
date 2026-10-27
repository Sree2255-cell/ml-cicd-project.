import json

import joblib

from src import config


def test_train_creates_artifacts(trained_model):
    assert config.MODEL_PATH.exists()
    assert config.METRICS_PATH.exists()


def test_metrics_are_valid(trained_model):
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= trained_model[key] <= 1.0
    assert trained_model["n_features"] == 30


def test_metrics_file_matches_returned_metrics(trained_model):
    on_disk = json.loads(config.METRICS_PATH.read_text())
    assert on_disk["accuracy"] == trained_model["accuracy"]


def test_model_meets_quality_bar(trained_model):
    assert trained_model["accuracy"] >= config.MIN_ACCURACY


def test_saved_model_predicts(trained_model):
    model = joblib.load(config.MODEL_PATH)
    sample = [[0.0] * 30]
    assert model.predict(sample).shape == (1,)
