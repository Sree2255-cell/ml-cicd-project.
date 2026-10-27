import json

from src import config, evaluate


def test_gate_passes_for_good_metrics():
    metrics = {"accuracy": 0.97, "f1": 0.97}
    assert evaluate.check(metrics, 0.93, 0.93) == []


def test_gate_fails_for_low_accuracy():
    failures = evaluate.check({"accuracy": 0.80, "f1": 0.97}, 0.93, 0.93)
    assert len(failures) == 1
    assert "accuracy" in failures[0]


def test_gate_reports_every_failure():
    assert len(evaluate.check({"accuracy": 0.5, "f1": 0.5}, 0.93, 0.93)) == 2


def test_main_returns_2_when_metrics_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "missing.json")
    assert evaluate.main() == 2


def test_main_returns_1_when_model_is_bad(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"accuracy": 0.5, "f1": 0.5}))
    monkeypatch.setattr(config, "METRICS_PATH", path)
    assert evaluate.main() == 1


def test_main_returns_0_when_model_is_good(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"accuracy": 0.99, "f1": 0.99}))
    monkeypatch.setattr(config, "METRICS_PATH", path)
    assert evaluate.main() == 0
