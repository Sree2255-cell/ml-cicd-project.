import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.app import app


@pytest.fixture()
def client(trained_model):
    # Using the client as a context manager runs the lifespan (loads the model).
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["n_features"] == 30
    assert len(body["feature_names"]) == 30
    assert body["metrics"]["accuracy"] > 0.9


def test_predict_valid_sample(client):
    data = load_breast_cancer()
    response = client.post("/predict", json={"features": data.data[0].tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_known_samples_are_correct(client):
    data = load_breast_cancer()
    # Sample 0 is malignant (0) and the last sample is benign (1) in this dataset.
    first = client.post("/predict", json={"features": data.data[0].tolist()}).json()
    last = client.post("/predict", json={"features": data.data[-1].tolist()}).json()
    assert first["prediction"] == data.target[0]
    assert last["prediction"] == data.target[-1]


def test_predict_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_predict_bad_payload(client):
    response = client.post("/predict", json={"features": "not a list"})
    assert response.status_code == 422
