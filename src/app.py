"""FastAPI service that serves the trained model.

Run locally:
    uvicorn src.app:app --reload
"""
import json
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if config.MODEL_PATH.exists():
        state["model"] = joblib.load(config.MODEL_PATH)
        state["metrics"] = json.loads(config.METRICS_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="ML CI/CD Demo - Breast Cancer Classifier", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., description="The 30 numeric features, in the order returned by /model-info."
    )


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability: float


def _require_model():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["model"], state["metrics"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    _, metrics = _require_model()
    return {
        "metrics": {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")},
        "n_features": metrics["n_features"],
        "feature_names": metrics["feature_names"],
        "class_names": metrics["class_names"],
        "trained_at": metrics["trained_at"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model, metrics = _require_model()
    expected = metrics["n_features"]
    if len(request.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(request.features)}",
        )

    x = np.array(request.features, dtype=float).reshape(1, -1)
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0][pred])
    return PredictResponse(
        prediction=pred,
        label=metrics["class_names"][pred],
        probability=round(proba, 4),
    )
