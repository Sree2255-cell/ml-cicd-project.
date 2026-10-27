"""Train the model and write the model artifact plus metrics.

Usage:
    python -m src.train
"""
import json
import sys
from datetime import datetime, timezone

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train() -> dict:
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=data.target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": int(x_train.shape[1]),
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    metrics = train()
    summary = {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")}
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Metrics: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
