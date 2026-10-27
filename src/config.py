"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Quality gate: the pipeline fails if the trained model scores below these.
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.93"))
MIN_F1 = float(os.getenv("MIN_F1", "0.93"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
