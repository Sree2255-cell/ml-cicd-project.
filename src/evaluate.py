"""Quality gate. Exits non-zero if the trained model does not meet the thresholds.

CI/CD runs this right after training, so a bad model can never be built into an
image or deployed.

Usage:
    python -m src.evaluate
"""
import json
import sys

from src import config


def check(metrics: dict, min_accuracy: float, min_f1: float) -> list[str]:
    """Return a list of failure messages (empty list means the gate passed)."""
    failures = []
    if metrics["accuracy"] < min_accuracy:
        failures.append(f"accuracy {metrics['accuracy']} < required {min_accuracy}")
    if metrics["f1"] < min_f1:
        failures.append(f"f1 {metrics['f1']} < required {min_f1}")
    return failures


def main() -> int:
    if not config.METRICS_PATH.exists():
        print(f"FAIL: {config.METRICS_PATH} not found. Run `python -m src.train` first.")
        return 2

    metrics = json.loads(config.METRICS_PATH.read_text())
    failures = check(metrics, config.MIN_ACCURACY, config.MIN_F1)

    if failures:
        print("QUALITY GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"QUALITY GATE PASSED: accuracy={metrics['accuracy']} (>= {config.MIN_ACCURACY}), "
        f"f1={metrics['f1']} (>= {config.MIN_F1})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
