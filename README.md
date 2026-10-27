# ML CI/CD Project

A complete, working CI/CD pipeline for a machine-learning service.

A scikit-learn model (breast cancer classifier) is trained, tested, **quality-gated**, packaged into a Docker image, smoke-tested, and published to GitHub Container Registry by GitHub Actions. The model is served through a FastAPI REST API.

## Pipeline

```
push / PR ──► CI:  lint ─► tests ─► train ─► quality gate ─► upload model artifact
push main ──► CD:  lint ─► tests ─► train ─► quality gate ─► docker build
                   ─► container smoke test ─► push to GHCR ─► deploy hook
```

The quality gate (`src/evaluate.py`) fails the pipeline if accuracy or F1 drops below the thresholds in `src/config.py` (default 0.93), so a degraded model can never be packaged or deployed.

## Layout

```
src/config.py      settings (env-overridable: MIN_ACCURACY, MIN_F1, MODEL_DIR ...)
src/train.py       trains the pipeline, writes models/model.joblib + metrics.json
src/evaluate.py    quality gate, exit code 1 if the model is not good enough
src/app.py         FastAPI service: /health, /model-info, /predict
tests/             unit tests for training, the gate, and the API
Dockerfile         slim, non-root image with a healthcheck
.github/workflows/ ci.yml (PRs and branches), cd.yml (main and tags)
Makefile           local shortcuts that mirror the pipeline
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
make install        # install dependencies
make all            # lint + test + train + quality gate (same as CI)
make serve          # API at http://localhost:8000/docs
```

Try the API:

```bash
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

## Docker

```bash
make docker-build   # trains, gates, then builds the image
make docker-run     # serves on port 8000
```

## Set up the pipeline on GitHub

1. Create a GitHub repository and push this folder to it.
2. Open a pull request: **CI** runs automatically.
3. Merge to `main`: **CD** builds, smoke-tests, and pushes `ghcr.io/<you>/<repo>:latest` and `:sha-xxxxxxx`.
4. Tag a release (`git tag v1.0.0 && git push --tags`) to publish a version-tagged image.
5. Optional: in *Settings → Environments*, create `production` and add required reviewers to approve deployments. Replace the echo in the `deploy` job of `cd.yml` with your real deploy command.

No secrets need to be configured: publishing uses the built-in `GITHUB_TOKEN`.

## Adapting it to your own model

Replace the dataset and estimator in `src/train.py`, adjust the thresholds in `src/config.py`, and update the request schema in `src/app.py`. The pipeline and tests keep working unchanged.
