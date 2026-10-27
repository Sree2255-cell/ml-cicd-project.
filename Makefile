.PHONY: install lint test train gate serve docker-build docker-run all

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest --cov=src --cov-report=term-missing

train:
	python -m src.train

gate:
	python -m src.evaluate

serve:
	uvicorn src.app:app --reload

docker-build: train gate
	docker build -t ml-cicd-demo .

docker-run:
	docker run --rm -p 8000:8000 ml-cicd-demo

# Same sequence the CI pipeline runs
all: lint test train gate
