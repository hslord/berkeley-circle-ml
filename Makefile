.PHONY: env install etl train mlflow test lint clean

# ── Environment ───────────────────────────────────────────────────────────────
env:
	conda env create -f environment.yml

install:
	pip install -e ".[dev]"

# ── Pipeline ──────────────────────────────────────────────────────────────────
etl:
	python src/run_etl.py

train:
	python src/train.py

predict:
	python src/predict.py $(TIME)

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow:
	mlflow ui --port 5001 --backend-store-uri mlruns

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .pytest_cache
