.DEFAULT_GOAL := help
PYTHON ?= python3

.PHONY: help install train serve test test-cov lint format docker-build docker-up docker-down clean smoke retrain retrain-dry all

help:
	@echo "AegisNet Management Commands:"
	@echo "  help         Show this help message"
	@echo "  install      Install application and development dependencies"
	@echo "  train        Run model training pipeline (train_all.py)"
	@echo "  serve        Start the FastAPI server locally with auto-reload"
	@echo "  test         Run test suite with pytest"
	@echo "  test-cov     Run test suite with coverage report"
	@echo "  lint         Run code linting and formatting check with ruff"
	@echo "  format       Format code and auto-fix lint issues with ruff"
	@echo "  docker-build Build Docker containers via docker compose"
	@echo "  docker-up    Start Docker services in background"
	@echo "  docker-down  Stop running Docker services"
	@echo "  clean        Remove build artifacts, models, caches, and test logs"
	@echo "  smoke        Run smoke test / direct execution of ai_engine.py"
	@echo "  retrain      Trigger model retraining pipeline (retrain.py)"
	@echo "  retrain-dry  Run model retraining in dry-run mode"
	@echo "  all          Run install, train, and test sequentially"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train_all.py

serve:
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format . && $(PYTHON) -m ruff check --fix .

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf models/*.pkl outputs/*.csv outputs/*.json logs/*.log logs/*.jsonl __pycache__ .pytest_cache .ruff_cache htmlcov

smoke:
	$(PYTHON) ai_engine.py

retrain:
	$(PYTHON) retrain.py

retrain-dry:
	$(PYTHON) retrain.py --dry-run

all: install train test
