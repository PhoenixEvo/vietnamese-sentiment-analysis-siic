# Makefile for SIIC - Vietnamese Emotion Detection System

.PHONY: help install install-dev install-gpu test clean lint format dashboard train-all evaluate-all setup-dev

# Default target
help:
	@echo "SIIC - Vietnamese Emotion Detection System"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install package in current environment"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  install-gpu  - Install with GPU support"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  dashboard    - Launch dashboard"
	@echo "  train-all    - Train all models"
	@echo "  evaluate-all - Evaluate all models"
	@echo "  setup-dev    - Setup development environment"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

# Development targets
setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

test:
	python -m pytest tests/ -v

lint:
	flake8 siic/ scripts/ tests/
	mypy siic/

format:
	black siic/ scripts/ tests/
	@echo "Code formatted with black"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Application targets
dashboard:
	python scripts/dashboard.py

dashboard-dev:
	python scripts/dashboard.py --dev

# Training targets
train-phobert:
	python scripts/train.py --model phobert --epochs 3 --batch_size 8

train-lstm:
	python scripts/train.py --model lstm --epochs 15 --batch_size 32

train-baselines:
	python scripts/train.py --model baselines

train-all: train-baselines train-lstm train-phobert
	@echo "All models trained successfully!"

# Evaluation targets
evaluate-comprehensive:
	python scripts/evaluate.py --comprehensive

evaluate-report:
	python scripts/evaluate.py --generate-report

evaluate-all: evaluate-comprehensive evaluate-report
	@echo "Evaluation completed!"

# Data targets
prepare-data:
	python -c "from siic.data.loaders import main; main()"

# Docker targets
docker-build:
	docker build -t siic:latest .

docker-run:
	docker run -p 8501:8501 siic:latest

# Package targets
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

# Documentation targets
docs:
	sphinx-build -b html docs/ docs/_build/

# Quick commands
quick-test: lint test
	@echo "Quick validation complete!"

quick-setup: clean install-dev test
	@echo "Quick setup complete!"

# Show project structure
structure:
	tree -I 'venv|__pycache__|*.pyc|.git|*.egg-info|build|dist' -L 3 