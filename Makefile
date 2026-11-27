# `make run` starts the dashboard. Other targets: install, models, prepare, test, lint.
PYTHON ?= .venv/bin/python
PORT ?= 8501

.PHONY: run install models prepare test lint

run: prepare
	$(PYTHON) -m streamlit run app/main.py --server.address localhost --server.port $(PORT) --server.headless true

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

models:
	$(PYTHON) -m src.download

prepare:
	$(PYTHON) -m src.dashboard

test:
	$(PYTHON) -m pytest -q --cov --cov-report=term

lint:
	$(PYTHON) -m ruff check .
