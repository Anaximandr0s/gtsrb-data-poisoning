# Demo day: `make run`. Other targets: install, export, test, lint, check.
PYTHON ?= .venv/bin/python
PORT ?= 8502

.PHONY: run install export test lint check

run: check results/metrics/tsne_demo.npz
	$(PYTHON) -m streamlit run app/main.py --server.address localhost --server.port $(PORT) --server.headless true

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

results/metrics/tsne_demo.npz:
	$(PYTHON) -m src.demo_charts

export:
	$(PYTHON) -m src.demo_charts

test:
	$(PYTHON) -m pytest -q --cov --cov-report=term

lint:
	$(PYTHON) -m ruff check .

check:
	@$(PYTHON) -m src.demo
