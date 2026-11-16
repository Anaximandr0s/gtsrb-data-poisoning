# Demo day: `make run`. Other targets: install, export, test, check.
PYTHON ?= .venv/bin/python
PORT ?= 8502

.PHONY: run install export test check

run: check results/metrics/tsne_demo.npz
	$(PYTHON) -m streamlit run app/main.py --server.address localhost --server.port $(PORT) --server.headless true

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

results/metrics/tsne_demo.npz:
	$(PYTHON) -m src.demo_charts

export:
	$(PYTHON) -m src.demo_charts

test:
	$(PYTHON) -m pytest -q

check:
	@$(PYTHON) -m src.demo
