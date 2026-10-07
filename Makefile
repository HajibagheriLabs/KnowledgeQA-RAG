PYTHON := .venv/bin/python
CONFIG ?= configs/default.yaml

.PHONY: setup test lint format data stats train eval report

setup:
	uv venv --allow-existing --python 3.11 .venv
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

data:
	$(PYTHON) scripts/download_data.py --config $(CONFIG)

stats:
	$(PYTHON) scripts/inspect_corpus.py --config $(CONFIG)

train:
	@echo "nothing to train yet: no implemented stage has learned parameters"

eval:
	$(PYTHON) scripts/run_bm25.py --config $(CONFIG)

report:
	$(PYTHON) scripts/make_report.py --config $(CONFIG)
