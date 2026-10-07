PYTHON := .venv/bin/python
CONFIG ?= configs/default.yaml

.PHONY: setup test lint format data stats train select eval ablation study report

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

select:
	$(PYTHON) scripts/select_models.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/run_retrieval.py --config $(CONFIG)
	$(PYTHON) scripts/run_rerank.py --config $(CONFIG)
	$(PYTHON) scripts/run_e2e.py --config $(CONFIG)

ablation:
	$(PYTHON) scripts/chunking_ablation.py --config $(CONFIG)

study:
	$(PYTHON) scripts/index_study.py --config $(CONFIG)

report:
	$(PYTHON) scripts/make_report.py --config $(CONFIG)
