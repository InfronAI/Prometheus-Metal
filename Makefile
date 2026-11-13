.PHONY: setup doctor test audit demo validate report
PYTHON ?= .venv/bin/python

setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install -e '.[mac,dev]'

doctor:
	$(PYTHON) -m prometheus_eval doctor

test:
	$(PYTHON) -m pytest -q
	.venv/bin/ruff check src tests scripts

audit:
	$(PYTHON) -m prometheus_eval audit-data

demo:
	$(PYTHON) -m prometheus_eval serve

validate:
	$(PYTHON) scripts/reproduce.py --profile validation

report:
	$(PYTHON) -m prometheus_eval analyze --suite experiments/mac-validation
