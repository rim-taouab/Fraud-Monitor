PY := .venv/bin/python

.PHONY: run test chart baseline

run: baseline

baseline:
	$(PY) src/run_baseline.py

test:
	$(PY) src/test_baseline.py

chart:
	$(PY) src/make_chart.py
