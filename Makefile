PY := .venv/bin/python

.PHONY: run test chart baseline phase2 chart2

run: baseline

baseline:
	$(PY) src/run_baseline.py

phase2:
	$(PY) src/run_phase2.py

test:
	$(PY) src/test_baseline.py
	$(PY) src/test_features.py

chart:
	$(PY) src/make_chart.py

chart2:
	$(PY) src/make_chart_phase2.py
