PY := .venv/bin/python

.PHONY: run test charts baseline phase2 phase3 phase4

run: baseline

baseline:
	$(PY) src/run_baseline.py

phase2:
	$(PY) src/run_phase2.py

phase3:
	$(PY) src/run_phase3.py

phase4:
	$(PY) src/run_phase4.py

test:
	$(PY) src/test_baseline.py
	$(PY) src/test_features.py
	$(PY) src/test_costs.py
	$(PY) src/test_attacker.py

charts:
	$(PY) src/make_chart.py
	$(PY) src/make_chart_phase2.py
	$(PY) src/make_chart_phase3.py
	$(PY) src/make_chart_phase4.py
