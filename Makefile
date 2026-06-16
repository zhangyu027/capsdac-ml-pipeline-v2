PYTHONPATH := $(PWD)/forecasting

.PHONY: install test run report api clean validate

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH) pytest

run:
	PYTHONPATH=$(PYTHONPATH) python forecasting/scripts/run_capsdac_pipeline.py

report:
	PYTHONPATH=$(PYTHONPATH) python forecasting/scripts/generate_visualization_report.py

validate: test run report

api:
	uvicorn serving.api.main:app --reload

clean:
	rm -f data/processed/*.csv outputs/metrics/*.json outputs/reports/*.json outputs/forecasts/*.csv
