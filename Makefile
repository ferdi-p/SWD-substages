PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MPLCONFIGDIR ?= .matplotlib-cache
export MPLCONFIGDIR
export PYTHONPATH := $(CURDIR)/src

PROCESSED_DATA = \
	data/processed/baser/development.csv \
	data/processed/baser/adult_survival.csv \
	data/processed/baser/fertility.csv

.PHONY: all model analysis setup preprocess processed direct comparison \
	intervention appendix-figure data-plots test verify-model verify help

all: model

model: direct intervention appendix-figure

analysis: model data-plots

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-lock.txt -e .

preprocess:
	$(PYTHON) scripts/preprocess_baser_data.py

processed: $(PROCESSED_DATA)

direct: processed
	$(PYTHON) scripts/calculate_baser_direct.py

comparison: processed
	$(PYTHON) scripts/compare_model_complexity.py

intervention: comparison
	$(PYTHON) scripts/analyze_mortality_interventions.py

appendix-figure: comparison
	$(PYTHON) scripts/plot_m3_fecundity_profile.py

data-plots: processed
	$(PYTHON) scripts/plot_baser_life_history.py

test:
	$(PYTHON) -m pytest

verify: test analysis
	$(PYTHON) scripts/verify_outputs.py

verify-model: test model
	$(PYTHON) scripts/verify_outputs.py

help:
	@echo "make setup       Create the exact tested Python environment"
	@echo "make model       Rebuild the model analyses, tables, reports, and figures"
	@echo "make verify-model Test, rebuild, and verify the model workflow"
	@echo "make analysis    Model workflow plus raw-data summary plots"
	@echo "make test        Run the unit tests"
	@echo "make verify      Verify the model workflow plus raw-data summary plots"
	@echo "make preprocess  Recreate tracked CSVs from optional source workbooks"

$(PROCESSED_DATA):
	@echo "Processed data are missing; restore data/processed/baser or run make preprocess with the optional source workbooks."
	@false
