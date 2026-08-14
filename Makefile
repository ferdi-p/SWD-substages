PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MPLCONFIGDIR ?= .matplotlib-cache
export MPLCONFIGDIR
export PYTHONPATH := $(CURDIR)/src

PROCESSED_DATA = \
	data/processed/baser/development.csv \
	data/processed/baser/adult_survival.csv \
	data/processed/baser/fertility.csv

MODEL_PUBLICATION_FIGURE_DIR = outputs/plots/publication
MODEL_SUPPLEMENTARY_FIGURE_DIR = $(MODEL_PUBLICATION_FIGURE_DIR)/supplementary
MODEL_FIGURE_ARGS = \
	--publication-figure-dir $(MODEL_PUBLICATION_FIGURE_DIR) \
	--supplementary-figure-dir $(MODEL_SUPPLEMENTARY_FIGURE_DIR) \
	--report outputs/reports/model_complexity.md

.PHONY: all setup processed direct model-comparison \
	model-intervention model-appendix model test verify help

all: verify

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-lock.txt -e .

processed: $(PROCESSED_DATA)

direct: processed
	"$(PYTHON)" scripts/calculate_baser_direct.py

model-comparison: processed
	"$(PYTHON)" scripts/compare_model_complexity.py $(MODEL_FIGURE_ARGS)

model-intervention: model-comparison
	"$(PYTHON)" scripts/analyze_mortality_interventions.py \
		--publication-figure-dir $(MODEL_PUBLICATION_FIGURE_DIR) \
		--report outputs/reports/mortality_interventions.md

model-appendix: model-comparison
	"$(PYTHON)" scripts/plot_m3_fecundity_profile.py \
		--output $(MODEL_PUBLICATION_FIGURE_DIR)/m3_adult_fecundity_profile.pdf

model: direct model-intervention model-appendix

test:
	"$(PYTHON)" -m pytest

verify: test model
	"$(PYTHON)" scripts/verify_paper_outputs.py --model-only

help:
	@echo "make setup       Create the exact tested Python environment"
	@echo "make model       Rebuild all analyses, tables, and figures"
	@echo "make test        Run the automated tests"
	@echo "make verify      Test and verify the complete model workflow"

$(PROCESSED_DATA):
	@echo "Processed data are missing; restore the version-controlled files under data/processed/baser/."
	@false
