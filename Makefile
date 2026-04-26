# ─────────────────────────────────────────────────────────────────────────────
# AML Transaction Monitoring - Makefile
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install download-data rules features graph train alerts triage report \
        full-pipeline evaluate dashboard test lint clean help

PYTHON ?= python
PIP ?= pip
STREAMLIT ?= streamlit

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

install: ## Install all Python dependencies
	$(PIP) install -r requirements.txt

download-data: ## Create data directory and print download instructions
	@mkdir -p data
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Download the IBM AML dataset from Kaggle:"
	@echo "  https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
	@echo ""
	@echo "Place the CSV files in the data/ directory:"
	@echo "  data/HI-Small_Trans.csv"
	@echo "  data/HI-Medium_Trans.csv"
	@echo "  data/HI-Large_Trans.csv"
	@echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Individual Pipeline Stages
# ─────────────────────────────────────────────────────────────────────────────

rules: ## Run rule engine on transaction data
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine']); \
"

features: ## Generate ML features
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine', 'feature_engineering']); \
"

graph: ## Run graph analysis
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine', 'graph_analysis']); \
"

train: ## Train ML models
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine', 'feature_engineering', 'graph_analysis', 'ml_scoring']); \
"

alerts: ## Generate alerts
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine', 'feature_engineering', 'graph_analysis', 'ml_scoring', 'alert_generation']); \
"

triage: ## Run LLM alert triage
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine', 'feature_engineering', 'graph_analysis', 'ml_scoring', 'alert_generation', 'llm_triage']); \
"

report: ## Generate compliance reports
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
pipeline.run(stages=['data_loading', 'rule_engine', 'feature_engineering', 'graph_analysis', 'ml_scoring', 'alert_generation', 'compliance_reporting'], skip_llm=True); \
"

# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

full-pipeline: ## Run the complete end-to-end AML pipeline
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
results = pipeline.run(data_variant='small'); \
pipeline.evaluate(); \
"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

evaluate: ## Run evaluation on existing pipeline results
	$(PYTHON) -c "\
from src.config import Config; \
from src.pipeline import AMLPipeline; \
config = Config(); \
pipeline = AMLPipeline(config); \
results = pipeline.run(data_variant='small', skip_llm=True); \
eval_results = pipeline.evaluate(); \
print(pipeline._results.get('evaluation', {}).get('system_comparison', 'No comparison available')); \
"

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

dashboard: ## Launch the Streamlit compliance dashboard
	$(STREAMLIT) run app/dashboard.py

# ─────────────────────────────────────────────────────────────────────────────
# Testing & Quality
# ─────────────────────────────────────────────────────────────────────────────

test: ## Run all unit tests with pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run code linting (black, flake8)
	@echo "Running black (check mode)..."
	-$(PYTHON) -m black --check src/ app/ tests/
	@echo "Running flake8..."
	-$(PYTHON) -m flake8 src/ app/ tests/ --max-line-length=120 --ignore=E501,W503

format: ## Auto-format code with black
	$(PYTHON) -m black src/ app/ tests/

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

clean: ## Remove generated files, caches, and build artifacts
	rm -rf __pycache__ src/__pycache__ app/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	rm -rf outputs/intermediates/
	rm -rf *.egg-info dist build
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean ## Remove all generated files including models and reports
	rm -rf models/ mlruns/ outputs/ alerts/ reports/
	@echo "Cleaned all generated artifacts."

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo "AML Transaction Monitoring - Available Commands:"
	@echo "═══════════════════════════════════════════════════════════════════"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo "═══════════════════════════════════════════════════════════════════"

.DEFAULT_GOAL := help
