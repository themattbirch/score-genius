# Makefile
# --- Generic helpers ---------------------------------------------------------
.PHONY: help lint test ablate ci

PYTHON ?= python

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS=":.*?## "}; {printf " \033[36m%-20s\033[0m %s\n", $$1, $$2}'

lint:  ## Run Ruff diff-only lint (same as workflow)
	$(PYTHON) -m ruff check $$(git diff --name-only --diff-filter=AM master -- '*.py')

test:  ## Run smoke tests locally
	cd backend && pytest -q tests_smoke

ablate:
	@mkdir -p reports
	set -a && source backend/.env && set +a && \
	PYTHONPATH=backend:$(PYTHONPATH) $(PYTHON) -m backend.scripts.run_ablation \
		--sample 5000 --folds 3 \
		--out reports/ablation_results.json

ci: lint test ablate  ## Run full CI suite locally