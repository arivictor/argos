ruff-format: ## Runs ruff formatter on the codebase
	@ruff format .

ruff-lint:  ## Runs ruff linter on the codebase
	@ruff check --fix  .

ruff-check: ## Runs ruff linter on the codebase without fixing
	@ruff check .
	@ruff format --check .

format: ruff-format ruff-lint ## Formatting and linting using Ruff

.PHONY: help
.DEFAULT_GOAL := help

help:
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# catch-all for any undefined targets - this prevents error messages
# when running things like make npm-install <package>
%:
	@:
