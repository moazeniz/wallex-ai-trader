# Wallex trading engine - developer and operator shortcuts.
# Every target runs from the repository root. Override variables on the command line, e.g.
#   make run-arena STATE=/tmp/arena ANTHROPIC_ENV=~/.secrets/anthropic.env
SHELL        := /bin/bash
PYTHON       ?= python3
VENV         ?= .venv
PY           := $(VENV)/bin/python
ARENA_CONFIG ?= experiments/wallex_arena/arena-config.json
STATE        ?= ./arena-state
AGENT_STATE  ?= ./.wallex-agent
ANTHROPIC_ENV ?= ./anthropic.env
MODEL        ?= artifacts/wallex_agent/quant-universe-v2/model.json
BOOKS_DB     ?= ./books.sqlite
IMAGE        ?= meowmeow-wallex:local
LIARA_APP    ?= meowmeow-wallex
DEPLOY_LIARA := $(PY) deploy/liara-wallex/deploy_liara.py
UNITS        := wallex-arena wallex-paper-agent wallex-book-recorder wallex-dashboard-web wallex-live-watchdog wallex-dashboard-export.timer

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev test test-arena lint check run-arena arena-status arena-report run-agent agent-status \
        run-recorder dashboard-export systemd-render systemd-install deploy status logs kill-switch clean \
        docker-build docker-run liara-deploy liara-deploy-api liara-releases liara-rollback liara-logs

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip

venv: $(PY) ## Create .venv with runtime dependencies
	$(PY) -m pip install -q -r requirements.txt

install: venv ## Alias of venv (runtime dependencies only)

install-dev: $(PY) ## Create .venv with runtime + test/lint dependencies
	$(PY) -m pip install -q -r requirements-dev.txt

test: ## Run the full offline test suite (no network, no keys)
	OMP_NUM_THREADS=2 $(PY) -m pytest -q

test-arena: ## Run only the arena / BoxAPI / live-client / watchdog tests (fast)
	$(PY) -m pytest -q tests/test_wallex_arena*.py

lint: ## Correctness lint (syntax errors, undefined names) on the Wallex code
	$(VENV)/bin/ruff check experiments services tests

check: lint test ## Lint + tests (what CI runs)

run-arena: ## Run the arena in the foreground (paper; needs ANTHROPIC_ENV for AI accounts)
	$(PY) -m experiments.wallex_arena run $(ARENA_CONFIG) $(STATE) $(ANTHROPIC_ENV)

arena-status: ## Print the arena leaderboard from STATE
	$(PY) -m experiments.wallex_arena status $(STATE)

arena-report: ## Write a Markdown shadow-test report from STATE to report.md
	$(PY) -m experiments.wallex_arena report $(STATE) report.md

run-agent: ## One bounded paper-agent research cycle (needs WALLEX_ANTHROPIC_SECRET_FILE for the council)
	$(PY) -m experiments.wallex_agent run --model $(MODEL) --state $(AGENT_STATE) --artifacts $(AGENT_STATE)/artifacts --cycles 1

agent-status: ## Show the paper agent's last completed cycle
	$(PY) -m experiments.wallex_agent status --state $(AGENT_STATE)

run-recorder: ## Record public Wallex/Binance books into BOOKS_DB (Ctrl-C to stop)
	$(PY) services/book_recorder.py $(BOOKS_DB) 12

dashboard-export: ## Export dashboard JSON from STATE into ./build/dashboard/data
	mkdir -p build/dashboard/data && cp services/dashboard/index.html build/dashboard/
	$(PY) services/dashboard_export.py build/dashboard/data $(STATE)

systemd-render: ## Render systemd unit templates into ./build/systemd for review
	deploy/systemd/install.sh render

systemd-install: ## (root) Render + install units, create service user and dirs; enables nothing
	sudo --preserve-env=APP_DIR,VENV,SERVICE_USER,SECRETS_DIR,ARENA_CONFIG,WEB_DIR,DASHBOARD_PORT,TLS_CERT,TLS_KEY deploy/systemd/install.sh install

deploy: venv systemd-install ## (root) Update deps + units, then restart the units that are currently active
	@for u in $(UNITS); do if systemctl is-active --quiet $$u; then echo "restart $$u"; sudo systemctl restart $$u; fi; done

status: ## Show the state of all wallex-* units
	@systemctl list-units 'wallex-*' --all --no-pager

logs: ## Follow the logs of all wallex-* units
	journalctl -f -u 'wallex-*'

kill-switch: ## EMERGENCY: block every live order immediately (creates the LIVE_DISABLED file)
	sudo touch /var/lib/wallex-arena/LIVE_DISABLED && ls -l /var/lib/wallex-arena/LIVE_DISABLED

docker-build: ## Build the Liara image from the repo root (runs the arena checks inside)
	docker build -t $(IMAGE) .

docker-run: ## Run the image locally on :8080 (dashboard only; engine OFF, DISARMED). Needs WALLEX_ADMIN_PASSWORD
	@test -n "$$WALLEX_ADMIN_PASSWORD" || { echo "export WALLEX_ADMIN_PASSWORD first"; exit 1; }
	docker run --rm -p 127.0.0.1:8080:8080 -e WALLEX_ADMIN_PASSWORD $(IMAGE)

liara-deploy: ## Deploy with the Liara CLI using ./liara.json (code only; envs are kept)
	liara deploy --app $(LIARA_APP)

liara-deploy-api: ## Build with a fresh results snapshot and release through the Liara API (build host only)
	$(DEPLOY_LIARA) build && $(DEPLOY_LIARA) deploy

liara-releases: ## List recent Liara releases
	$(DEPLOY_LIARA) releases

liara-rollback: ## Roll back to RELEASE=<id> (POST /v1/releases/{id}/rollback)
	@test -n "$(RELEASE)" || { echo "usage: make liara-rollback RELEASE=<release id>"; exit 1; }
	$(DEPLOY_LIARA) rollback $(RELEASE)

liara-logs: ## Follow the Liara app logs (Liara CLI)
	liara logs --app $(LIARA_APP) --follow

clean: ## Remove caches and rendered files (keeps .venv and runtime state)
	find . -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache build
