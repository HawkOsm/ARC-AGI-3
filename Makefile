# ARC Prize 2026 — ARC-AGI-3 local dev workflow.
#
# Five commands cover the whole loop:
#   make setup        # one-time: venv + arc-agi + clone framework
#   make play-local   # fast inner loop: run agent/my_agent.py on a real game
#   make pull-sample  # fetch the official Stochastic Goose sample for reference
#   make submit       # build notebook from agent/my_agent.py + push to Kaggle
#   make status       # tail the latest Kaggle run

PYTHON          ?= 3.12
VENV            := .venv
VENV_PY         := $(VENV)/bin/python
# Local change vs upstream starter: this machine has no system python3.12, so
# the venv is uv-managed (uv fetches CPython 3.12 itself).
VENV_PIP        := uv pip install --python $(VENV_PY)
# Token: project-local .kaggle/access_token if present, else ~/.kaggle/access_token
# (local change vs upstream, which only reads the project-local file). Exposed
# as KAGGLE_API_TOKEN, the only env var the modern Kaggle CLI honours.
KAGGLE_TOKEN_FILE := $(firstword $(wildcard .kaggle/access_token) $(wildcard $(HOME)/.kaggle/access_token))
KAGGLE          := KAGGLE_API_TOKEN=$$(cat $(KAGGLE_TOKEN_FILE)) $(VENV)/bin/kaggle
FRAMEWORK_REPO  := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR   := vendor/ARC-AGI-3-Agents
# Pinned to what Kaggle's competition dataset ships (ARC-AGI-3-Agents/ dated
# 2026-04-17 -> last upstream commit before that is 135f20a). Local == Kaggle.
FRAMEWORK_REF   := 135f20a
# arc-agi/arcengine from the dataset's arc_agi_3_wheels/; numpy = what the Kaggle image already
# has (2.0.2 — pip keeps it, the 2.4.4 wheel is never installed; seen in kernel v2 log).
ARC_AGI_PIN     := arc-agi==0.9.8 arcengine==0.9.3 numpy==2.0.2
COMP_SLUG       := arc-prize-2026-arc-agi-3
GAME            ?=
STEPS           ?= 200
AGENT           ?= agent/my_agent.py
REPS            ?= 30
BUDGET          ?= 2000

.PHONY: help setup play-local pull-sample notebook submit status verify-local clean _check-kaggle \
        build check test interface maskcheck eval eval-paired eval-swarm ci llm-local-setup llm-local eval-llm dashboard debug

_check-kaggle:
	@if [ -z "$(KAGGLE_TOKEN_FILE)" ] || [ ! -s "$(KAGGLE_TOKEN_FILE)" ]; then \
	    echo "ERROR: no Kaggle token at .kaggle/access_token or ~/.kaggle/access_token."; \
	    echo "       Generate a token at https://www.kaggle.com/settings (API → Create New Token)"; \
	    echo "       and save it as a one-line file at: $(PWD)/.kaggle/access_token"; \
	    exit 1; \
	fi

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Vars: PYTHON=$(PYTHON)  GAME=$(GAME)  STEPS=$(STEPS)"

setup: ## One-time install: venv, arc-agi, kaggle CLI, clone framework
	@[ -x $(VENV_PY) ] || uv venv --python $(PYTHON) $(VENV)
	$(VENV_PIP) $(ARC_AGI_PIN) "kaggle>=2.2" python-dotenv pandas pyarrow ruff mypy pytest
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then \
	    mkdir -p vendor && git clone $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); \
	fi
	git -C $(FRAMEWORK_DIR) checkout -q -- . && git -C $(FRAMEWORK_DIR) fetch -q origin && \
	    git -C $(FRAMEWORK_DIR) checkout -q $(FRAMEWORK_REF)
	@# Slim agents/__init__.py so we don't need langgraph/langsmith/smolagents/etc.
	@# (Same trick the official Stochastic Goose sample uses on Kaggle.)
	@$(VENV_PY) scripts/slim_framework.py
	@echo ""
	@echo "Setup complete. Try:  make play-local"

play-local: ## Run agent/my_agent.py against ALL games (or GAME=ls20 for a single one)
	$(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify-local: ## Quick smoke test: 50 steps on ls20 + vc33 only
	$(VENV_PY) scripts/play_local.py --game ls20,vc33 --max-steps 50

list-games: ## Show all available games
	$(VENV_PY) scripts/play_local.py --list

pull-sample: _check-kaggle ## Download the official Stochastic Goose sample notebook for reference
	mkdir -p reference/stochastic-goose
	$(KAGGLE) kernels pull inversion/arc3-sample-submission-stochastic-goose \
	    -p reference/stochastic-goose -m
	@echo "Open reference/stochastic-goose/*.ipynb for the canonical pattern."

notebook: ## Splice agent/my_agent.py (or AGENT=path) into notebooks/submission.ipynb
	ARC3_AGENT_SRC=$(AGENT) $(VENV_PY) scripts/build_notebook.py

submit: notebook _check-kaggle ## Build notebook and push to Kaggle (one-line submission)
	@grep -q REPLACE_WITH_YOUR_USERNAME notebooks/kernel-metadata.json && { \
	    echo "ERROR: edit notebooks/kernel-metadata.json and replace REPLACE_WITH_YOUR_USERNAME"; \
	    exit 1; } || true
	$(KAGGLE) kernels push -p notebooks/
	@echo ""
	@echo "Pushed. Track it with:  make status"

status: _check-kaggle ## Show the status of your most recent Kaggle kernel run
	@KERNEL_ID=$$(python3 -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

clean: ## Remove generated artefacts (venv, downloaded games, vendored repos)
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb \
	       reference logs.log __pycache__ .pytest_cache

# ── Project additions (not in the upstream starter) ─────────────────────────

build: ## Regenerate agent/my_agent.py from agent/brain/
	$(VENV_PY) scripts/build_submission.py

check: ## Lint + type-check agent/brain + verify agent/my_agent.py is freshly built
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy
	$(VENV_PY) scripts/build_submission.py --check

test: ## Unit + end-to-end tests of agent/brain (tests/; no network, no GPU)
	$(VENV_PY) -m pytest tests -q

interface: ## Re-verify the engine/framework facts the agent depends on
	$(VENV_PY) eval/interface_checks.py

eval: build ## N-repeat eval of AGENT on all games (REPS, BUDGET)
	$(VENV_PY) eval/run.py --agent $(AGENT) --reps $(REPS) --max-actions $(BUDGET)

eval-paired: build ## Paired AGENT vs eval/champion/my_agent.py
	$(VENV_PY) eval/run.py --agent $(AGENT) --champion eval/champion/my_agent.py --reps $(REPS) --max-actions $(BUDGET)

eval-swarm: build ## All games as threads in one process (Kaggle's execution model)
	$(VENV_PY) eval/run.py --agent $(AGENT) --swarm --max-actions $(BUDGET)

# ── Local LLM for development (small stand-in for Kaggle's Qwen3.8-27B; see docs/ARCHITECTURE.md) ──
LLM_VENV   := .venv-llm
LLM_MODEL  ?= models/Qwen3.5-4B-AWQ
LLM_PORT   ?= 8000

llm-local-setup: ## One-time: vLLM 0.30.0 (same as Kaggle) in .venv-llm + the 4B dev model in models/
	@[ -x $(LLM_VENV)/bin/python ] || uv venv --python $(PYTHON) $(LLM_VENV)
	uv pip install --python $(LLM_VENV)/bin/python "vllm==0.30.0"
	$(VENV_PY) -c "from huggingface_hub import snapshot_download as d; d('cyankiwi/Qwen3.5-4B-AWQ-4bit', revision='ef85d23beb', local_dir='models/Qwen3.5-4B-AWQ')"

llm-local: ## Serve the dev model on :$(LLM_PORT) (same flags as the Kaggle dev notebook); Ctrl-C stops it
	VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 $(LLM_VENV)/bin/python -m vllm.entrypoints.openai.api_server \
	    --model $(LLM_MODEL) --served-model-name qwen --host 127.0.0.1 --port $(LLM_PORT) \
	    --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.85 \
	    --max-num-seqs 32 --max-num-batched-tokens 4096 \
	    --enable-prefix-caching --reasoning-parser qwen3 \
	    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true, "reasoning_parser": "qwen3"}'

eval-llm: build ## Swarm eval with the advisor on, against `make llm-local` (BUDGET actions per game)
	ARC3_LLM_URL=http://127.0.0.1:$(LLM_PORT)/v1 ARC3_LLM_LOG=.claude/scratch/llm/local/requests.jsonl ARC3_LLM_MAX_CONCURRENT=8 \
	    $(VENV_PY) eval/run.py --agent $(AGENT) --swarm --max-actions $(BUDGET) --label llm_local

debug: ## Play ONE game, printing every decision: make debug GAME=ar25 STEPS=150 [LLM=1 | HYP=1] (needs make llm-local)
	$(VENV_PY) eval/debug_run.py $(or $(GAME),ls20) --steps $(STEPS) $(if $(LLM),--llm,) $(if $(HYP),--hypothesis,)

dashboard: ## Rebuild the run board page from the run files (.claude/scratch/dashboard/arc3-run-board.html)
	$(VENV_PY) eval/dashboard.py

maskcheck: ## HUD-counter masking == known counter tracks (all characterised games)
	$(VENV_PY) eval/mask_check.py

ci: check test verify-local interface maskcheck ## Local superset of CI: lint, types, build, tests, starter smoke, interface, masks, regression
	$(VENV_PY) eval/regression_check.py
