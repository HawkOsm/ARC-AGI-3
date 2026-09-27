# ARC Prize 2026 — ARC-AGI-3. The agent is one file: agent/model.py (an LLM that plays by writing Python).
#
#   make setup                      one-time: venv + arc-agi + dev tools
#   make games                      download the public game sources into environment_files/
#   make llm-local-setup            one-time: vLLM + the local 4B dev model
#   make llm-local                  serve the local model (vision + tools + 32K context, as the Duck needs)
#   make play GAME=ar25 MINUTES=10  play games with agent/model.py against the local model
#   make check test                 lint + types, unit tests

PYTHON        ?= 3.12
VENV          := .venv
VENV_PY       := $(VENV)/bin/python
VENV_PIP      := uv pip install --python $(VENV_PY)
# arc-agi/arcengine as in the competition's arc_agi_3_wheels/; numpy as in the Kaggle image
ARC_AGI_PIN   := arc-agi==0.9.8 arcengine==0.9.3 numpy==2.0.2
GAME          ?= ar25
MINUTES       ?= 10
CONCURRENCY   ?= 1

.PHONY: help setup games play check test llm-local-setup llm-local

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-16s %s\n",$$1,$$2}' $(MAKEFILE_LIST)

setup: ## One-time install: venv, arc-agi, kaggle CLI, dev tools
	@[ -x $(VENV_PY) ] || uv venv --python $(PYTHON) $(VENV)
	$(VENV_PIP) $(ARC_AGI_PIN) "kaggle>=2.2" ruff mypy pytest

games: ## Download the public game sources (NORMAL mode fetches each game once) into environment_files/
	$(VENV_PY) -c "import arc_agi, logging; logging.disable(logging.WARNING); \
	    a = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL); \
	    [a.make(e.game_id) for e in a.available_environments]; print('games:', len(a.available_environments))"

play: ## Play GAME (comma list or all) for MINUTES each with agent/model.py (needs make llm-local)
	AGENT_TRACE_DIR=.claude/scratch/agent/play AGENT_MAX_TOKENS=8192 $(VENV_PY) -m agent.model --games $(GAME) --minutes $(MINUTES) \
	    --concurrency $(CONCURRENCY)

check: ## Lint + type-check
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy

test: ## Unit + end-to-end tests (no GPU, no network; needs make games for ls20)
	$(VENV_PY) -m pytest -q tests

# ── Local LLM for development (small stand-in for the Kaggle model) ─────────────────────────────
LLM_VENV     := .venv-llm
LLM_MODEL    ?= models/Qwen3.5-4B-AWQ
LLM_PORT     ?= 8000
LLM_MAX_LEN  ?= 32768
LLM_GPU_UTIL ?= 0.93
LLM_FLAGS    := --limit-mm-per-prompt '{"image": 32, "video": 0}' \
                --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
                --default-chat-template-kwargs '{"preserve_thinking": true}'

llm-local-setup: ## One-time: vLLM 0.30.0 in .venv-llm + the 4B dev model in models/
	@[ -x $(LLM_VENV)/bin/python ] || uv venv --python $(PYTHON) $(LLM_VENV)
	uv pip install --python $(LLM_VENV)/bin/python "vllm==0.30.0"
	$(VENV_PY) -c "from huggingface_hub import snapshot_download as d; d('cyankiwi/Qwen3.5-4B-AWQ-4bit', revision='ef85d23beb', local_dir='models/Qwen3.5-4B-AWQ')"

llm-local: ## Serve the dev model on :$(LLM_PORT) with vision, tool calls and 32K context; Ctrl-C stops it
	VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 $(LLM_VENV)/bin/python -m vllm.entrypoints.openai.api_server \
	    --model $(LLM_MODEL) --served-model-name qwen --host 127.0.0.1 --port $(LLM_PORT) \
	    $(LLM_FLAGS) --max-model-len $(LLM_MAX_LEN) --gpu-memory-utilization $(LLM_GPU_UTIL) \
	    --max-num-seqs 8 --max-num-batched-tokens 4096 --enable-prefix-caching
