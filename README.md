# ARC-AGI-3 agent

Agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle competition (interactive reasoning games; deadline 2026-11-02).

**Start with [AGENT.md](AGENT.md)** — rules, commands, verified interface and scoring facts.

| | |
|---|---|
| `agent/model.py` | the whole agent: an LLM that plays by writing Python (read top to bottom) |
| `tests/` | `make test` (no GPU, no network) |
| `notebooks/agent/` | our Kaggle notebook: `build.py` embeds `agent/model.py` + settings (model, context, prefix cache, ...) |
| `notebooks/duck_repro/` | the public Duck notebook, unchanged — our Kaggle baseline and notebook template |
| `notebooks/llm_prep/`, `llm_dev/` | serving our own model on Kaggle (vLLM wheelhouse, model mirror check) |
| `docs/` | `ARCHITECTURE.md`, `GAME_GUIDE.md`, `research/`, `tests/LOG.md`, `tests/IDEAS_TRIED.md` |

Credits: [CREDITS.md](CREDITS.md).
