# ARC-AGI-3 agent

Agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle competition (interactive reasoning games; deadline 2026-11-02).

**Start with [AGENT.md](AGENT.md)** — architecture, verified interface facts, standing rules, commands.

| | |
|---|---|
| `agent/brain/` | the agent (edit here) — start with `brain.py`, beliefs in `world.py` |
| `agent/my_agent.py` | generated single file that ships (`make build`) |
| `eval/` | local harness: `run.py` (N-repeat / paired / swarm), `kaggle_sim.py`, `interface_checks.py`, `regression_check.py`, `champion/` |
| `docs/` | `ROADMAP.md`, `GAME_GUIDE.md`, `tests/LOG.md`, `tests/IDEAS_TRIED.md`, `STARTER_README.md` |
| `experiments/` | dated approach folders with frozen variants |
| `scripts/` | starter-kit scripts + `build_submission.py` |

Built on the official [starter kit](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter); see [CREDITS.md](CREDITS.md).
