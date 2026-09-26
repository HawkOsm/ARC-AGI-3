# The Duck harness (Tufa Labs, Milestone #1 winner) — outline

Read from the public source bundle [`jeroencottaar/taaf-kaggle-source-share`](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share)
(MIT licence; local copy `.claude/scratch/duck/src_original/`, not in git) and the notebooks
[`jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner)
and [`keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp)
(the ~5-point fork: same prompts, tool loop and policy; only the model and serving changed). Most top public
notebooks are forks of this harness. Leaderboard 2026-09-26: Tufa Labs 27.29 first; 20th place 7.22.

## What it is
One LLM, one tool. Each "turn" the model gets a short user message and may call a `python` tool as many times as it
wants; the code it writes inspects the game and **calls `action([...])` itself**, one action or a batch, even in
loops. No hand-written game logic; everything game-specific is written by the model at play time.

| part | file | what it does |
|---|---|---|
| system prompt | `inference/agent/prompts.py` (114 lines) | game overview, visual-game guidance (HUD/timer bars are not puzzle pieces; no player may exist), the runtime variables, Python-tool guidance ("write BFS / search when the goal is understood; probe when not; batch reliable sequences") |
| the tool | `inference/agent/python_tool_sandbox.py` | every call starts fresh; allow-listed stdlib only (collections, heapq, itertools, math, re, ...); 30 s limit; output capped ~1024 tokens |
| what the code sees | `inference/agent/runtime_state.py` | `current_frame` (`.ascii` letter grid, `.segmentation` = objects: id, colour, shape hash, pixels, boundary, children + adjacency list, `.step`, `.level`), `previous_frame`, `history`, `transitions` (before/after/result), `last_action_result` (board_changed, level_completed, game_over, reward), `valid_actions`, `action()`. The raw numeric grid is hidden on purpose |
| the loop | `inference/agent/tool_agent.py` (2063 lines) | per turn: user prompt = what the last sequence did + state + the carried world model; tool calls until actions ran; context kept under the limit by evicting the oldest messages |
| memory | `tool_agent.py` `_summarized_knowledge` | the model writes `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, `Cross-level notes:`; these are carried into the next turn's prompt ("copy it and revise it before acting") |
| image | `inference/agent/vision_context.py` | the current grid as an image (upscale 4) next to the text |
| actions | `inference/agent/action_names.py` | `UP DOWN LEFT RIGHT SPACE MOUSE(row, col)` (row/col, not x/y) |
| game runner | `inference/framework/solver.py` + TAAF framework | 28 games at once, ~2.2 h per game budget, restarts, scoring |

## Model and serving (config `configs/inference.json`, fork's notebook)
- Original: Qwen3.6-27B-FP8; the ~5-point fork: Qwen3.8-Flash-Next NVFP4 (135 GB checkpoint) with 3-token MTP
  speculative decoding, 32K context, 8 sequences, CUDA graphs, a server watchdog.
- **Thinking ON** (temperature 0.6, top_p 0.95, top_k 20), tool-call parser `qwen3_coder`, reasoning parser `qwen3`,
  `preserve_thinking` in the chat template, vision on (`language_model_only: false`).

## How it differs from our agent
| | Duck | us (hypothesis mode) |
|---|---|---|
| who acts | the model, through code it writes | the brain; the LLM ~5% of actions |
| game knowledge in code | none (segmentation + history only) | object tracking, shapes, control modes, goals, steering |
| model output | free reasoning (thinking on) + Python | one forced-JSON answer, thinking off |
| memory | model-written world model, carried every turn | brain beliefs + LLM history lines |
| search / BFS | the model writes it when it understands the goal | the brain's BFS on its own goal list |
