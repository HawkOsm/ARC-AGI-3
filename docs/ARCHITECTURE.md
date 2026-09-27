# Architecture

**One file: `agent/model.py`.** An LLM plays each game by writing Python: its code inspects the screen and calls
`action(...)` itself. The idea is Tufa Labs' "Duck" (Milestone #1 winner; outline in
[`research/DUCK_HARNESS.md`](research/DUCK_HARNESS.md)); the code is our own, smaller rewrite. LLM first:
algorithms are added only where the model measurably needs them.

```
 play_games ── one thread per game ──► Agent(game).play()
                                          loop until won / time up / action cap:
                                            GAME_OVER? -> RESET
                                            play_turn():
 ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Conversation   system prompt + recent turns (old ones dropped when the context fills; only the   │
 │                newest screen image kept) + the model's notes ("World model:", "Plan:", ...)     │
 │      │ user message: what the last actions did, level, valid actions, notes, screen image       │
 │      ▼                                                                                           │
 │ Model.chat     vLLM OpenAI API, thinking on, one tool: `python`                                  │
 │      │ tool call: code                                                                           │
 │      ▼                                                                                           │
 │ Sandbox.run    a fresh `python -I -S` process, stdlib only, 30 s: the code reads current_frame / │
 │                previous_frame (.ascii, .segmentation = objects + adjacency), history,            │
 │                transitions, last_action_result, valid_actions — and calls action([...]) ──┐      │
 │      ▲                                                                                    │      │
 │      └─ refreshed state after every action ◄── Game.do() → arc_agi env.step() ◄───────────┘      │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            the turn ends once actions ran (or after 12 tool calls)
```

| class | job |
|---|---|
| `Config` | every setting (`AGENT_*` environment variables) |
| `Sandbox` | runs one piece of model code in its own process; relays its `action()` calls |
| `Game` | one arc_agi game: frames, actions in the model's names (UP DOWN LEFT RIGHT SPACE MOUSE UNDO), RESET |
| `LLM` | the chat request with the python tool (recovers tool calls written as markup) |
| `Conversation` | the messages sent, the model's notes, fitting the context budget |
| `Agent` | plays one game: turns, the per-turn prompt, traces (`AGENT_TRACE_DIR`) |
| `play_games`, `main` | many games in parallel on one scorecard; official scores per game |

Local development: `make llm-local` serves a 4B model (vision, tool calls, 32K context) on an 8 GB laptop GPU —
good for checking the plumbing, too weak to judge strategy. Kaggle (RTX PRO 6000) runs the real model; see
`notebooks/duck_repro/` for the serving setup of the public ~5-point notebook.
