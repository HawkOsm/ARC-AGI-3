# AGENT.md

Instructions for any AI agent (Claude Code, `agy`, or otherwise) working in this repo. Read this first.

## What this repo is

An agent for Kaggle's **ARC Prize 2026 — ARC-AGI-3** (deadline 2026-11-02): play unseen turn-based grid games
(64×64, 16 colours) with no instructions, scored per level by action efficiency against human baselines.

**The agent is one file: `agent/duck.py`** — an LLM that plays by writing Python (the idea of Tufa Labs' "Duck",
the Milestone #1 winner; our own rewrite, see CREDITS.md). LLM first, algorithms later: code is added only where
the model measurably needs it. Map: docs/ARCHITECTURE.md.

Reference numbers (docs/tests/LOG.md): the public Duck notebook, unchanged, scores **9.81 / 38 levels** on the 25
public games (Kaggle run A, 2026-09-27). Our deleted brain-first agent reached 10–12 levels (~0.7).

## Quick commands

```bash
make setup games           # venv + pinned arc-agi, then the 25 public game sources
make llm-local-setup       # once: vLLM + the local 4B dev model
make llm-local             # serve it (vision + tool calls + 32K context)
make play GAME=ar25 MINUTES=10     # play with agent/duck.py; traces in .claude/scratch/duck/play/
make check test            # ruff + mypy, pytest
```

Kaggle runs and submissions are **manual** (the owner pushes, or explicitly allows it for a session). GPU runs cost
quota (30 h/week): develop locally, and use Kaggle only to answer "does it score". Submissions: 1 per day.

## Standing rules

1. **Every test run is logged** in `docs/tests/LOG.md` (newest after the top `---`): what changed, the exact
   command, raw numbers, verdict. Format in `docs/tests/README.md`.
2. **Check `docs/tests/IDEAS_TRIED.md` before trying a strategic idea**; add a row when an idea gets a verdict.
3. **Change one thing, measure it enough times to trust it** (runs vary ±0.1 on the 25-game score with a
   sampling LLM), keep only what wins. Compare against the current best on the same games and budget.
4. **Less code is better.** Every piece must earn its place with a measurement; delete what does not.
5. **Scratch files go in `.claude/scratch/<task-name>/`** — never the repo root or /tmp.
6. **Credit** anything taken from a public notebook, paper or repo in CREDITS.md.
7. **Public-set scores are provisional** (25 public games vs a larger hidden set). Never special-case a game id.
8. **No model training** (owner, 2026-09-26): pre-trained models used as-is (inference only).
9. Kaggle GPU runs need the owner's explicit go-ahead for that run.

## Verified interface facts

Checked by code or by reading `arc_agi`/`arcengine` (2026-09-25), not assumed. Re-verify after any version change.

- `FrameData` has **no `score`**: progress is `levels_completed` / `win_levels`. `frame` is a
  **list of grids** (animation layers; up to 42 observed) — the settled state is `frame[-1]`.
- **`GameAction(6)` raises `ValueError`** (enum values are `(id, type)` tuples). Use
  `GameAction.from_id()`. `set_data({"x": 64})` raises `ValidationError` (0–63 only). Either,
  uncaught in `choose_action`, kills the game's thread → the game silently stops scoring.
- **Actions not in `available_actions` do not raise but still count** as actions.
- **RESET semantics**: RESET = restart the current level. But RESET with 0 actions taken in the
  level (right after a level-up, or two RESETs in a row) = **full reset to level 1**, and the
  scorecard starts a **new play**. Never RESET at a level start. After GAME_OVER, any non-RESET
  action returns an empty frame — RESET is mandatory.
- Initial grids are deterministic across fresh environments (same game → same level start).

## Scoring facts (from `arc_agi/scorecard.py`, identical in 0.9.8 and 0.9.9)

- Per level: `min(115, 100 · (baseline / actions)²)` if completed, else 0. **RESETs count as
  actions.** Actions spent on a level you never complete cost nothing extra.
- Per game: weighted mean over *all* levels with weight = level index (1-based) — level 1 of an
  8-level game is worth at most 1/36 of the game. **Depth dominates.**
- Per game = **max over plays** (full reset → new play); total = mean over games.
- **Competition mode** (arc_agi server, what Kaggle's gateway almost certainly runs): a RESET
  issued with 0 actions in the level is *intercepted* (not executed, current frame returned), a
  second scorecard cannot be opened, and a game cannot be re-created. So the "full-reset and
  replay known solutions" lever that works locally is expected to be **dead on Kaggle** (see
  IDEAS_TRIED "Scoring levers" and LOG.md). By source reading (not yet exercised — the agent
  has never won a whole game), RESET after WIN still executes as a full reset in competition
  mode, since the final level's action count is not zeroed on WIN.
- No action cap on Kaggle; the binding limit is wall-clock: **9 h** notebook run-time.
- Kaggle's overview says per-game scores are capped at 100% (the local scorer caps each *level*
  at 115 and the game at 100 via the max-weights rule — consistent).

## Compute rules (Kaggle overview, read 2026-09-25)

- Internet disabled in scored runs; freely & publicly available external data and pre-trained
  models are allowed.
- RTX 6000 (`g4-standard-48`) is available only to notebooks **attached to this competition**
  (any such notebook, not just the submission); using it for anything else risks moderation.
  Internet must be off in RTX sessions; it burns GPU quota faster and queues can be long.
- Prize eligibility requires open-sourcing; milestone prizes require a public notebook by the
  milestone date (Milestone 2: 2026-09-30).

## Code style: Python written like Java (owner, 2026-09-26)

- **Typed classes, not loose dicts** for our own data (`@dataclass(slots=True)`); dicts only where JSON goes in
  or out (model messages, sandbox state).
- **One job per class** (`Sandbox`, `Game`, `Model`, `Conversation`, `Duck`), dependencies passed in.
- **mypy is the compiler** (`make check`); only the standard library + numpy in `agent/` (Kaggle ships nothing
  else we can rely on).

## Tried and ruled out

See `docs/tests/IDEAS_TRIED.md` (by idea) and `docs/tests/LOG.md` (by date).
