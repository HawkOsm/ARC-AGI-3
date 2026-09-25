# AGENT.md

Instructions for any AI agent (Claude Code, `agy`, or otherwise) working in this repo. Read this
first. Working conventions are carried over from the sibling Kaggriculture repo — they are why
results here are trustworthy instead of noise.

## What this repo is

An agent for Kaggle's **ARC Prize 2026 — ARC-AGI-3** (deadline 2026-11-02): play unseen
turn-based grid games (64×64, 16 colours) with no instructions, scored per level by action
efficiency against human baselines. Built on the official starter kit (CREDITS.md).

**Live champion: the "floor" agent** in `agent/core/` (flattened into `agent/my_agent.py`, frozen
copy in `eval/champion/my_agent.py`). See "Architecture" below and LOG.md for its numbers.

## Quick commands

```bash
make setup                 # uv venv + pinned arc-agi 0.9.8 + framework @135f20a (== Kaggle)
make build                 # agent/core/ -> agent/my_agent.py (never hand-edit the latter)
make check                 # ruff + "my_agent.py is freshly built"
make interface             # re-verify interface facts (run after any version bump)
make eval REPS=30 BUDGET=2000          # N-repeat eval of agent/my_agent.py on all 25 games
make eval-paired REPS=30 BUDGET=2000   # candidate vs eval/champion/my_agent.py (promotion test)
make eval-swarm            # all games as threads in one process (Kaggle's execution model)
.venv/bin/python eval/kaggle_sim.py --games all --max-actions 500   # run the real notebook vs a competition-mode gateway
make ci                    # what GitHub Actions runs
make notebook [AGENT=...]  # build notebooks/submission.ipynb (no push)
```

Kaggle actions are **manual** unless autonomous mode is explicitly on (rule 7): `make submit`
pushes the kernel (Phase A); the leaderboard submission is the "Submit to Competition" click on
the kernel page (Phase B) — 5/day.

## Standing rules

1. **Every test run is logged** in `docs/tests/LOG.md` (newest after the top `---`): what changed,
   the exact command, raw numbers, verdict. Format in `docs/tests/README.md`.
2. **Check `docs/tests/IDEAS_TRIED.md` before trying a strategic idea**; add a row the moment an
   idea gets a verdict.
3. **Promotion = paired head-to-head against the CURRENT champion** (`make eval-paired`, all 25
   games, N≥30, same seeds) **plus** a per-game regression check in the same run (no game may
   drop meaningfully to buy gains elsewhere). Beating a static benchmark or the random baseline
   is not beating the incumbent. After promoting: copy the flattened file to
   `eval/champion/my_agent.py`, update `eval/champion/CHAMPION.md`, re-baseline CI with
   `eval/regression_check.py --update`.
4. **N ≥ 30** reps before trusting a close result; a diff inside its 95% CI is logged as NOISE.
5. **Never read a full replay/trace into context.** Script a summary from `runs.jsonl`
   (scores, per-level actions, `agent_stats`, failure step).
6. **Scratch files go in `.claude/scratch/<task-name>/`** — never the repo root or /tmp.
7. **Git and Kaggle submission actions stay manual** unless the owner turned on autonomous mode
   for the session; hand over the exact command instead.
8. **Credit** anything studied from `make pull-sample` or a public ARC-AGI-3 notebook/paper in
   CREDITS.md if it shapes the design.
9. **Document load-bearing constraints** (below) instead of silently removing them.
10. **Public-set scores are provisional.** 25 public games vs a ~4× larger hidden set. Never
    special-case a `game_id`. A gain concentrated in 1–2 games is tuning until shown otherwise.
11. **Local == Kaggle is maintained on purpose**: versions are pinned to the competition
    dataset's wheels and framework commit. If Kaggle's dataset changes, re-pin and re-run
    `make interface` before trusting any local number.

## Verified interface facts

Each is checked by code (`eval/interface_checks.py`, source reading of `arc_agi`/`arcengine`),
not assumed. Re-verify after any version change.

- `FrameData` has **no `score`**: progress is `levels_completed` / `win_levels`. `frame` is a
  **list of grids** (animation layers; up to 42 observed) — the settled state is `frame[-1]`.
- **`GameAction(6)` raises `ValueError`** (enum values are `(id, type)` tuples). Use
  `GameAction.from_id()`. `set_data({"x": 64})` raises `ValidationError` (0–63 only). Either,
  uncaught in `choose_action`, kills the game's thread → the game silently stops scoring.
- **`GameAction` members are process-global singletons.** `ACTION6.set_data()` in one Swarm
  thread is visible to all others (verified with two threads) → on Kaggle, concurrent games can
  send each other's click coordinates. `MyAgent.do_action_request` passes this instance's own
  coordinates straight to `arc_env.step()`.
- **Actions not in `available_actions` do not raise but still count** as actions.
- **RESET semantics**: RESET = restart the current level. But RESET with 0 actions taken in the
  level (right after a level-up, or two RESETs in a row) = **full reset to level 1**, and the
  scorecard starts a **new play**. Never RESET at a level start. After GAME_OVER, any non-RESET
  action returns an empty frame — RESET is mandatory.
- A failed step (gateway HTTP error / 10 s timeout on Kaggle) returns `None` and the base
  `Agent.take_action` then raises → thread death. `MyAgent.take_action` swallows, backs off,
  and gives up after `MAX_CONSECUTIVE_ERRORS`.
- The base `Agent` keeps every `FrameData` forever (Python lists of 64×64 ints, ~GBs over a long
  game ×100 games). `MyAgent.append_frame` keeps `KEEP_FRAMES`.
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
- No action cap is enforced by the framework on Kaggle other than the agent's own
  `MAX_ACTIONS`; the binding limit is wall-clock: **9 h** notebook run-time.
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

## Architecture (agent/core/)

```
config.py      every knob, commented
perception.py  settled_grid, components() (row runs + union-find, ~2 ms), VolatileMask, AvatarTracker
memory.py      LevelGraph (masked-state graph, re-keyable), ActionModel (learned priors)
explorer.py    Explorer: the policy (framework-independent)
framework.py   MyAgent(Agent): thread-safe action data, error hardening, frame cap, time guard
```

Per step: settled grid → mask update (re-key all graphs if the mask grew) → masked key → record
`(prev_key, prev_action) → key | DEATH | LEVELUP` → drop the plan if the key is not what the plan
expected (open-loop plans are never trusted) → choose:
1. level already solved → BFS shortest path to its LEVELUP edge;
2. else the best untried action here if its learned p ≥ `LOCAL_GREEDY_P`;
3. else BFS over known edges for the frontier minimising `distance + λ(1 − p)`;
4. else refine the current state's click lattice (stride 4 → 2 → 1) and retry; else random legal.

Learned model (`ActionModel`): per-action and per-click-colour success rates (Laplace with
priors), undo detection (action returns to the pre-previous state ≥80%), pair-undo (b after a
reverts ≥90% → demote b after a), avatar-position no-op/death tables (walls/pits generalise
across states that differ elsewhere). Avatar = colour whose changed-cell centroid shift is
consistent per action and differs across actions (environmental movers shift identically
regardless of action).

Deviations from the architecture brief, and why:
- Volatile masking is **border-tick based**, not frequency based: the HUD counter present in
  every public game is monotone, so per-cell frequency never flags it (frequency masking is kept
  as a secondary rule for blinkers).
- Transition "model" is an exact per-level graph plus generalising priors, not a learned
  next-state predictor: games are deterministic given the full state, so exact edges are
  cheaper and never wrong where recorded; generalisation lives in the priors.
- No LLM: programmatic floor first (ROADMAP Phase 3).

## Load-bearing constraints

- `KEEP_FRAMES = 4` — without it memory grows ~unbounded per game (framework keeps every frame).
- `TIME_BUDGET_S = 7.5 h` from import — Kaggle's limit is **9 h run-time** per notebook (CPU or
  GPU, competition Code Requirements); the margin covers notebook setup, the gateway wait and
  scorecard close. The official sample uses 8 h − 5 min. All games share one process/GIL, so per-action compute sets the
  achievable action count per game.
- Thread-safe action data (`_pending` → `arc_env.step`) — the shared enum race is real.
- Never RESET with 0 actions in the level (full reset). The explorer only RESETs on GAME_OVER,
  NOT_PLAYED, or (replay variant only) deliberately.
- Graph re-key on mask change (nodes keep a zlib-packed raw grid, ~200 B each) — dropping the
  graph instead would throw away all exploration each time the HUD mask grows.
- Click lattice + refinement — component-centroid clicks alone stall on games whose clickable
  unit is a sub-part of a component (ft09, lp85).

## Tried and ruled out

See `docs/tests/IDEAS_TRIED.md` (concept-indexed) and `docs/tests/LOG.md` (chronological).
