# AGENT.md

Instructions for any AI agent (Claude Code, `agy`, or otherwise) working in this repo. Read this
first. Working conventions are carried over from the sibling Kaggriculture repo — they are why
results here are trustworthy instead of noise.

## What this repo is

An agent for Kaggle's **ARC Prize 2026 — ARC-AGI-3** (deadline 2026-11-02): play unseen
turn-based grid games (64×64, 16 colours) with no instructions, scored per level by action
efficiency against human baselines. Built on the official starter kit (CREDITS.md).

**Live champion: the pruned brain** (`agent/brain/`, promoted 2026-09-26, frozen in
`eval/champion/my_agent.py`; local only — the Kaggle LB entry is still floor v1, LB 0.44).

## Quick commands

```bash
make setup                 # uv venv + pinned arc-agi 0.9.8 + framework @135f20a (== Kaggle)
make build                 # agent/brain/ -> agent/my_agent.py (never hand-edit the latter)
make check                 # ruff + "my_agent.py is freshly built"
make interface             # re-verify interface facts (run after any version bump)
make eval REPS=30 BUDGET=2000          # N-repeat eval of agent/my_agent.py on all 25 games
make eval-paired REPS=30 BUDGET=2000   # candidate vs eval/champion/my_agent.py (promotion test)
make eval-swarm            # all games as threads in one process (Kaggle's execution model)
.venv/bin/python eval/kaggle_sim.py --games all --max-actions 500   # run the real notebook vs a competition-mode gateway
make ci                    # what GitHub Actions runs
.venv/bin/python eval/explain.py ls20 800   # play a game, then print the World (beliefs) in words
make notebook [AGENT=...]  # build notebooks/submission.ipynb (no push)
```

Kaggle actions are **manual** unless autonomous mode is explicitly on (rule 7): `make submit`
pushes the kernel (Phase A); the leaderboard submission is the "Submit to Competition" click on
the kernel page (Phase B) — **1 per day** (competition rules §2; the starter README's "5" is wrong
for this competition). ~38 submissions remain from 2026-09-25 to the deadline: spend them only on
paired-verified champions or deliberate calibration probes. Each scored rerun takes ~TIME_BUDGET_S
(7.5 h) of Kaggle time.

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
12. **No model training** (owner decision, 2026-09-26): training loops slow development down.
    The agent learns *while it plays* (the World beliefs) — that is fine; offline training or
    fine-tuning of networks is not. Pre-trained models used as-is (inference only) are still
    allowed by the competition, but only via ROADMAP Phase 3 and a head-to-head win.

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

## Architecture (agent/brain/)

The idea: **fixed, game-agnostic code + a `World` of beliefs that starts almost empty and fills
up as the agent plays.** Code never changes per game; only the beliefs do. Map for newcomers:
docs/ARCHITECTURE.md.

```
agent/brain/
  settings.py      every tunable number, with why it matters (constants)
  world.py         MODEL   - typed belief classes + read-only queries: World, CounterMemory, AvatarBelief,
                             ColourRoles, Kind, Pattern, Item, ActionBeliefs, Rules, Goal, Death, Screen,
                             LevelMemory, Tally, WorkingMemory; enums Outcome, Role; records Action, ClickInfo
  look.py          READER  - ScreenReader (static: settled picture, objects -> Blob, changed blobs, shapes),
                             FirstLook (a level's inventory: item kinds with purpose "?", patterns)
  learn.py         LEARNERS- the only writers of beliefs: CounterLearner, AvatarLearner, TouchLearner,
                             ActionLearner, DeathJudge
  goals.py         GOALS   - ScreenFeatures (static), Goals (goals from wins, progress toward them)
  decide.py        DECIDER - Explorer: reads beliefs, picks the next action
  llm.py           ADVISOR - Advisor (HTTP client to a local vLLM server), PromptWriter, parse_plan
  brain.py         ORCHESTRATOR - Brain: look -> events -> learn -> decide (read this first)
  kaggle_agent.py  MyAgent: plugs the Brain into Kaggle's framework, with all the hardening
agent/my_agent.py  generated single file that ships (make build) - never edit by hand
```

### Code style: Python written like Java (owner, 2026-09-26)

- **Typed classes, not dicts.** Beliefs are classes with named, annotated fields (`@dataclass(slots=True)`
  for plain records, a normal class when it has behaviour). No `thing["field"]` for our own data.
- **Enums for fixed sets** (`Outcome`, `Role`, later game modes / object classes), never magic strings.
- **Records for small immutable values** (`Action`, `ClickInfo` are `NamedTuple`s: equal to plain tuples).
- **One job per class, explicit dependencies** passed to the constructor (`Explorer(world, memory, goals,
  rng, stats, restart)`), no reaching into another object's internals.
- **Model vs services:** `world.py` holds data + read-only queries; learners are the only writers;
  deciders only read beliefs (and write short-term memory).
- **Static utility classes** for pure helpers (`ScreenReader`, `ScreenFeatures`).
- **mypy is the compiler:** `make check` runs it on agent/brain (config in pyproject.toml); keep it clean.
- Only the standard library + numpy in agent/brain (Kaggle ships nothing else we can rely on); no pydantic
  or other frameworks in the per-action path.

The beliefs (`world.py`):

| belief | holds | written by | kept across levels |
|---|---|---|---|
| `world.counters` | which pixels are HUD step-counters (ignored when comparing screens) | `learn.CounterLearner` | yes |
| `world.avatar` | which colour is the player (moves differently per key), its extra "part" colours, how far each key moves it | `learn.AvatarLearner`, `learn.TouchLearner` | yes |
| `world.kinds` | every distinct item type (colour + exact shape) seen at a first look, purpose "?" until learned | `look.FirstLook` | yes |
| `world.colour_roles` | what touching each colour did: floor / wall / hazard / goal / pushable / collectible (a belief; using it to predict wall bumps measured harmful and was removed) | `learn.TouchLearner` | yes |
| `world.patterns` | first-look observations: repeated kinds, one shape in several colours | `look.FirstLook` | yes |
| `world.actions` | per action: changes the screen? undo? reverses another? no-op / deadly at a position? | `learn.ActionLearner` | yes |
| `world.rules` | steps per life | `learn.DeathJudge` | yes |
| `world.levels[n]` | the first-look inventory (`items`) and the map: every screen seen and what each action did on it | `look.FirstLook`, `Brain._record_move` | per level |
| `world.goals` | what finishes a level, as the game showed it: feature changes (colour counts, colour-colour touches, what the avatar touches) between a level's first screen and its winning screen; confirmed or contradicted by every later win | `goals.Goals` (from wins; weak first-look cues) | yes |
| `world.deaths` | real deaths, re-judged as timeouts when a budget appears | `Brain._on_death` | — |

One step (`Brain.next_action`):
1. **LOOK** — the settled picture (last animation layer).
2. **EVENTS** — game over → blame the last action (unless it was a timeout) and RESET; level
   up → record LEVEL_UP, start a new level map. The level change is the only real reward.
3. **LEARN** — first screen of a level → `FirstLook` inventory. Then grow the counter mask (re-identify all screens if it grew), update the avatar,
   identify this screen, record `last screen + last action → this screen`, update action
   beliefs, learn what touching each colour did (one wall bump teaches every cell of that colour). A planned route whose next screen isn't what we got is dropped (never trust a plan
   the world disagrees with).
4. **DECIDE** (`decide.py`) — level solved → shortest known route to LEVEL_UP; else the best
   untried action here if promising enough; else the
   (screen, untried action) with the fewest expected actions per new screen, `(distance+1)/chance`,
   within this life's step budget; else RESET (only if steps were taken); else click the screen
   on a finer grid; else random.

Every piece in `agent/brain/` survived the 2026-09-25 ablation (docs/tests/IDEAS_TRIED.md): additions
that measured nothing were deleted rather than kept "just in case".

Deviations from the original architecture brief, and why:
- Counter masking is **tick-track based**, not frequency based: the HUD counter present in every
  public game is monotone, so per-cell frequency never flags it (frequency masking is kept for
  blinkers).
- The transition "model" is an exact per-level screen map plus generalising action beliefs, not a
  learned next-state predictor: games are deterministic given the full state, so exact results are
  never wrong where recorded; generalisation lives in `world.actions`.
- LLM advisor (`llm.py`, ROADMAP Phase 3): optional, on only when `ARC3_LLM_URL` is set; with it off
  the agent is run-for-run identical to the brain. See docs/ARCHITECTURE.md.

Goals are never written in the code (owner's rule): the only goal vocabulary is generic screen
features, and the game's own level completions decide which features matter and in which
direction. Before the first win the only goals are first-look cues (one shape in two colours ->
"make them touch"), which later wins confirm or kill.

Next: a view-scroll detector so the map is kept in world coordinates if a game's camera moves (none
of the 25 public games scroll in 300 random moves each; the engine supports cameras).

## Load-bearing constraints

- `KEEP_FRAMES = 4` — without it memory grows ~unbounded per game (framework keeps every frame).
- `TIME_BUDGET_S = 7.5 h` from import — Kaggle's limit is **9 h run-time** per notebook (CPU or
  GPU, competition Code Requirements); the margin covers notebook setup, the gateway wait and
  scorecard close. The official sample uses 8 h − 5 min. All games share one process/GIL, so per-action compute sets the
  achievable action count per game.
- Thread-safe clicks (`MyAgent._click` → `arc_env.step`) — the shared enum race is real.
- Never RESET with 0 actions in the level (that restarts the whole game). The Brain only RESETs
  on GAME_OVER / NOT_PLAYED / WIN, or voluntarily when `life_length > 0`.
- Screen re-identification when the counter mask grows (`LevelMemory.rename_screens`; each
  screen keeps a compressed snapshot, ~200 B) — dropping the map instead would throw away all
  exploration each time the mask grows.
- Click grids inside big objects + whole-screen click grids on exhausted screens — object-centre
  clicks alone stall on games whose clickable unit is part of an object (ft09, lp85).
- Timeout deaths are not blamed on the last action (`learn.DeathJudge`) — otherwise legal
  moves get blacklisted as deadly (sp80: 44/44 "deaths" were timeouts).
- Memory: compact `(prior, colour)` candidate info, capped 1-pixel scans, `MEM_GUARD_*` — all games
  share one Kaggle process and an out-of-memory kill zeroes the whole submission.

## Tried and ruled out

See `docs/tests/IDEAS_TRIED.md` (concept-indexed) and `docs/tests/LOG.md` (chronological).
