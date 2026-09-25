# Roadmap

Competition: ARC Prize 2026 — ARC-AGI-3 (Kaggle), final deadline **2026-11-02**. Public LB top on
2026-09-25: 19.45 (score is 0–100). 5 submissions/day.

## Where we are (2026-09-25)

- ✅ Environment: `make setup` (uv venv; `arc-agi` 0.9.8 / `arcengine` 0.9.3 / framework `135f20a`,
  all pinned to what Kaggle's dataset ships), `make verify-local` passes.
- ✅ Interface facts verified in code (`make interface`, 11/11) — see AGENT.md.
- ✅ Eval harness (`eval/run.py`): N-repeat, official scorecard, paired champion-vs-candidate,
  swarm mode. Kaggle emulator (`eval/kaggle_sim.py`) runs the real notebook against a local
  competition-mode gateway.
- ✅ Floor agent (`agent/core/`): masked state graph + learned action model + avatar tracking +
  verified BFS navigation. Beats random at N=30 (LOG.md).
- ⏳ **Kaggle Phase A/B pipeline check — blocked on the owner accepting the competition rules**
  (`kaggle` CLI: `userHasEntered: False`, data download 403). Commands are ready (AGENT.md).

## Phases

### Phase 1 — pipeline truth (now)
1. Owner accepts rules → push the random baseline (`make submit AGENT=experiments/2026-09-25_random_baseline/my_agent.py`),
   Phase A completes, submit → confirms the pipeline end to end (expected LB ≈ 0).
2. Submit the floor champion → first real LB number. Compare against local N=30 aggregate: the
   ratio (LB / local) is the first calibration point for how much public-set scores transfer.
3. Replay question: settled locally — under a competition-mode gateway (eval/kaggle_sim.py) the
   level-start full reset is intercepted and the gain vanishes (9.76 → 1.36). Only revisit if a
   Kaggle-side observation contradicts competition mode; not worth a daily slot otherwise.

### Phase 2 — exploration efficiency (the measured bottleneck)
The floor is **stuck, not budget-limited** (5× budget bought nothing). In order of expected value:
- **State abstraction**: thousands of states on re86/tr87/su15 with no level-up. Hash the
  *object configuration* (positions of movers) rather than full pixels; merge states differing
  only in irrelevant pixels.
- **Hidden-state handling**: g50t shows hundreds of graph conflicts (same key, different
  outcome). Detect and stop trusting the graph there (fall back to model-guided exploration).
- **Death economics**: every death costs a RESET plus the walk back. Use the avatar-position
  death table more aggressively; prefer frontiers near the level start after deaths.
- **Goal inference**: prioritise actions/clicks that change *rare* colours or move the avatar
  toward salient unvisited objects (a cheap novelty/salience prior, not a planner).
- **Speed**: per-action compute matters on Kaggle — all games share one GIL. Profile per game;
  keep < ~5 ms/action median.

### Phase 3 — reasoning rescue (only if Phase 2 plateaus)
Programmatic floor stays in charge; an LLM (loaded once, thread-safe singleton — never per game)
is queried only on stall (N actions without new states/levels). Must beat the floor head-to-head
on the full public set before any submission uses GPU quota.

## Explicit non-goals

- **No per-game special-casing** keyed on `game_id` (the hidden set is ~4× larger; GAME_GUIDE.md is
  for understanding mechanisms, not for lookup tables).
- **No promotion from a single-game or two-game gain**, and no promotion without a paired N≥30
  run against the current champion (AGENT.md rule 3).
- **No reliance on the max-over-plays replay lever** unless Kaggle behaviour is confirmed.
- **No LLM-primary agent.** Programmatic floor first; LLM as rescue only.
- **No GPU accelerator** while the agent is rule-based (`ACCELERATOR = "cpu"`).
