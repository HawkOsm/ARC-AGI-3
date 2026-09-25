# Test Log

See [`README.md`](README.md) for the entry format and convention. Newest first.

---

## 2026-09-25 — Parity fix: Kaggle runtime numpy is 2.0.2, not the 2.4.4 wheel
- Change: Makefile pin numpy==2.4.4 → 2.0.2. Kernel v2 log shows pip keeping the image's
  preinstalled numpy 2.0.2 ("Requirement already satisfied ... (2.0.2)").
- Also: kernel v2 (floor champion) Phase A COMPLETE, clean log, dummy submission.parquet.
- Command: `make setup && make interface && eval/regression_check.py && eval/run.py --reps 3 --max-actions 1000`
- Result: interface 11/11; regression PASS with an exact match (1.2847 / 1.17 levels); all-games
  sanity 0.577 ± 0.264 (n=3), consistent with the N=30 0.628 ± 0.106 at 2000 actions.
- Verdict: INFO — no behaviour change from the numpy version; parity restored.

## 2026-09-25 — Kaggle Phase A: random baseline kernel v1
- Change: none — `make submit AGENT=experiments/2026-09-25_random_baseline/my_agent.py` after the
  owner accepted the rules (CLI now `userHasEntered: True`)
- Result: hawkosm/arc-agi-3-agent v1 → `KernelWorkerStatus.COMPLETE`; output has the expected dummy
  `submission.parquet` (1 row) + log. Kaggle's ARC-AGI-3-Agents agent.py/swarm.py/main.py
  downloaded and diffed: byte-identical to commit 135f20a (the local pin).
- Verdict: INFO — Phase A pipeline OK. Phase B (competition submission) awaiting owner go-ahead.

## 2026-09-25 — Floor through the real notebook, competition-mode gateway, all 25 games
- Change: none — champion (agent/my_agent.py == eval/champion/my_agent.py) after re-pinning to
  arc-agi 0.9.8 / numpy 2.4.4 (later corrected to 2.0.2, see next entry) / framework 135f20a (Kaggle's dataset files verified byte-identical
  to 135f20a after rules acceptance)
- Command: `eval/kaggle_sim.py --agent agent/my_agent.py --games all --max-actions 500`
- Result: runs end to end (25 threads, HTTP, competition mode), 0 errors in logs.log, 119 s,
  aggregate **0.407** at a 500-action cap (single run; not comparable to the 2000-action N=30).
  `make interface` 11/11 on the pinned versions; `regression_check.py` PASS (baseline written).
- Verdict: INFO — the submission artifact works under Kaggle-like execution.

## 2026-09-25 — PROMOTED: floor agent vs random starter, all 25 games, N=30 paired
- Change: floor agent (agent/core/, incl. click lattice fix) vs experiments/2026-09-25_random_baseline
- Command: `eval/run.py --agent agent/my_agent.py --champion experiments/2026-09-25_random_baseline/my_agent.py --reps 30 --max-actions 2000`
- Result: floor **0.628 ± 0.106** vs random **0.063 ± 0.030**; paired **+0.565 ± 0.110** (n=30,
  SIGNIFICANT); 0/1500 crashes. Wins on 16 games (30/0 on ar25, bp35, lf52, lp85, ls20, s5i5,
  su15, tn36, vc33). Biggest: lp85 +5.18 (3.0 levels, max 5), tn36 +3.56, vc33 +2.39, sc25 +1.38.
  **Regression: sp80 −1.18 ± 0.62** (2/28/0) — random clears L1 in ~242 actions, floor in ~1214.
  0 levels on 9 games (cn04, dc22, g50t, ka59, re86, sb26, sk48, tr87, wa30); ft09 0.5 levels.
  (Ran on arc-agi 0.9.9 before the pin; 0.9.8 differs only in an error message.)
- Verdict: PROMOTED — first champion; frozen to eval/champion/my_agent.py, CI baseline written.
  sp80 regression logged in IDEAS_TRIED (likely click-target dilution in a mixed game).

## 2026-09-25 — Replay variant under a competition-mode gateway (kaggle_sim)
- Change: none — measurement of experiments/2026-09-25_replay/my_agent.py (rebuilt with the
  `ARC3_MAX_ACTIONS` override) through the real notebook
- Command: `eval/kaggle_sim.py --agent experiments/2026-09-25_replay/my_agent.py --games vc33,tu93,lp85,m0r0 --max-actions 600 [--no-competition]`
- Result: competition_mode=True **1.364** aggregate, every game `plays=1` (full resets
  intercepted); competition_mode=False **9.757**, plays 2–4 (vc33 L1 3 actions, lp85 4 levels at
  5/41/18/60 actions). Single run each.
- Verdict: REJECTED for Kaggle unless Kaggle's gateway is shown to run non-competition mode. The
  lever is real only under the local non-competition scorer. Do not pursue further without a
  Kaggle-side observation.

## 2026-09-25 — Kaggle emulator works; competition-mode gateway (random baseline)
- Change: new `eval/kaggle_sim.py` — builds the real notebook, runs its cells against a local
  arc_agi gateway with `competition_mode=True`, parses the framework's FINAL SCORECARD.
- Command: `.venv/bin/python eval/kaggle_sim.py --agent experiments/2026-09-25_random_baseline/my_agent.py --games ls20,vc33`
- Result: runs end to end in 9 s; scorecard reports `competition_mode: true`; ls20/vc33 81 actions
  each (MAX_ACTIONS 80 + framework off-by-one), score 0.
- Verdict: INFO — the notebook artifact itself (not just the agent file) is now testable locally.

## 2026-09-25 — Replay-after-level-up variant (max-over-plays), local scorer
- Change: `REPLAY_ENABLED=True` (experiments/2026-09-25_replay/my_agent.py)
- Command: `eval/run.py --agent experiments/2026-09-25_replay/my_agent.py --champion agent/my_agent.py --games vc33,lp85,tu93,tn36,bp35,m0r0 --reps 2 --max-actions 2000`
- Result: 6.940 ± 4.105 vs 3.082 ± 2.945; paired +3.858 ± 1.160 (n=2). vc33 L1 in 3 actions
  (baseline 7), tu93 L2 in 10 (baseline 16). 0 plan mismatches during replays.
- Verdict: ⚠️ scorer-dependent. Source reading afterwards: arc_agi's server in competition mode
  intercepts the level-start RESET that triggers the full reset → expected dead on Kaggle. n=2
  and 6 games; not a promotion candidate.

## 2026-09-25 — Click lattice + exhausted-state refinement
- Change: `perception.components()` rewritten (row runs + union-find, ~2 ms vs ~11 ms, 0/300
  mismatches vs reference BFS); lattice click points in components > 24 cells; stride 4→2→1
  whole-grid refinement when no untried action is reachable.
- Command: `eval/run.py --games ft09,lp85,vc33,r11l,s5i5,tn36,su15 --reps 2 --max-actions 2000`
- Result: lp85 1.0 → 2.5 levels (max 4); tn36 0 → 1 level (11 actions, 3.57); ft09 still 0 but now
  explores (153 states vs 1). Diagnosis probe (`click_scan.py ft09`): 72/180 colour-9 lattice
  clicks are effective; the centroid click of the connected tile block is not.
- Verdict: 🐛 fix kept; n=2 dev run, covered by the N=30 run below.

## 2026-09-25 — Floor budget scaling 1000 → 5000 actions (pre-lattice floor)
- Command: `eval/run.py --reps 3 --max-actions 5000` vs `--reps 2 --max-actions 1000`
- Result: 0.415 ± 0.307 (5000) vs 0.546 ± 0.290 (1000) — within noise. Levels barely move.
  Diagnostics: ft09 1 state / lp85 6 states with ~4950 random fallbacks (click stall); g50t 615
  graph conflicts (hidden state); re86/tr87/su15 ~3000 states, 0 new levels (state explosion);
  bp35/sb26 ~38 ms/action (half engine, half `components()`).
- Verdict: INFO — the agent is stuck, not budget-limited. Motivated the click fixes above and
  ROADMAP Phase 2.

## 2026-09-25 — Floor agent first light (smoke + all games, dev)
- Change: `agent/core/` floor agent (masked state graph, learned action model, avatar tracker,
  verified BFS navigation), flattened by `scripts/build_submission.py`.
- Command: `eval/run.py --games all --reps 2 --max-actions 1000`; `eval/run.py --swarm --max-actions 300`
- Result: 0.546 ± 0.290 (n=2) vs random 0.063 at the same budget; ≥1 level on 17/25 games; 0
  crashes. Swarm (25 threads, one process): 0 crashes.
- Verdict: INFO — dev numbers, superseded by the N=30 paired run.

## 2026-09-25 — Random baseline, all 25 games, N=30
- Change: none — starter-kit random agent (experiments/2026-09-25_random_baseline/)
- Command: `eval/run.py --agent experiments/2026-09-25_random_baseline/my_agent.py --reps 30 --max-actions 1000`
- Result: **0.063 ± 0.030** (n=30). sp80 1.19 ± 0.62 (level 1 every run), cd82 0.34, rest ≈0.
  0 crashes. 750 runs in 3m14s on 20 workers.
- Verdict: INFO — the floor for "beats random".

## 2026-09-25 — Interface facts (eval/interface_checks.py)
- Command: `.venv/bin/python eval/interface_checks.py`
- Result: 11/11 match. Bare `GameAction(6)` → ValueError (spec's claim confirmed; my first
  expectation was wrong); ACTION6 x=64 → ValidationError; illegal ACTION6 on ls20 counts, no
  change; RESET after an action = level reset, RESET with 0 level actions = full reset; initial
  frames deterministic; ACTION6 data shared across threads (both threads read x=50); RESET frames
  can have 2+ layers; post-GAME_OVER actions return empty frames.
- Verdict: INFO — see AGENT.md "Verified interface facts".

## 2026-09-25 — Pipeline: make setup + make verify-local (random starter)
- Command: `make setup`, `make list-games`, `make verify-local`
- Result: 25 environments; verify-local exit 0, ls20/vc33 51 actions each, score 0.0.
- Verdict: INFO — local pipeline OK. Kaggle half (Phase A/B) blocked: rules not accepted
  (`userHasEntered: False`, 403 on submissions/data).
