# Current champion

`eval/champion/my_agent.py` — floor agent, promoted 2026-09-25.

- Source: `agent/core/` at promotion time (flattened by `scripts/build_submission.py`).
- Evidence: docs/tests/LOG.md "PROMOTED: floor agent vs random starter" — 0.628 ± 0.106 vs
  0.063 ± 0.030, paired +0.565 ± 0.110, N=30, all 25 games, 2000-action cap, 0 crashes.
- Known regression vs random: sp80 (−1.18).
- CI baseline: `smoke_baseline.json` (ls20+vc33, 3 reps, 400 actions, seed 12345).

To replace it: `make eval-paired REPS=30 BUDGET=2000` must show a significant paired gain with no
meaningful per-game regression (AGENT.md rule 3); then copy, update this file, and run
`eval/regression_check.py --update`.
