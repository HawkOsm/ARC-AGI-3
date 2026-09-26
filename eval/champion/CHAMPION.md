# Current champion

`eval/champion/my_agent.py` — **pruned brain** (agent/brain/, World-of-beliefs design), promoted
2026-09-26 by the owner. **Local champion only — not submitted to Kaggle** (owner instruction).

- Evidence: docs/tests/LOG.md "Pruned agent (after ablation)": vs the full pre-prune build
  +0.019 ± 0.013 score (significant); vs the previous champion -0.024 ± 0.106 score, +0.87 ± 1.09
  levels (a statistical tie at N=30, all 25 games, 2000 actions). Promoted on the owner's call for
  the robustness it adds: memory guard, background-colour avatar fix, timeout detection.
- Known regression vs the previous champion: sp80 (lost the new-position/quick-test gain).
- CI baseline: `smoke_baseline.json` (ls20+vc33, 3 reps, 400 actions, seed 12345) — re-baselined.
- Previous champion (floor v1, public LB 0.44): `experiments/2026-09-25_floor_v1/my_agent.py`.

To replace it: `make eval-paired REPS=30 BUDGET=2000` must show a significant paired gain with no
meaningful per-game regression (AGENT.md rule 3); then copy, update this file, and run
`eval/regression_check.py --update`.
