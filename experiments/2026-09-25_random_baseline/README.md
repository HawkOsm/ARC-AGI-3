# Random baseline (starter kit)

The starter kit's original `agent/my_agent.py`, frozen verbatim: uniform over *all* GameActions
(ignores `available_actions`), random ACTION6 coordinates, `MAX_ACTIONS = 80`, RESET on
GAME_OVER. Used for (1) the Kaggle pipeline sanity submission and (2) the "beats random" check.

Result: see docs/tests/LOG.md (2026-09-25 entries). At a 1000-action cap it scores
0.063 ± 0.030 (N=30); nearly all of it is sp80 level 1, which random clears every run.
