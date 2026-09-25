"""Tunable constants for the agent. Every value here is a knob, and every knob
that turns out to be load-bearing gets a sentence in AGENT.md explaining why.
"""

# ── Perception ──────────────────────────────────────────────────────────────
# A diff component this small, lying within BORDER_BAND cells of the frame
# edge, is treated as HUD "tick" evidence (step counters / energy bars change
# 1-3 pixels on the border on EVERY action; see docs/GAME_GUIDE.md).
TICK_MAX_CELLS = 4
BORDER_BAND = 3
# A border line (row or column) with this many tick-evidence cells is masked
# in full, so counter cells that have not ticked yet are masked in advance.
TICK_LINE_MIN_CELLS = 2
# Per-cell frequency masking for blinkers/animations away from the border.
FREQ_MASK_MIN_TRANSITIONS = 20
FREQ_MASK_RATE = 0.8

# ── Click targets (ACTION6) ─────────────────────────────────────────────────
MAX_CLICK_TARGETS = 96  # per state; targets ranked by prior, rest dropped
LARGE_COMPONENT = 400  # components bigger than this are likely background
# Components are not always the clickable unit (ft09: a connected block of 6x6
# tiles only reacts on some cells). Components bigger than LATTICE_MIN_SIZE
# also get up to LATTICE_PER_COMPONENT extra points on a LATTICE_STRIDE grid.
LATTICE_MIN_SIZE = 24
LATTICE_STRIDE = 4
LATTICE_PER_COMPONENT = 12
LATTICE_PRIOR_FACTOR = 0.8
# When no untried action is reachable, the current state gets whole-grid click
# points at these strides, one refinement per exhaustion.
REFINE_STRIDES = (4, 2, 1)

# ── Action priors (probability an untried action changes the state) ─────────
PRIOR_SIMPLE = 0.5
PRIOR_UNDO_CANDIDATE = 0.3  # ACTION7 before it is characterised
PRIOR_CLICK_SMALL = 0.5
PRIOR_CLICK_SINGLE_PIXEL = 0.3
PRIOR_CLICK_LARGE = 0.1
PRIOR_WEIGHT = 2.0  # pseudo-count weight of the prior in Laplace-style updates

# ── Interface learning ──────────────────────────────────────────────────────
UNDO_MIN_TRIES = 3
UNDO_RATE = 0.8  # action reverts the previous transition this often -> undo
PAIR_UNDO_MIN_TRIES = 3
PAIR_UNDO_RATE = 0.9  # b right after a returns to the prior state this often
P_DEMOTED = 0.05  # score for actions predicted to be no-op / undo
P_DEATH = 0.01  # score for actions predicted to kill

# ── Avatar tracking ─────────────────────────────────────────────────────────
AVATAR_MIN_OBS = 6  # transitions before an avatar colour is trusted
AVATAR_MIN_CONSISTENCY = 0.6

# ── Frontier selection ──────────────────────────────────────────────────────
# Choose frontier node minimising  distance + FRONTIER_LAMBDA * (1 - best_p):
# willing to walk this many extra steps to go from a p=0 to a p=1 action.
FRONTIER_LAMBDA = 10.0
# Take the best local untried action without searching if its p is at least this.
LOCAL_GREEDY_P = 0.35

# ── Replay (see AGENT.md "max-over-plays") ──────────────────────────────────
# After a level-up, full-reset and replay known shortest solutions when that
# strictly lowers the action count of this play's completed levels. Only
# valid if the scorer takes the max over plays (local scorer does; Kaggle
# gateway unverified).
REPLAY_ENABLED = False

# ── Runtime guards ──────────────────────────────────────────────────────────
MAX_ACTIONS = 1_000_000  # framework per-game cap; the wall-clock guard binds first
TIME_BUDGET_S = 7.5 * 3600  # from module import (process start on Kaggle)
KEEP_FRAMES = 4  # framework keeps every FrameData otherwise (~GBs per game)
MAX_CONSECUTIVE_ERRORS = 50  # failed steps in a row before giving up on a game
