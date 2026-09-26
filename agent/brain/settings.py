"""Every tunable number the agent uses, in one place.

A "chance" below is the agent's estimate that trying an action will show it a
screen it has not seen before. Numbers that turned out to matter have a comment
saying why (see also AGENT.md "Load-bearing constraints").
"""

# ── Counters / HUD (world.counters) ─────────────────────────────────────────
# Every public game draws a step counter that changes 1-3 pixels on EVERY action.
# If the agent looked at those pixels, every screen would look new, so they are
# learned and ignored:
COUNTER_TICK_MAX_CELLS = 4  # a change this small ...
COUNTER_BORDER_BAND = 3  # ... this close to the frame edge is a counter "tick"
# A row/column is a counter track (ignored in full) once ticks landed on it at
# this many different positions forming one near-contiguous run. Counting
# positions, not cells, stops a 2-row counter (ls20) or two mirrored counters
# (m0r0) from hiding whole columns of the board. Checked by eval/mask_check.py.
COUNTER_TRACK_MIN_POSITIONS = 3
COUNTER_TRACK_MAX_GAP = 2
# Blinkers away from the border: a cell that changes in most transitions.
BLINKER_MIN_TRANSITIONS = 20
BLINKER_CHANGE_RATE = 0.8

# ── Click candidates (ACTION6) ──────────────────────────────────────────────
MAX_CLICKS_PER_SCREEN = 96  # ranked by prior chance, the rest dropped
LARGE_OBJECT_CELLS = 400  # objects bigger than this are probably background
# The clickable unit is not always a whole object (ft09: a connected block of
# 6x6 tiles only reacts on some cells), so big objects also get extra click
# points on a grid inside them.
CLICK_GRID_MIN_OBJECT_CELLS = 24
CLICK_GRID_STEP = 4
CLICK_GRID_MAX_PER_OBJECT = 12
CLICK_GRID_PRIOR_FACTOR = 0.8
# When a screen has nothing left to try, click it on a whole-screen grid, finer
# each time. Step 1 (4096 clicks, ~60 KB of memory) only on a few screens per level.
EXHAUSTED_SCREEN_CLICK_STEPS = (4, 2, 1)
MAX_PIXEL_SCANS_PER_LEVEL = 8

# ── Prior chances (before the agent has any evidence) ───────────────────────
PRIOR_KEY_ACTION = 0.5  # ACTION1-5
PRIOR_ACTION7 = 0.3  # often undo; lower until characterised
PRIOR_CLICK_SMALL = 0.5
PRIOR_CLICK_SINGLE_PIXEL = 0.3
PRIOR_CLICK_LARGE = 0.1
PRIOR_WEIGHT = 2.0  # how many observations the prior is worth

# ── What the agent learns about actions (world.actions) ─────────────────────
UNDO_MIN_TRIES = 3
UNDO_RATE = 0.8  # returns to the screen before last this often -> "undo"
REVERSE_PAIR_MIN_TRIES = 3
REVERSE_PAIR_RATE = 0.9  # b right after a returns to the earlier screen this often -> b reverses a
CHANCE_IF_KNOWN_USELESS = 0.05  # known no-op / undo / reversal here
CHANCE_IF_KNOWN_DEADLY = 0.01  # killed us from this avatar position before

# ── Lives (world.rules["steps_per_life"]) ───────────────────────────────────
# Running out the step counter is GAME_OVER whatever the last action was. Two
# deaths in a level at the same life length (+- tolerance) reveal that budget;
# those deaths are then not blamed on their last action.
TIMEOUT_TOLERANCE = 1  # at least this ...
TIMEOUT_TOLERANCE_FRACTION = 0.05  # ... growing with life length (dc22 timed out at 82-84)
TIMEOUT_MIN_LIFE = 8

# ── The avatar (world.avatar) ───────────────────────────────────────────────
AVATAR_MIN_OBSERVATIONS = 6
AVATAR_REFIT_EVERY = 5  # re-check which colour is the avatar every N moves
AVATAR_MIN_CONSISTENCY = 0.6

# ── First look (look.first_look) ─────────────────────────────────────────────
BACKGROUND_FRACTION = 0.15  # an item covering more of the screen than this is probably background
PATTERN_MIN_ITEM_CELLS = 4  # ignore specks when looking for patterns
MAX_PATTERNS_PER_LEVEL = 8

# ── Touching things (world.colour_roles) ────────────────────────────────────
# One bump into a colour teaches that the colour blocks; after ROLE_MIN_EVIDENCE
# consistent observations the agent stops trying to walk into that colour anywhere.
ROLE_MIN_EVIDENCE = 2
ROLE_MIN_SHARE = 0.6  # ... and the role must be this share of all evidence (mixed evidence = "?")
STEP_MIN_OBSERVATIONS = 2  # a key's usual step must be seen this often before predicting with it
AVATAR_MAX_CELLS = 400  # avatar colour covering more than this is too ambiguous to reason about
# A colour that moves with the avatar on at least this share of its moves is part of the
# avatar (multi-colour sprite: ls20's colour 9), not a pushable object.
AVATAR_PART_MIN_MOVES = 5
AVATAR_PART_RATE = 0.6  # wa30's colour-0 top edge co-moved on ~73% of moves (contact edge cases)
AVATAR_PART_SEARCH_MARGIN = 6  # parts must be within this many cells of the main colour

# ── Goals (goals.py) — decided by the game: what changed when a level was won ─
GOALS_PER_WIN = 3  # new hypotheses taken from each win (the biggest relative changes)
GOAL_MIN_RELATIVE_CHANGE = 0.3  # smaller changes between level start and win are ignored
GOAL_MIN_CONFIDENCE = 0.4  # hypotheses below this don't steer exploration
GOAL_CUE_CONFIDENCE = 0.45  # first-look cues (before any win) start just above that
GOAL_WEIGHT = 1.0  # frontier cost is divided by 1 + GOAL_WEIGHT * progress toward goals

# ── Choosing where to explore ───────────────────────────────────────────────
# Target = (screen, untried action) with the fewest expected actions per new
# screen: (walking distance + 1) / chance. (An earlier `distance + 10*(1-chance)`
# capped the cost of near-certain no-ops and burned 46% of sp80's actions on clicks.)
MIN_CHANCE = 0.005
TAKE_LOCAL_IF_CHANCE_AT_LEAST = 0.35  # skip the search if something here is this promising

# ── The LLM advisor (llm.py) ────────────────────────────────────────────────
# Off unless the environment variable ARC3_LLM_URL points at an OpenAI-compatible server.
LLM_MODEL = "qwen"  # the served model name (vllm serve --served-model-name)
# Thinking mode is off by default: on the RTX PRO 6000 with 25 games asking at once, 38/38 thinking
# answers were still thinking at 6,144 tokens (234 s each, ~26 tokens/s per request). Each mode can be
# chosen per run with ARC3_LLM_THINKING=0/1 (and ARC3_LLM_MAX_TOKENS / ARC3_LLM_TIMEOUT_S).
LLM_THINKING = False
LLM_MAX_TOKENS = 1024  # answer length without thinking
LLM_MAX_TOKENS_THINKING = 16384  # thinking + answer
LLM_TIMEOUT_S = 180  # one request; slower answers are dropped and the brain decides
LLM_TIMEOUT_THINKING_S = 1200
# Qwen3.8's recommended sampling (model README), per mode
LLM_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
LLM_SAMPLING_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
LLM_PLAN_MAX = 8  # actions taken from one answer before asking again
LLM_RECENT_STEPS = 16  # recent steps (action -> what happened) shown to the model
LLM_FAILS_BEFORE_PAUSE = 3  # failed requests in a row (any game) ...
LLM_PAUSE_S = 300  # ... pause the advisor for everyone this long

# ── Runtime guards ──────────────────────────────────────────────────────────
MAX_ACTIONS = 1_000_000  # framework per-game cap; the wall-clock guard binds first
TIME_BUDGET_S = 7.5 * 3600  # from process start; Kaggle notebook limit is 9 h
KEEP_FRAMES = 4  # the framework otherwise keeps every frame (~GBs per game)
MAX_CONSECUTIVE_ERRORS = 50  # failed steps in a row before giving up on a game
# All games share one process on Kaggle and an out-of-memory kill zeroes the
# whole submission: above MEM_GUARD_MB of process memory, games whose maps hold
# more than MEM_GUARD_SCREENS screens stop playing.
MEM_GUARD_MB = 20_000
MEM_GUARD_SCREENS = 20_000
