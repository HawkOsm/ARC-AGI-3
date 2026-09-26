"""BRAIN — the loop. One Brain plays one game.

Every step (Brain.next_action):
    1. LOOK     the frame arrives -> settled picture
    2. EVENTS   died? level up? new play?  (the level change is the only real reward)
    3. LEARN    first screen of a level? take a first look (look.first_look).
                What did my last action do? update the beliefs in self.world
    4. DECIDE   follow the LLM advisor's plan if one is on (llm.py), otherwise pick the
                next action from the beliefs (decide.py)

Long-term beliefs live in `self.world` (see world.py). Everything else on the Brain
is short-term working memory: what just happened, and the route being followed.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np

from .decide import candidate_actions, choose_next_action
from .goals import goals_from_first_look, learn_goals_from_win, screen_features
from .learn import (
    avatar_position,
    hide_counters,
    is_timeout_death,
    learn_what_touching_does,
    record_action_result,
    update_avatar,
    update_counters,
)
from .llm import advisor, describe, describe_step
from .look import first_look, settled_grid
from .settings import LLM_RECENT_STEPS
from .world import (
    DIED,
    LEVEL_UP,
    NO_INFO,
    RESET_ACTION,
    LevelMemory,
    Screen,
    World,
    compress_grid,
    decompress_grid,
)


class Brain:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.world = World()
        # ── working memory ──
        self.level: int | None = None
        self.last_grid: np.ndarray | None = None  # picture before last_action
        self.last_screen_id: int | None = None
        self.screen_before_last_id: int | None = None
        self.last_action: tuple | None = None
        self.came_by: tuple | None = None  # the action that brought us to this screen
        self.route: list = []  # planned [(action, screen it should lead to), ...]
        self.expected_screen = None  # where the route's last step should have taken us
        self.at_level_start = False
        self.life_length = 0  # actions since the last reset / level start
        self.pixel_scans: dict[int, int] = {}  # level -> screens given a 1-pixel click scan
        self.plan: list = []  # the LLM advisor's remaining actions
        self.last_from_advisor = False
        self.ask_again_at = 0  # after an empty answer the brain plays a few steps first
        self.recent: deque = deque(maxlen=LLM_RECENT_STEPS)  # "ACTIONn -> what happened", for the advisor
        self.stats: dict[str, int] = defaultdict(int)  # diagnostics only

    # ── public ──────────────────────────────────────────────────────────────

    def next_action(self, state: str, frame_layers, levels_completed: int, available: list[int],
                    full_reset: bool = False) -> tuple:
        """The whole agent: given the latest frame, return (action_id, x, y)."""
        action = self._think(state, frame_layers, levels_completed, available, full_reset)
        self.stats["actions"] += 1
        self.life_length = 0 if action == RESET_ACTION else self.life_length + 1
        return action

    def forget_last_action(self) -> None:
        """The last action never reached the game (network error): learn nothing from it."""
        self.last_action = None
        self.route, self.expected_screen = [], None

    def restart_level(self) -> tuple:
        """RESET. Only ever sent after GAME_OVER or after steps were taken in this life:
        a RESET with 0 steps taken in the level restarts the WHOLE GAME."""
        self.world.avatar["under"] = None
        self.route, self.expected_screen, self.plan = [], None, []
        self.recent.append("RESET -> the level restarts")
        self.last_action, self.last_grid, self.came_by = RESET_ACTION, None, None
        self.at_level_start = True
        return RESET_ACTION

    # ── the loop ────────────────────────────────────────────────────────────

    def _think(self, state, frame_layers, level, available, full_reset) -> tuple:
        # 1. LOOK
        grid = settled_grid(frame_layers)
        # 2. EVENTS
        if state == "NOT_PLAYED" or (grid is None and state != "GAME_OVER"):
            return self.restart_level()
        if state == "GAME_OVER":
            self._on_death()
            return self.restart_level()
        if full_reset:
            self._on_new_play()
        if self.level is None:
            self.level = level
            self.at_level_start = True
        elif level > self.level or state == "WIN":
            self._on_level_up(level if level > self.level else self.level + 1)
            if state == "WIN":
                return self.restart_level()
        elif level < self.level:
            self._on_new_play()
            self.level = level
            self.at_level_start = True

        # 3. LEARN
        level_map = self.world.levels.setdefault(self.level, LevelMemory())
        if not level_map.items:
            first_look(self.world, self.level, grid)
            goals_from_first_look(self.world, self.level)
        self._learn_from_last_move(grid)
        screen_id = self._screen_id(grid)
        screen = self._remember_screen(level_map, screen_id, grid, available)
        if self.at_level_start:
            if level_map.start_screen is None:
                level_map.start_screen = screen_id
            self.at_level_start = False
        if self.last_action is not None and self.last_action != RESET_ACTION:
            self._record_move(level_map, screen_id)
        if self.route and self.expected_screen is not None and self.expected_screen != screen_id:
            self.stats["route_mismatches"] += 1  # never trust a plan the world disagrees with
            self.route = []
        self._note_last_step(grid, screen_id)

        # 4. DECIDE
        action = self._advice(level_map, grid, available)
        self.last_from_advisor = action is not None
        if action is None:
            action = choose_next_action(self, level_map, screen, available)
        self.screen_before_last_id = self.last_screen_id if self.last_action != RESET_ACTION else None
        self.last_grid, self.last_screen_id = grid, screen_id
        self.last_action = action
        return action

    def _note_last_step(self, grid: np.ndarray, screen_id: int) -> None:
        """History line for the advisor; an advised action that changed nothing ends its plan."""
        if self.last_action is None or self.last_action == RESET_ACTION or self.last_grid is None:
            return
        self.recent.append(describe_step(self.last_action, hide_counters(self.world, self.last_grid),
                                         hide_counters(self.world, grid), None))
        if self.last_from_advisor and screen_id == self.last_screen_id and self.plan:
            self.stats["llm_plans_dropped"] += 1  # the plan expected something; nothing happened
            self.plan = []

    def _advice(self, level_map: LevelMemory, grid: np.ndarray, available: list[int]) -> tuple | None:
        """The advisor's next action, or None (advisor off, level already solved, no usable answer)."""
        adviser = advisor()
        if adviser is None or level_map.has_solution:
            self.plan = []
            return None
        if not self.plan and adviser.available() and self.stats["actions"] >= self.ask_again_at:
            self.stats["llm_questions"] += 1
            self.plan = adviser.propose(describe(self, grid, available), available)
            if self.plan:
                self.stats["llm_plans"] += 1
            else:
                self.ask_again_at = self.stats["actions"] + 4
        while self.plan:
            action = self.plan.pop(0)
            if action[0] in available:
                self.route, self.expected_screen = [], None
                self.stats["llm_actions"] += 1
                return action
        return None

    # ── learning from the last move ─────────────────────────────────────────

    def _learn_from_last_move(self, grid: np.ndarray) -> None:
        """Update counters (and re-identify all screens if they grew) and the avatar."""
        if self.last_grid is None or self.last_action in (None, RESET_ACTION):
            return
        before = hide_counters(self.world, self.last_grid)
        if update_counters(self.world, self.last_grid, grid):
            self.stats["counter_mask_updates"] += 1
            for level_map in self.world.levels.values():
                level_map.rename_screens(self._screen_id_from_snapshot)
            self.last_screen_id = self._screen_id(self.last_grid)
            self.screen_before_last_id = None
            self.route, self.expected_screen = [], None
            before = hide_counters(self.world, self.last_grid)
        after = hide_counters(self.world, grid)
        update_avatar(self.world, before, after, self.last_action[0])
        learn_what_touching_does(self.world, before, after, self.last_action[0])

    def _record_move(self, level_map: LevelMemory, screen_id: int) -> None:
        """Last screen + last action -> this screen. Also teaches world.actions."""
        previous = level_map.screens.get(self.last_screen_id)
        if previous is None:
            return
        if not level_map.record_result(self.last_screen_id, self.last_action, screen_id):
            self.stats["contradictions"] += 1
            self.route = []
        changed = screen_id != self.last_screen_id
        undid = None
        if self.screen_before_last_id is not None and self.screen_before_last_id != self.last_screen_id:
            undid = screen_id == self.screen_before_last_id
        record_action_result(self.world, self.last_action, previous.candidates.get(self.last_action, NO_INFO),
                             changed, False, previous.avatar_at, undid, self.came_by)
        self.came_by = self.last_action if changed else self.came_by

    # ── screen identity ─────────────────────────────────────────────────────

    def _screen_id(self, grid: np.ndarray) -> int:
        """What makes a screen THIS screen: the picture minus counters."""
        return hash(hide_counters(self.world, grid).tobytes())

    def _screen_id_from_snapshot(self, snapshot: bytes) -> int:
        return self._screen_id(decompress_grid(snapshot, self.world.counters["shape"]))

    def _remember_screen(self, level_map: LevelMemory, screen_id: int, grid: np.ndarray,
                         available: list[int]) -> Screen:
        screen = level_map.screens.get(screen_id)
        if screen is not None:
            return screen
        candidates = candidate_actions(self.world, grid, available)
        screen = Screen(screen_id, candidates, avatar_position(self.world, hide_counters(self.world, grid)),
                        compress_grid(grid))
        screen.features = screen_features(self.world, grid)
        level_map.screens[screen_id] = screen
        self.stats["screens"] += 1
        return screen

    # ── events ──────────────────────────────────────────────────────────────

    def _on_death(self) -> None:
        self.stats["deaths"] += 1
        self.plan = []
        if self.last_action not in (None, RESET_ACTION):
            self.recent.append(describe_step(self.last_action, None, None, "you lost a life; the level restarts"))
        level_map = self.world.levels.get(self.level)
        if level_map is None or self.last_screen_id is None or self.last_action in (None, RESET_ACTION):
            return
        previous = level_map.screens.get(self.last_screen_id)
        deaths_before = len(self.world.deaths)
        if is_timeout_death(self.world, self.level, self.life_length):
            self.stats["timeouts"] += 1 + deaths_before - len(self.world.deaths)
            return
        level_map.record_result(self.last_screen_id, self.last_action, DIED)
        if self.last_grid is not None:
            learn_what_touching_does(self.world, hide_counters(self.world, self.last_grid), None,
                                     self.last_action[0], outcome="died")
        if previous is not None:
            record_action_result(self.world, self.last_action, previous.candidates.get(self.last_action, NO_INFO),
                                 True, True, previous.avatar_at, None, None)
            self.world.deaths.append((self.level, self.life_length, previous, self.last_action, previous.avatar_at))

    def _on_level_up(self, new_level: int) -> None:
        self.stats["level_ups"] += 1
        self.plan = []
        if self.last_action not in (None, RESET_ACTION):
            self.recent.append(describe_step(self.last_action, None, None, "LEVEL COMPLETE; a new level starts"))
        self.life_length = 0
        level_map = self.world.levels.get(self.level)
        if level_map is not None and self.last_screen_id is not None \
                and self.last_action not in (None, RESET_ACTION):
            level_map.record_result(self.last_screen_id, self.last_action, LEVEL_UP)
            if self.last_grid is not None:
                learn_what_touching_does(self.world, hide_counters(self.world, self.last_grid), None,
                                         self.last_action[0], outcome="level_up")
            winning_screen = level_map.screens.get(self.last_screen_id)
            if winning_screen is not None:
                learn_goals_from_win(self.world, self.level, level_map, winning_screen)
                self.stats["goals_known"] = len(self.world.goals)
        self.level = new_level
        self.world.avatar["under"] = None
        self.last_action, self.last_grid, self.came_by = None, None, None
        self.route, self.expected_screen = [], None
        self.at_level_start = True

    def _on_new_play(self) -> None:
        """A full reset happened (back to level 1)."""
        self.plan = []
        self.level = 0
        self.at_level_start = True
