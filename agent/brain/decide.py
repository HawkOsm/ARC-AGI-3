"""DECIDE — read the beliefs and pick the next action.

    Explorer(world, memory, goals, rng, stats, restart)
      .choose(level_map, screen, available)
          1. level already solved       -> follow the shortest known route to LEVEL_UP
          2. otherwise                  -> plan_route(): the cheapest untried thing to try
          3. nothing left anywhere      -> a random legal action
      .plan_route(...)
          a. the best untried action right here, if promising enough
          b. search the map for the (screen, untried action) with the fewest expected
             actions per new screen: (walking distance + 1) / chance, discounted for screens
             that moved toward the game's goals (goals.py) - but only screens reachable
             before this life's step counter runs out
          c. nothing reachable this life   -> RESET the level (safe: only when steps were taken)
          d. screen exhausted              -> add a whole-screen click grid and try again
      .chance_of_new_screen(screen, action, came_by)   the "chance" used everywhere above
      .candidate_actions(grid, available)              what can be tried on a new screen
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from .goals import GoalTracker
from .look import Blob, ScreenReader
from .settings import (
    CHANCE_IF_KNOWN_DEADLY,
    CHANCE_IF_KNOWN_USELESS,
    CLICK_GRID_MAX_PER_OBJECT,
    CLICK_GRID_MIN_OBJECT_CELLS,
    CLICK_GRID_PRIOR_FACTOR,
    CLICK_GRID_STEP,
    EXHAUSTED_SCREEN_CLICK_STEPS,
    GOAL_WEIGHT,
    LARGE_OBJECT_CELLS,
    MAX_CLICKS_PER_SCREEN,
    MAX_PIXEL_SCANS_PER_LEVEL,
    MIN_CHANCE,
    PRIOR_ACTION7,
    PRIOR_CLICK_LARGE,
    PRIOR_CLICK_SINGLE_PIXEL,
    PRIOR_CLICK_SMALL,
    PRIOR_KEY_ACTION,
    REVERSE_PAIR_MIN_TRIES,
    REVERSE_PAIR_RATE,
    TAKE_LOCAL_IF_CHANCE_AT_LEAST,
    UNDO_MIN_TRIES,
    UNDO_RATE,
)
from .world import (
    NO_INFO,
    RESET_ACTION,
    Action,
    ClickInfo,
    LevelMemory,
    Screen,
    WorkingMemory,
    World,
    decompress_grid,
)


class Explorer:
    """Picks the next action from the beliefs. Writes only short-term memory (route) and stats."""

    def __init__(self, world: World, memory: WorkingMemory, goals: GoalTracker, rng: random.Random,
                 stats: defaultdict[str, int], restart: Callable[[], Action]) -> None:
        self.world = world
        self.memory = memory
        self.goals = goals
        self.rng = rng
        self.stats = stats
        self.restart = restart  # Brain.restart_level: RESET with the bookkeeping it needs
        # Debug runs (eval/debug_run.py) set this to [] to see every option plan_route weighed:
        # (cost = expected actions per new screen, walking distance, chance, action). None = off.
        self.considered: list[tuple[float, int, float, Action]] | None = None

    # ── what to do next ─────────────────────────────────────────────────────

    def choose(self, level_map: LevelMemory, screen: Screen, available: list[int]) -> Action:
        m = self.memory
        if level_map.has_solution and not m.route:
            route = level_map.route_to_level_up(screen.id)
            if route:
                m.route = route
                m.route_reason = f"replay the known solution of this level ({len(route)} steps)"
        if not m.route:
            m.route = self.plan_route(level_map, screen, available)
        if m.route:
            action, m.expected_screen = m.route.pop(0)
            m.why = m.route_reason + (f" - {len(m.route)} more step(s) planned" if m.route else "")
            if action == RESET_ACTION:
                return self.restart()
            if action[0] in available:
                return action
            m.route = []
        m.expected_screen = None
        self.stats["random_fallbacks"] += 1
        m.why = "random: nothing left to try that the map knows of"
        return self._random_action(screen, available)

    def plan_route(self, level_map: LevelMemory, screen: Screen, available: list[int]) -> list:
        """Route [(action, expected screen or None), ...] to the most promising untried action."""
        world, m = self.world, self.memory
        budget = world.rules.steps_per_life.get(m.level) if m.level is not None else None
        steps_left = float("inf") if budget is None else budget - 1 - m.life_length
        here = self._best_untried(screen, m.came_by, available)
        if self.considered is not None and here is not None:
            self.considered.append((self._route_cost(0, here[0]), 0, here[0], here[1]))
        if here is not None and here[0] >= TAKE_LOCAL_IF_CHANCE_AT_LEAST and steps_left >= 1:
            m.route_reason = f"explore: best untried action on this screen (chance of a new screen {here[0]:.2f})"
            return [(here[1], None)]
        best_cost, best, out_of_reach = float("inf"), None, False
        if here is not None:
            if steps_left >= 1:
                best_cost, best = self._route_cost(0, here[0]), (screen.id, here[1])
            else:
                out_of_reach = True
        distance, came_from = level_map.walking_distances(screen.id, stop_beyond=best_cost - 1)
        for other_id, d in distance.items():
            if d == 0 or d + 1 >= best_cost:  # cost is at least d + 1
                continue
            came_by = came_from[other_id][1] if came_from[other_id] else None
            option = self._best_untried(level_map.screens[other_id], came_by, available)
            if option is None:
                continue
            if d + 1 > steps_left:
                out_of_reach = True
                continue
            cost = self._route_cost(d, option[0]) * (1 + self.rng.random() * 1e-3)
            if world.goals:  # screens that moved toward what the game rewards are explored first
                cost /= 1 + GOAL_WEIGHT * self.goals.progress(level_map, level_map.screens[other_id])
            if self.considered is not None:
                self.considered.append((cost, d, option[0], option[1]))
            if cost < best_cost:
                best_cost, best = cost, (other_id, option[1])
        if best is None:
            if out_of_reach and m.life_length > 0:  # something left, but not in this life
                self.stats["voluntary_resets"] += 1
                m.route_reason = "reset: the untried actions left can't be reached before this life runs out"
                return [(RESET_ACTION, None)]
            if 6 in available and self._add_click_grid(screen):
                return self.plan_route(level_map, screen, available)
            return []
        target_id, action = best
        walk = [] if target_id == screen.id else LevelMemory.route_to(came_from, target_id)
        chance = 1 / best_cost * (len(walk) + 1)
        m.route_reason = (f"explore: best untried action here (chance ≈{chance:.2f})" if not walk else
                          f"explore: walk {len(walk)} step(s) to a known screen, then try an untried action there "
                          f"(chance ≈{chance:.2f})")
        return walk + [(action, None)]

    @staticmethod
    def _route_cost(distance: int, chance: float) -> float:
        """Expected actions per new screen: walk `distance`, then try something with `chance`."""
        return (distance + 1) / max(chance, MIN_CHANCE)

    def _best_untried(self, screen: Screen, came_by: Action | None, available: list[int]):
        """(chance, action) of the most promising untried legal action on `screen`, or None."""
        best = None
        for action in screen.untried():
            if action[0] not in available:
                continue
            chance = self.chance_of_new_screen(screen, action, came_by) + self.rng.random() * 1e-6
            if best is None or chance > best[0]:
                best = (chance, action)
        return best

    def _random_action(self, screen: Screen, available: list[int]) -> Action:
        options = [a for a in screen.candidates if a[0] in available] \
            or [Action(a, -1, -1) for a in available if a != 6]
        if not options:
            return Action(6, self.rng.randrange(64), self.rng.randrange(64)) if 6 in available else RESET_ACTION
        return self.rng.choice(options)

    # ── estimating chances from world.actions ───────────────────────────────

    def chance_of_new_screen(self, screen: Screen, action: Action, came_by: Action | None) -> float:
        """Estimated chance that trying `action` on `screen` shows a screen we haven't seen."""
        action_id = action[0]
        prior, colour = screen.candidates.get(action, NO_INFO)
        a = self.world.actions
        if action_id != 6:
            if screen.avatar_at is not None:
                if a.died_at.get((screen.avatar_at, action_id)):
                    return CHANCE_IF_KNOWN_DEADLY
                if a.no_op_at.get((screen.avatar_at, action_id)):
                    return CHANCE_IF_KNOWN_USELESS
            if self._is_undo(action_id):
                return CHANCE_IF_KNOWN_USELESS
            if came_by is not None and came_by[0] != 6 and self._reverses(came_by[0], action_id):
                return CHANCE_IF_KNOWN_USELESS
            tally = a.changed_screen.get(action_id)
            return tally.chance(prior) if tally else prior
        click_tally = a.click_colour_changed_screen.get(colour)
        return click_tally.chance(prior) if click_tally else prior

    def _is_undo(self, action_id: int) -> bool:
        t = self.world.actions.undid_last_move.get(action_id)
        return t is not None and t.tries >= UNDO_MIN_TRIES and t.hits / t.tries >= UNDO_RATE

    def _reverses(self, previous_id: int, action_id: int) -> bool:
        t = self.world.actions.reversed_after.get((previous_id, action_id))
        return t is not None and t.tries >= REVERSE_PAIR_MIN_TRIES and t.hits / t.tries >= REVERSE_PAIR_RATE

    # ── what can be tried on a screen ───────────────────────────────────────

    def candidate_actions(self, grid: np.ndarray, available: list[int]) -> dict[Action, ClickInfo]:
        """{action: ClickInfo} for a newly seen screen, most promising first:
        every legal key action, plus clicks on objects (and on a grid inside big objects)."""
        world = self.world
        candidates: dict[Action, ClickInfo] = {}
        for action_id in sorted(available):
            if action_id == 6:
                continue
            candidates[Action(action_id, -1, -1)] = ClickInfo(
                PRIOR_ACTION7 if action_id == 7 else PRIOR_KEY_ACTION, -1)
        if 6 in available:
            clicks = []  # (prior, object size, (x, y), colour)
            picture = world.counters.hide(grid)
            visible = ~world.counters.ignore
            for obj in ScreenReader.find_objects(picture, visible=visible):
                prior = self._click_prior(obj)
                clicks.append((prior, obj.size, (obj.centre_cell[1], obj.centre_cell[0]), obj.colour))
                if obj.size > CLICK_GRID_MIN_OBJECT_CELLS:
                    cells = obj.cells
                    inner = cells[(cells[:, 0] % CLICK_GRID_STEP == CLICK_GRID_STEP // 2)
                                  & (cells[:, 1] % CLICK_GRID_STEP == CLICK_GRID_STEP // 2)]
                    if len(inner) > CLICK_GRID_MAX_PER_OBJECT:
                        inner = inner[np.linspace(0, len(inner) - 1, CLICK_GRID_MAX_PER_OBJECT).astype(int)]
                    for y, x in inner.tolist():
                        clicks.append((prior * CLICK_GRID_PRIOR_FACTOR, obj.size, (x, y), obj.colour))
            clicks.sort(key=lambda t: (-t[0], t[1]))
            for prior, _size, (x, y), colour in clicks[:MAX_CLICKS_PER_SCREEN]:
                action = Action(6, x, y)
                if action not in candidates:
                    candidates[action] = ClickInfo(prior, colour)
        return candidates

    def _add_click_grid(self, screen: Screen) -> bool:
        """Screen exhausted: add clicks on a whole-screen grid (step 4, then 2, then 1)."""
        m = self.memory
        if screen.click_scans >= len(EXHAUSTED_SCREEN_CLICK_STEPS):
            return False
        step = EXHAUSTED_SCREEN_CLICK_STEPS[screen.click_scans]
        level = m.level if m.level is not None else 0
        if step == 1:  # 4096 clicks - memory-heavy, only a few screens per level
            if m.pixel_scans.get(level, 0) >= MAX_PIXEL_SCANS_PER_LEVEL:
                screen.click_scans = len(EXHAUSTED_SCREEN_CLICK_STEPS)
                return False
            m.pixel_scans[level] = m.pixel_scans.get(level, 0) + 1
        screen.click_scans += 1
        ignore = self.world.counters.ignore
        grid = decompress_grid(screen.snapshot, ignore.shape)
        added = 0
        for y in range(step // 2, grid.shape[0], step):
            for x in range(step // 2, grid.shape[1], step):
                action = Action(6, x, y)
                if action in screen.candidates or ignore[y, x]:
                    continue
                screen.candidates[action] = ClickInfo(PRIOR_CLICK_LARGE, int(grid[y, x]))
                added += 1
        self.stats["click_grids"] += 1
        return added > 0 or self._add_click_grid(screen)

    @staticmethod
    def _click_prior(obj: Blob) -> float:
        if obj.size > LARGE_OBJECT_CELLS:
            return PRIOR_CLICK_LARGE
        if obj.size == 1:
            return PRIOR_CLICK_SINGLE_PIXEL
        return PRIOR_CLICK_SMALL
