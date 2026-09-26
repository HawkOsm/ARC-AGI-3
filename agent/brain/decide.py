"""DECIDE — read the beliefs and pick the next action.

    choose_next_action(brain, level_map, screen, available)
        1. level already solved       -> follow the shortest known route to LEVEL_UP
        2. otherwise                  -> plan_route(): the cheapest untried thing to try
        3. nothing left anywhere      -> a random legal action
    plan_route(...)
        a. the best untried action right here, if promising enough
        b. search the map for the (screen, untried action) with the fewest expected
           actions per new screen: (walking distance + 1) / chance, discounted for screens
           that moved toward the game's goals (goals.py) - but only screens reachable
           before this life's step counter runs out
        c. nothing reachable this life   -> RESET the level (safe: only when steps were taken)
        d. screen exhausted              -> add a whole-screen click grid and try again
    chance_of_new_screen(world, screen, action, came_by)   the "chance" used everywhere above
    candidate_actions(world, grid, available)               what can be tried on a new screen
"""
from __future__ import annotations

import numpy as np

from .goals import goal_progress
from .learn import hide_counters
from .look import find_objects
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
from .world import NO_INFO, RESET_ACTION, LevelMemory, Screen, World, decompress_grid

# ── What to do next ─────────────────────────────────────────────────────────


def choose_next_action(brain, level_map: LevelMemory, screen: Screen, available: list[int]) -> tuple:
    if level_map.has_solution and not brain.route:
        route = level_map.route_to_level_up(screen.id)
        if route:
            brain.route = route
    if not brain.route:
        brain.route = plan_route(brain, level_map, screen, available)
    if brain.route:
        action, brain.expected_screen = brain.route.pop(0)
        if action == RESET_ACTION:
            return brain.restart_level()
        if action[0] in available:
            return action
        brain.route = []
    brain.expected_screen = None
    brain.stats["random_fallbacks"] += 1
    return random_action(brain, screen, available)


def plan_route(brain, level_map: LevelMemory, screen: Screen, available: list[int]) -> list:
    """Route [(action, expected screen or None), ...] to the most promising untried action."""
    world = brain.world
    budget = world.rules["steps_per_life"].get(brain.level)
    steps_left = float("inf") if budget is None else budget - 1 - brain.life_length
    here = best_untried_action(brain, screen, brain.came_by, available)
    if here is not None and here[0] >= TAKE_LOCAL_IF_CHANCE_AT_LEAST and steps_left >= 1:
        return [(here[1], None)]
    best_cost, best, out_of_reach = float("inf"), None, False
    if here is not None:
        if steps_left >= 1:
            best_cost, best = _route_cost(0, here[0]), (screen.id, here[1])
        else:
            out_of_reach = True
    distance, came_from = level_map.walking_distances(screen.id, stop_beyond=best_cost - 1)
    for other_id, d in distance.items():
        if d == 0 or d + 1 >= best_cost:  # cost is at least d + 1
            continue
        came_by = came_from[other_id][1] if came_from[other_id] else None
        option = best_untried_action(brain, level_map.screens[other_id], came_by, available)
        if option is None:
            continue
        if d + 1 > steps_left:
            out_of_reach = True
            continue
        cost = _route_cost(d, option[0]) * (1 + brain.rng.random() * 1e-3)
        if world.goals:  # screens that moved toward what the game rewards are explored first
            cost /= 1 + GOAL_WEIGHT * goal_progress(world, level_map, level_map.screens[other_id])
        if cost < best_cost:
            best_cost, best = cost, (other_id, option[1])
    if best is None:
        if out_of_reach and brain.life_length > 0:  # something left, but not in this life
            brain.stats["voluntary_resets"] += 1
            return [(RESET_ACTION, None)]
        if 6 in available and add_click_grid(brain, screen):
            return plan_route(brain, level_map, screen, available)
        return []
    target_id, action = best
    walk = [] if target_id == screen.id else LevelMemory.route_to(came_from, target_id)
    return walk + [(action, None)]


def _route_cost(distance: int, chance: float) -> float:
    """Expected actions per new screen: walk `distance`, then try something with `chance`."""
    return (distance + 1) / max(chance, MIN_CHANCE)


def best_untried_action(brain, screen: Screen, came_by, available: list[int]):
    """(chance, action) of the most promising untried legal action on `screen`, or None."""
    best = None
    for action in screen.untried():
        if action[0] not in available:
            continue
        chance = chance_of_new_screen(brain.world, screen, action, came_by) + brain.rng.random() * 1e-6
        if best is None or chance > best[0]:
            best = (chance, action)
    return best


def random_action(brain, screen: Screen, available: list[int]) -> tuple:
    options = [a for a in screen.candidates if a[0] in available] or [(a, -1, -1) for a in available if a != 6]
    if not options:
        return (6, brain.rng.randrange(64), brain.rng.randrange(64)) if 6 in available else RESET_ACTION
    return brain.rng.choice(options)


# ── Estimating chances from world.actions ───────────────────────────────────


def chance_of_new_screen(world: World, screen: Screen, action: tuple, came_by: tuple | None) -> float:
    """Estimated chance that trying `action` on `screen` shows a screen we haven't seen."""
    action_id = action[0]
    prior, colour = screen.candidates.get(action, NO_INFO)
    a = world.actions
    if action_id != 6:
        if screen.avatar_at is not None:
            if a["died_at"].get((screen.avatar_at, action_id)):
                return CHANCE_IF_KNOWN_DEADLY
            if a["no_op_at"].get((screen.avatar_at, action_id)):
                return CHANCE_IF_KNOWN_USELESS
        if is_undo(world, action_id):
            return CHANCE_IF_KNOWN_USELESS
        if came_by is not None and came_by[0] != 6 and reverses(world, came_by[0], action_id):
            return CHANCE_IF_KNOWN_USELESS
        tally = a["changed_screen"].get(action_id)
        return tally.chance(prior) if tally else prior
    tally = a["click_colour_changed_screen"].get(colour)
    return tally.chance(prior) if tally else prior


def is_undo(world: World, action_id: int) -> bool:
    t = world.actions["undid_last_move"].get(action_id)
    return t is not None and t.tries >= UNDO_MIN_TRIES and t.hits / t.tries >= UNDO_RATE


def reverses(world: World, previous_id: int, action_id: int) -> bool:
    t = world.actions["reversed_after"].get((previous_id, action_id))
    return t is not None and t.tries >= REVERSE_PAIR_MIN_TRIES and t.hits / t.tries >= REVERSE_PAIR_RATE


# ── What can be tried on a screen ───────────────────────────────────────────


def candidate_actions(world: World, grid: np.ndarray, available: list[int]) -> dict:
    """{action: (prior chance, colour)} for a newly seen screen, most promising first:
    every legal key action, plus clicks on objects (and on a grid inside big objects)."""
    candidates = {}
    for action_id in sorted(available):
        if action_id == 6:
            continue
        candidates[(action_id, -1, -1)] = (PRIOR_ACTION7 if action_id == 7 else PRIOR_KEY_ACTION, -1)
    if 6 in available:
        clicks = []  # (prior, object size, (x, y), colour)
        picture = hide_counters(world, grid)
        visible = ~world.counters["ignore"]
        for obj in find_objects(picture, visible=visible):
            prior = _click_prior(obj)
            clicks.append((prior, obj["size"], (obj["centre_cell"][1], obj["centre_cell"][0]), obj["colour"]))
            if obj["size"] > CLICK_GRID_MIN_OBJECT_CELLS:
                cells = obj["cells"]
                inner = cells[(cells[:, 0] % CLICK_GRID_STEP == CLICK_GRID_STEP // 2)
                              & (cells[:, 1] % CLICK_GRID_STEP == CLICK_GRID_STEP // 2)]
                if len(inner) > CLICK_GRID_MAX_PER_OBJECT:
                    inner = inner[np.linspace(0, len(inner) - 1, CLICK_GRID_MAX_PER_OBJECT).astype(int)]
                for y, x in inner.tolist():
                    clicks.append((prior * CLICK_GRID_PRIOR_FACTOR, obj["size"], (x, y), obj["colour"]))
        clicks.sort(key=lambda t: (-t[0], t[1]))
        for prior, _size, (x, y), colour in clicks[:MAX_CLICKS_PER_SCREEN]:
            action = (6, x, y)
            if action not in candidates:
                candidates[action] = (prior, colour)
    return candidates


def add_click_grid(brain, screen: Screen) -> bool:
    """Screen exhausted: add clicks on a whole-screen grid (step 4, then 2, then 1)."""
    if screen.click_scans >= len(EXHAUSTED_SCREEN_CLICK_STEPS):
        return False
    step = EXHAUSTED_SCREEN_CLICK_STEPS[screen.click_scans]
    if step == 1:  # 4096 clicks - memory-heavy, only a few screens per level
        if brain.pixel_scans.get(brain.level, 0) >= MAX_PIXEL_SCANS_PER_LEVEL:
            screen.click_scans = len(EXHAUSTED_SCREEN_CLICK_STEPS)
            return False
        brain.pixel_scans[brain.level] = brain.pixel_scans.get(brain.level, 0) + 1
    screen.click_scans += 1
    ignore = brain.world.counters["ignore"]
    grid = decompress_grid(screen.snapshot, ignore.shape)
    added = 0
    for y in range(step // 2, grid.shape[0], step):
        for x in range(step // 2, grid.shape[1], step):
            action = (6, x, y)
            if action in screen.candidates or ignore[y, x]:
                continue
            screen.candidates[action] = (PRIOR_CLICK_LARGE, int(grid[y, x]))
            added += 1
    brain.stats["click_grids"] += 1
    return added > 0 or add_click_grid(brain, screen)


def _click_prior(obj: dict) -> float:
    if obj["size"] > LARGE_OBJECT_CELLS:
        return PRIOR_CLICK_LARGE
    if obj["size"] == 1:
        return PRIOR_CLICK_SINGLE_PIXEL
    return PRIOR_CLICK_SMALL
