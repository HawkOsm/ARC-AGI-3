"""WORLD — everything the agent believes about the game it is playing.

It starts almost empty and fills up as the agent plays. Code never changes per
game; only these beliefs do. `learn.py` writes them, `decide.py` reads them.

    world.counters      which pixels are HUD step-counters (ignored when comparing screens)
    world.avatar        which colour is the player, and how far each key moves it
    world.kinds         every distinct item type seen (colour + shape) and its purpose ("?" until learned)
    world.colour_roles  what each colour does when the player touches it: floor, wall, hazard, goal, ...
    world.patterns      what the first look noticed: repeated items, same shape in different colours
    world.actions       what each action tends to do: changes the screen? undoes? kills? no-op here?
    world.rules         game-level facts: steps per life
    world.levels        per level: the first-look inventory and every screen seen (the "map")
    world.goals         what finishes a level, as the GAME showed it: features that changed on each win
    world.deaths        real deaths, kept so they can be re-judged as timeouts later

Kept across levels: counters, avatar, kinds, colour_roles, actions, rules (mechanics carry over).
Per level: levels[n] (layouts differ).
"""
from __future__ import annotations

import zlib
from collections import Counter, deque

import numpy as np

from .settings import PRIOR_WEIGHT

DIED = "DIED"
LEVEL_UP = "LEVEL_UP"

# An action is a tuple (action_id, x, y); key actions use x = y = -1.
Action = tuple
RESET_ACTION: Action = (0, -1, -1)
# Candidate info is (prior chance, colour clicked or -1).
NO_INFO = (0.5, -1)


class World:
    def __init__(self, shape=(64, 64)) -> None:
        self.counters = new_counter_memory(shape)
        self.avatar = {
            "colour": None,  # best current guess
            "moves_seen": 0,
            "shifts": {},  # colour -> action_id -> Counter of (dy, dx) centroid shifts
            "steps": {},  # action_id -> Counter of (dy, dx): how far the avatar moves when it moves
            "moves": 0,  # times the avatar actually moved
            "moved_with": Counter(),  # colour -> times it moved the same way as the avatar
            "parts": set(),  # extra colours that belong to the avatar sprite
            "under": None,  # what each covered cell showed before the avatar arrived (255 = n/a)
        }
        self.kinds: dict[tuple, dict] = {}  # (colour, shape) -> {"name", "colour", "size", "purpose", "seen_in_levels"}
        self.colour_roles: dict[int, Counter] = {}  # colour -> Counter of observed roles
        self.patterns: list[dict] = []  # {"type", "kinds": [...], "level"}
        self.actions = {
            "changed_screen": {},  # action_id -> Tally: did it change the screen?
            "click_colour_changed_screen": {},  # clicked colour -> Tally
            "undid_last_move": {},  # action_id -> Tally: went back to the screen before last?
            "reversed_after": {},  # (previous_id, action_id) -> Tally
            "no_op_at": {},  # (avatar position, action_id) -> count
            "died_at": {},  # (avatar position, action_id) -> count
        }
        self.rules = {
            "steps_per_life": {},  # level -> step budget of one life
        }
        self.levels: dict[int, LevelMemory] = {}
        self.goals: list[dict] = []  # {"feature", "direction", "support", "against", "source"}
        self.deaths: list[tuple] = []  # (level, life_length, screen, action, avatar_at)


def new_counter_memory(shape) -> dict:
    return {
        "shape": shape,
        "ignore": np.zeros(shape, dtype=bool),  # True = not part of a screen's identity
        "tick_cells": np.zeros(shape, dtype=bool),
        "ticks_along_row": [set() for _ in range(shape[0])],  # positions of ticks on each row
        "ticks_along_col": [set() for _ in range(shape[1])],
        "change_counts": np.zeros(shape, dtype=np.int32),
        "transitions": 0,
        "version": 0,
    }


class Tally:
    """How often something happened out of how many tries."""
    __slots__ = ("tries", "hits")

    def __init__(self) -> None:
        self.tries = 0
        self.hits = 0

    def add(self, hit: bool) -> None:
        self.tries += 1
        self.hits += int(hit)

    def chance(self, prior: float) -> float:
        return (self.hits + PRIOR_WEIGHT * prior) / (self.tries + PRIOR_WEIGHT)


class Screen:
    """One remembered screen (identified by what matters on it)."""
    __slots__ = ("id", "candidates", "results", "avatar_at", "snapshot", "click_scans", "features")

    def __init__(self, screen_id: int, candidates: dict, avatar_at, snapshot: bytes) -> None:
        self.id = screen_id
        self.candidates = candidates  # action -> (prior chance, colour), best first
        self.results: dict = {}  # action -> next screen id | DIED | LEVEL_UP
        self.avatar_at = avatar_at  # avatar position here (or None)
        self.snapshot = snapshot  # compressed raw picture, to re-identify after the counter mask grows
        self.click_scans = 0  # whole-screen click grids added so far
        self.features = None  # goals.screen_features vector (colour counts, touches, avatar touches)

    def untried(self) -> list:
        return [a for a in self.candidates if a not in self.results]


class LevelMemory:
    """The map of one level: screens and what each action did on them."""

    def __init__(self) -> None:
        self.items: list[dict] = []  # first-look inventory: {"kind", "colour", "size", "centre", "bbox"}
        self.screens: dict[int, Screen] = {}
        self.start_screen: int | None = None
        self.contradictions = 0  # same screen + same action -> different result
        self.has_solution = False

    def record_result(self, screen_id: int, action: Action, result) -> bool:
        """Store what an action did. Returns False if it contradicts what we saw before."""
        screen = self.screens.get(screen_id)
        if screen is None:
            return True
        before = screen.results.get(action)
        screen.results[action] = result
        if result == LEVEL_UP:
            self.has_solution = True
        if before is not None and before != result:
            self.contradictions += 1
            return False
        return True

    def rename_screens(self, id_of) -> None:
        """Recompute every screen id with id_of(snapshot) - needed when the
        counter mask grows. Screens that now look the same are merged."""
        new_id = {old: id_of(s.snapshot) for old, s in self.screens.items()}
        merged: dict[int, Screen] = {}
        for old, screen in self.screens.items():
            screen.id = new_id[old]
            screen.results = {a: (new_id.get(r, r) if isinstance(r, int) else r) for a, r in screen.results.items()}
            keep = merged.get(screen.id)
            if keep is None:
                merged[screen.id] = screen
                continue
            for a, info in screen.candidates.items():
                keep.candidates.setdefault(a, info)
            for a, r in screen.results.items():
                if a in keep.results and keep.results[a] != r:
                    self.contradictions += 1
                keep.results.setdefault(a, r)
        self.screens = merged
        if self.start_screen is not None:
            self.start_screen = new_id.get(self.start_screen, self.start_screen)

    def walking_distances(self, start: int, stop_beyond: float | None = None):
        """Breadth-first search over known moves: (distance, came_from) per reachable screen."""
        distance, came_from = {start: 0}, {start: None}
        queue = deque([start])
        while queue:
            here = queue.popleft()
            if stop_beyond is not None and distance[here] > stop_beyond:
                break
            screen = self.screens.get(here)
            if screen is None:
                continue
            for action, there in screen.results.items():
                if isinstance(there, int) and there not in distance and there in self.screens:
                    distance[there] = distance[here] + 1
                    came_from[there] = (here, action)
                    queue.append(there)
        return distance, came_from

    @staticmethod
    def route_to(came_from: dict, target: int) -> list:
        """[(action, screen it should lead to), ...] from the search start to target."""
        route = []
        here = target
        while came_from[here] is not None:
            previous, action = came_from[here]
            route.append((action, here))
            here = previous
        route.reverse()
        return route

    def route_to_level_up(self, start: int) -> list | None:
        """Shortest known route from `start` to finishing the level."""
        if start not in self.screens:
            return None
        distance, came_from = self.walking_distances(start)
        best = None
        for screen_id, d in distance.items():
            for action, result in self.screens[screen_id].results.items():
                if result == LEVEL_UP and (best is None or d + 1 < best[0]):
                    best = (d + 1, screen_id, action)
        if best is None:
            return None
        return self.route_to(came_from, best[1]) + [(best[2], LEVEL_UP)]


def compress_grid(grid) -> bytes:
    return zlib.compress(grid.tobytes(), 1)


def decompress_grid(blob: bytes, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(shape)
