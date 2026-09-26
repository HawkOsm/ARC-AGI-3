"""GOALS — what finishes a level is decided by the game, not written in the code.

The agent describes every screen with the same generic, measurable FEATURES (ScreenFeatures):
    ("count", c)             how many cells of colour c are visible
    ("touch", a, b)          how many cell edges where colour a meets colour b
    ("avatar_touch", c)      how many cells of colour c the avatar's body is touching

When the game says "level complete", GoalTracker compares the screen just before the winning
action with the level's first screen. The features that changed most are what the game
rewarded -> they become goal hypotheses in world.goals:
    Goal(feature=("touch", 4, 9), direction=+1, support=1, against=0, source="won level 0")
Every later win supports or contradicts each hypothesis (Goal.confidence()).

Before any win, the only goals are weak cues the game itself shows at the first look
(one shape in two colours -> "make them touch").

GoalTracker.progress(level_map, screen) scores how far a screen has moved in the direction of
the confident goals; decide.Explorer explores such screens first.
"""
from __future__ import annotations

import numpy as np

from .settings import (
    GOAL_CUE_CONFIDENCE,
    GOAL_MIN_CONFIDENCE,
    GOAL_MIN_RELATIVE_CHANGE,
    GOALS_PER_WIN,
)
from .world import Goal, LevelMemory, Role, Screen, World, decompress_grid

N_COLOURS = 16


class ScreenFeatures:
    """The generic feature vector of a screen (a utility class: static members only)."""
    NAMES: list[tuple] = (
        [("count", c) for c in range(N_COLOURS)]
        + [("touch", a, b) for a in range(N_COLOURS) for b in range(a + 1, N_COLOURS)]
        + [("avatar_touch", c) for c in range(N_COLOURS)]
    )
    INDEX = {name: i for i, name in enumerate(NAMES)}
    _TOUCH_OFFSET = N_COLOURS
    _AVATAR_OFFSET = N_COLOURS + N_COLOURS * (N_COLOURS - 1) // 2

    @staticmethod
    def of(world: World, grid: np.ndarray) -> np.ndarray:
        """The feature vector of a screen (int16, len(NAMES)); counters are ignored."""
        picture = world.counters.hide(grid)
        f = np.zeros(len(ScreenFeatures.NAMES), dtype=np.int32)
        visible = picture[picture < N_COLOURS]
        f[:N_COLOURS] = np.bincount(visible, minlength=N_COLOURS)[:N_COLOURS]
        touches = np.zeros((N_COLOURS, N_COLOURS), dtype=np.int32)
        for a, b in ((picture[:, :-1], picture[:, 1:]), (picture[:-1, :], picture[1:, :])):
            edge = (a != b) & (a < N_COLOURS) & (b < N_COLOURS)
            lo, hi = np.minimum(a[edge], b[edge]), np.maximum(a[edge], b[edge])
            np.add.at(touches, (lo, hi), 1)
        iu = np.triu_indices(N_COLOURS, k=1)
        f[ScreenFeatures._TOUCH_OFFSET:ScreenFeatures._AVATAR_OFFSET] = touches[iu]
        body = world.avatar.body(picture) if world.avatar.colour is not None else None
        if body is not None:
            around = np.zeros_like(body)
            around[1:, :] |= body[:-1, :]
            around[:-1, :] |= body[1:, :]
            around[:, 1:] |= body[:, :-1]
            around[:, :-1] |= body[:, 1:]
            around &= ~body
            near = picture[around]
            near = near[near < N_COLOURS]
            f[ScreenFeatures._AVATAR_OFFSET:] = np.bincount(near, minlength=N_COLOURS)[:N_COLOURS]
        return np.clip(f, -32768, 32767).astype(np.int16)


class GoalTracker:
    """Writes world.goals (from wins and first-look cues) and scores progress toward them."""

    def __init__(self, world: World) -> None:
        self.world = world

    def learn_from_win(self, level: int, level_map: LevelMemory, winning_screen: Screen) -> None:
        """The game just said "level complete": compare the last screen with the first one."""
        world = self.world
        start = level_map.screens.get(level_map.start_screen) if level_map.start_screen is not None else None
        if start is None:
            return
        # recompute both with today's beliefs (the start screen was described before, e.g.,
        # the avatar was known - comparing those vectors would invent avatar "changes")
        shape = world.counters.shape
        before = ScreenFeatures.of(world, decompress_grid(start.snapshot, shape)).astype(np.int32)
        after = ScreenFeatures.of(world, decompress_grid(winning_screen.snapshot, shape)).astype(np.int32)
        change = after - before
        relative = change / (np.abs(before) + 1)
        for goal in world.goals:  # does this win agree with what we already believed?
            moved = change[ScreenFeatures.INDEX[goal.feature]]
            if moved * goal.direction > 0:
                goal.support += 1
            else:
                goal.against += 1
        known = {g.feature for g in world.goals}
        ranked = np.argsort(-np.abs(relative))
        added = 0
        for i in ranked:
            if added >= GOALS_PER_WIN or abs(relative[i]) < GOAL_MIN_RELATIVE_CHANGE:
                break
            name = ScreenFeatures.NAMES[i]
            if name in known or self._is_just_moving(name):
                continue
            world.goals.append(Goal(feature=name, direction=int(np.sign(change[i])), support=1, against=0,
                                    source=f"won level {level}"))
            added += 1

    def cues_from_first_look(self, level: int) -> None:
        """Weak cues the game shows before any win: one shape in several colours -> make them touch."""
        world = self.world
        if any(g.source.startswith("won") for g in world.goals):
            return
        known = {g.feature for g in world.goals}
        for pattern in world.patterns:
            if pattern.level != level or pattern.type != "same_shape":
                continue
            colours = sorted({k.colour for k in world.kinds.values() if k.name in pattern.kinds})
            for i, a in enumerate(colours):
                for b in colours[i + 1:]:
                    name = ("touch", a, b)
                    if name not in known:
                        world.goals.append(Goal(feature=name, direction=1, support=0, against=0,
                                                source=f"first look level {level}: same shape",
                                                prior=GOAL_CUE_CONFIDENCE))
                        known.add(name)

    def progress(self, level_map: LevelMemory, screen: Screen) -> float:
        """How far this screen has moved toward the confident goals, relative to the level's
        first screen (0 = no progress; only positive progress counts)."""
        world = self.world
        if not world.goals or screen.features is None:
            return 0.0
        start = level_map.screens.get(level_map.start_screen) if level_map.start_screen is not None else None
        if start is None or start.features is None:
            return 0.0
        total = 0.0
        for goal in world.goals:
            c = goal.confidence()
            if c < GOAL_MIN_CONFIDENCE:
                continue
            i = ScreenFeatures.INDEX[goal.feature]
            base = int(start.features[i])
            moved = (int(screen.features[i]) - base) * goal.direction / (abs(base) + 1)
            total += c * moved
        return max(0.0, total)

    def _is_just_moving(self, name: tuple) -> bool:
        """Changes that only say 'the avatar is somewhere else' are not goals: the avatar's
        own colours and what it happens to stand next to (floor)."""
        av = self.world.avatar
        own = {av.colour} | av.parts
        colours = set(name[1:])
        if name[0] == "avatar_touch":
            return self.world.colour_roles.role_of(name[1]) == Role.FLOOR
        return bool(colours & own)
