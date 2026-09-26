"""GOALS — what finishes a level is decided by the game, not written in the code.

The agent describes every screen with the same generic, measurable FEATURES (ScreenFeatures):
    ("count", c)             how many cells of colour c are visible
    ("touch", a, b)          how many cell edges where colour a meets colour b
    ("avatar_touch", c)      how many cells of colour c the avatar's body is touching

When the game says "level complete", Goals compares the screen just before the winning
action with the level's first screen. The features that changed most are what the game
rewarded -> they become goal hypotheses in world.goals:
    Goal(feature=("touch", 4, 9), direction=+1, support=1, against=0, source="won level 0")
Every later win supports or contradicts each hypothesis (Goal.confidence()).

Before any win, the only goals are weak cues the game itself shows at the first look
(one shape in two colours -> "make them touch").

Goals.progress(level_map, screen) scores how far a screen has moved in the direction of
the confident goals; decide.Explorer explores such screens first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import LevelScene, SceneObject
from .settings import (
    GOAL_CUE_CONFIDENCE,
    GOAL_MIN_CONFIDENCE,
    GOAL_MIN_RELATIVE_CHANGE,
    GOALS_PER_WIN,
)
from .shapes import Correspondence, ShapeRelations
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


class Goals:
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

    @staticmethod
    def reacting_objects(scene: LevelScene) -> set[int]:
        """Objects that moved or changed right after one of our actions this level."""
        return {c.object_id for o in scene.observations for c in o.changes if c.kind != "appeared"}

    @staticmethod
    def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Overlap of two boxes (y0, x0, y1, x1) as a share of the smaller one: 1 = one lies on the other."""
        h = min(a[2], b[2]) - max(a[0], b[0]) + 1
        w = min(a[3], b[3]) - max(a[1], b[1]) + 1
        smaller = min((a[2] - a[0] + 1) * (a[3] - a[1] + 1), (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
        return max(0, h) * max(0, w) / smaller if smaller else 0.0

    @staticmethod
    def distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Distance between box centres, in cells."""
        return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))

    @staticmethod
    def progress_text(scene: LevelScene, goal: GoalCandidate | None) -> str:
        if goal is None:
            return ""
        boxes = {o.id: o.bbox for o in scene.objects}
        if goal.relation == "clear":
            left = sum(1 for o in scene.objects if (o.colour, o.shape) == goal.kind)
            return f"{left} of {goal.b} copies left"
        if goal.a not in boxes or goal.b not in boxes:
            return "one of the two objects is not visible"
        return (f"overlap {Goals.overlap(boxes[goal.a], boxes[goal.b]):.2f}, "
                f"distance {Goals.distance(boxes[goal.a], boxes[goal.b]):.1f} cells")

    @staticmethod
    def purposes(scene: LevelScene, roles: dict[int, str] | None = None) -> list[str]:
        """What each object seems to be for: how it responded to actions, and the purpose the LLM gave it."""
        moves: dict[int, dict[str, set[str]]] = {}
        for obs in scene.observations:
            a = obs.action
            name = "click" if a[0] == 6 else f"ACTION{a[0]}"
            for c in obs.changes:
                if c.kind == "moved":
                    d = ("right" if c.dx > 0 else "left") if c.dx else ("down" if c.dy > 0 else "up")
                    moves.setdefault(c.object_id, {}).setdefault(name, set()).add(d)
                elif c.kind in ("recoloured", "disappeared"):
                    moves.setdefault(c.object_id, {}).setdefault(name, set()).add(c.kind)
        roles = roles or {}
        visible = {o.id for o in scene.objects}
        lines = []
        for oid in sorted(set(moves) | {i for i in roles if i in visible}):
            parts = [f"{act} -> {'/'.join(sorted(ds))}" for act, ds in sorted(moves.get(oid, {}).items())]
            text = ", ".join(parts) if parts else "never reacted to any action"
            if oid in roles:
                text += f"; you called it: {roles[oid]}"
            lines.append(f"#{oid}: {text}")
        return lines[:14]

    def _is_just_moving(self, name: tuple) -> bool:
        """Changes that only say 'the avatar is somewhere else' are not goals: the avatar's
        own colours and what it happens to stand next to (floor)."""
        av = self.world.avatar
        own = {av.colour} | av.parts
        colours = set(name[1:])
        if name[0] == "avatar_touch":
            return self.world.colour_roles.role_of(name[1]) == Role.FLOOR
        return bool(colours & own)


# ── possible goals: which things on screen should go together ───────────────


@dataclass(slots=True)
class GoalCandidate:
    """One possible goal: two objects that belong together should coincide, or copies of a kind should go."""
    relation: str  # a shapes.RELATION_STRENGTH key, or "clear" (all copies of a kind gone)
    a: int  # object id
    b: int  # object id (for "clear": the number of copies at the level start)
    score: float
    why: str
    kind: tuple | None = None  # "clear": the (colour, shape) whose copies should go

    @property
    def key(self) -> tuple[str, int, int]:
        return self.relation, self.a, self.b

    def describe(self) -> str:
        if self.relation == "clear":
            return f"all {self.b} copies of #{self.a} are gone ({self.why})"
        return f"#{self.a} and #{self.b} end up together / overlapping ({self.why})"


class GoalBoard:
    """The ranked list of possible goals, re-ranked as evidence arrives (hypothesis mode).

    score = shape strength x (1 + reacts to our actions) x (1 + 0.5 x LLM picks) x (1 + wins of this relation)
            x (1 + 2 x levels won by bringing these two colours together) x (1 + levels won on the target colour)
    A goal reached without finishing the level, while more objects of the target's colour remain, is kept as
    DONE (one of several targets) and its two objects leave the list.
    """

    def __init__(self) -> None:
        self.llm_votes: dict[tuple[str, int, int], int] = {}
        self.won_relations: dict[str, int] = {}  # relation kind -> levels finished while it held
        self.chosen: GoalCandidate | None = None
        self.roles: dict[int, str] = {}  # object id -> purpose the LLM gave it (this level; shown in questions)
        self.won_pairs: dict[tuple[int, int], int] = {}  # (mover colour, target colour) -> levels won (lessons)
        self.done: list[tuple[int, int]] = []  # (mover, target) pairs reached this level; the level goes on
        self.controlled: set[int] = set()  # objects the LLM says the player moves (this level)
        self._cache_key: frozenset = frozenset()
        self._cache: list[Correspondence] = []

    def new_level(self) -> None:
        self.chosen, self.roles, self.done, self.controlled = None, {}, [], set()
        self._cache_key = frozenset()

    def possible_goals(self, scene: LevelScene) -> list[GoalCandidate]:
        reacting = Goals.reacting_objects(scene)
        colour = {o.id: o.colour for o in scene.objects}
        held = {i for pair in self.done for i in pair}
        won_targets: dict[int, int] = {}
        for (_m, t), n in self.won_pairs.items():
            won_targets[t] = won_targets.get(t, 0) + n
        kinds_now = frozenset((o.colour, o.shape) for o in scene.objects if not o.background)
        if kinds_now != self._cache_key:  # shapes only change when the set of kinds does
            self._cache_key, self._cache = kinds_now, ShapeRelations.find(scene)
        candidates = []
        for c in self._cache:
            if c.relation == "same-colour":
                continue  # a grouping hint for purposes, too weak on its own to be a goal
            if c.a in held or c.b in held:
                continue
            live = c.a in reacting or c.b in reacting
            if not live and colour.get(c.a) == colour.get(c.b):
                continue  # two still objects of one colour (two targets): nothing to bring together
            votes = self.llm_votes.get((c.relation, c.a, c.b), 0)
            wins = self.won_relations.get(c.relation, 0)
            ca, cb = colour.get(c.a, -1), colour.get(c.b, -1)
            pair_wins = self.won_pairs.get((ca, cb), 0) + self.won_pairs.get((cb, ca), 0)
            target_wins = won_targets.get(ca, 0) + won_targets.get(cb, 0)
            score = c.strength * (2.0 if live else 1.0) * (1 + 0.5 * votes) * (1 + wins) \
                * (1 + 2 * pair_wins) * (1 + target_wins)
            why = c.describe().split(" ", 1)[1].rsplit(" ", 1)[0]
            if live:
                why += ", and one of them reacts to actions"
            if pair_wins or target_wins:
                why += ", and earlier levels were won with these colours"
            candidates.append(GoalCandidate(c.relation, c.a, c.b, score, why))
        candidates += self._reach_goals(scene, reacting, held)
        copies: dict[tuple, list[SceneObject]] = {}
        for o in scene.objects:
            if not o.background and scene.root(o.id) == o.id and o.size > 1:
                copies.setdefault((o.colour, o.shape), []).append(o)
        for group in copies.values():
            if len(group) >= 3:  # counting: "as few as possible" is a goal that needs no partner
                key = ("clear", group[0].id, len(group))
                score = 0.4 * (1 + 0.5 * self.llm_votes.get(key, 0)) * (1 + self.won_relations.get("clear", 0))
                candidates.append(GoalCandidate("clear", group[0].id, len(group), score,
                                                f"{len(group)} identical copies", (group[0].colour, group[0].shape)))
        candidates.sort(key=lambda g: -g.score)
        return candidates[:MAX_GOAL_CANDIDATES]

    def _reach_goals(self, scene: LevelScene, reacting: set[int], held: set[int]) -> list[GoalCandidate]:
        """The player reaching what the LLM called a goal / collectible / switch (avatar games: no shape pair)."""
        visible = {o.id for o in scene.objects if scene.root(o.id) == o.id}
        movers = [i for i in self.controlled if i in visible and i in reacting and i not in held]
        found = []
        for target, role in self.roles.items():
            if role not in REACH_ROLES or target not in visible or target in held:
                continue
            for mover in movers:
                if mover != target:
                    key = ("reach", mover, target)
                    score = 0.6 * (1 + 0.5 * self.llm_votes.get(key, 0))
                    found.append(GoalCandidate("reach", mover, target, score, f"the LLM called #{target} a {role}"))
        return found

    def vote(self, candidates: list[GoalCandidate], choice: int) -> None:
        """The LLM picked candidate `choice` (1-based; 0 = none of them): it moves up the list."""
        if 1 <= choice <= len(candidates):
            picked = candidates[choice - 1]
            self.llm_votes[picked.key] = self.llm_votes.get(picked.key, 0) + 1
            self.chosen = picked

    def learn_from_win(self, scene: LevelScene) -> None:
        """Which kinds of relation held when the level was won? They rank higher on later levels."""
        boxes = {o.id: o.bbox for o in scene.objects}
        for c in self._cache:
            if c.a in boxes and c.b in boxes and Goals.overlap(boxes[c.a], boxes[c.b]) >= 0.5:
                self.won_relations[c.relation] = self.won_relations.get(c.relation, 0) + 1
        self._cache_key = frozenset()


MAX_GOAL_CANDIDATES = 8
REACH_ROLES = ("goal", "collectible", "switch")
