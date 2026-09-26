"""WORLD — the model: everything the agent believes about the game it is playing.

It starts almost empty and fills up as the agent plays. Code never changes per game; only these
beliefs do. The classes here hold data and answer read-only questions about it; the learners
(look.FirstLook, learn.*Learner, goals.Goals) are the only code that writes them.

    World                 the root: one per game
      .counters           CounterMemory   HUD step-counter pixels, ignored when comparing screens
      .avatar             AvatarBelief    which colour is the player, how far each key moves it
      .kinds              {KindKey: Kind} every distinct item type seen (colour + shape), purpose "?"
      .colour_roles       ColourRoles     what touching each colour did: floor, wall, hazard, ...
      .patterns           [Pattern]       what first looks noticed: repeated kinds, one shape in many colours
      .actions            ActionBeliefs   what each action tends to do: change the screen? undo? kill?
      .rules              Rules           game-level facts: steps per life
      .levels             {int: LevelMemory}  per level: the first-look inventory and the map of screens
      .goals              [Goal]          what finishes a level, as the GAME showed it
      .deaths             [Death]         real deaths, kept so they can be re-judged as timeouts

Kept across levels: everything except levels[n] (layouts differ per level).
"""
from __future__ import annotations

import zlib
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np

from .settings import (
    AVATAR_MAX_CELLS,
    AVATAR_PART_SEARCH_MARGIN,
    PRIOR_WEIGHT,
    ROLE_MIN_EVIDENCE,
    ROLE_MIN_SHARE,
    STEP_MIN_OBSERVATIONS,
)


class Action(NamedTuple):
    """One input: (id, x, y). Key actions use x = y = -1. Equal to (and hashes like) a plain tuple."""
    id: int
    x: int = -1
    y: int = -1

    @property
    def is_click(self) -> bool:
        return self.id == 6


RESET_ACTION = Action(0, -1, -1)


class Outcome(Enum):
    """Results of an action that are not another screen."""
    DIED = "DIED"
    LEVEL_UP = "LEVEL_UP"


DIED = Outcome.DIED
LEVEL_UP = Outcome.LEVEL_UP
Result = int | Outcome  # a screen id, or an Outcome


class Role(Enum):
    """What touching a colour did to the avatar."""
    FLOOR = "floor"
    WALL = "wall"
    HAZARD = "hazard"
    GOAL = "goal"
    PUSHABLE = "pushable"
    COLLECTIBLE = "collectible"


class ClickInfo(NamedTuple):
    """What the agent knew about a candidate action before trying it."""
    prior: float  # prior chance of a new screen
    colour: int  # colour clicked, or -1 for key actions


NO_INFO = ClickInfo(0.5, -1)
KindKey = tuple  # (colour, (height, width, mask bytes))


class GameMode(Enum):
    """What kind of game the offered buttons say this is (docs/design/PLAYABLES.md)."""
    MOVER = "mover"  # keys 1-4, no clicking: what moves, what blocks it
    CLICKER = "clicker"  # clicking, no keys 1-4: which objects react to a click, and how
    MIXED = "mixed"  # both: select / act by clicking, move with keys
    OTHER = "other"  # neither (e.g. only ACTION5 / ACTION7)

    @staticmethod
    def from_actions(available: list[int]) -> GameMode:
        keys = any(a in available for a in (1, 2, 3, 4))
        click = 6 in available
        if keys and click:
            return GameMode.MIXED
        if keys:
            return GameMode.MOVER
        if click:
            return GameMode.CLICKER
        return GameMode.OTHER


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


# ── beliefs kept across levels ──────────────────────────────────────────────


class CounterMemory:
    """Which pixels are HUD counters / blinkers (written by learn.CounterLearner)."""

    def __init__(self, shape: tuple[int, ...] = (64, 64)) -> None:
        self.version = 0
        self._start_over(shape)

    def _start_over(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ignore = np.zeros(shape, dtype=bool)  # True = not part of a screen's identity
        self.tick_cells = np.zeros(shape, dtype=bool)
        self.ticks_along_row: list[set[int]] = [set() for _ in range(shape[0])]  # tick positions per row
        self.ticks_along_col: list[set[int]] = [set() for _ in range(shape[1])]
        self.change_counts = np.zeros(shape, dtype=np.int32)
        self.transitions = 0

    def fit(self, shape: tuple[int, ...]) -> None:
        """The frame size changed: start over (and bump the version: old screen ids are stale)."""
        if self.shape != shape:
            self._start_over(shape)
            self.version += 1

    def hide(self, grid: np.ndarray) -> np.ndarray:
        """Copy of the picture with ignored (counter) pixels set to 255."""
        self.fit(grid.shape)
        picture = grid.copy()
        picture[self.ignore] = 255
        return picture


class AvatarBelief:
    """Which colour is the player (written by learn.AvatarLearner / learn.TouchLearner)."""

    def __init__(self) -> None:
        self.colour: int | None = None  # best current guess
        self.moves_seen = 0
        self.shifts: dict[int, dict[int, Counter]] = {}  # colour -> action_id -> Counter of (dy, dx) shifts
        self.steps: dict[int, Counter] = {}  # action_id -> Counter of (dy, dx): how far the avatar moves
        self.moves = 0  # times the avatar actually moved
        self.moved_with: Counter = Counter()  # colour -> times it moved the same way as the avatar
        self.parts: set[int] = set()  # extra colours that belong to the avatar sprite
        self.under: np.ndarray | None = None  # what each covered cell showed before the avatar (255 = n/a)

    def usual_step(self, action_id: int) -> tuple[int, int] | None:
        """The avatar's usual (dy, dx) for a key, once seen STEP_MIN_OBSERVATIONS times."""
        steps = self.steps.get(action_id)
        if not steps:
            return None
        step, count = steps.most_common(1)[0]
        return step if count >= STEP_MIN_OBSERVATIONS else None

    def position(self, grid: np.ndarray) -> tuple[int, int] | None:
        """Centroid of the avatar colour at quarter-cell resolution, or None."""
        if self.colour is None:
            return None
        ys, xs = np.nonzero(grid == self.colour)
        if len(ys) == 0:
            return None
        return int(round(ys.mean() * 4)), int(round(xs.mean() * 4))

    def body(self, picture: np.ndarray) -> np.ndarray | None:
        """Cells of the avatar: its main colour plus 'part' colours touching it
        (multi-colour sprites). None if unknown, absent or implausibly large."""
        main = picture == self.colour
        if not main.any():
            return None
        body = main
        if self.parts:  # grow from the main colour into touching part-colour cells
            allowed = main | np.isin(picture, list(self.parts))
            for _ in range(AVATAR_PART_SEARCH_MARGIN):
                grown = body.copy()
                grown[1:, :] |= body[:-1, :]
                grown[:-1, :] |= body[1:, :]
                grown[:, 1:] |= body[:, :-1]
                grown[:, :-1] |= body[:, 1:]
                grown &= allowed
                if np.array_equal(grown, body):
                    break
                body = grown
        return None if body.sum() > AVATAR_MAX_CELLS else body

    def own_colours(self) -> set:
        """The avatar's colours plus 255 (hidden cells): never 'something else'."""
        return {self.colour, 255} | self.parts


class ColourRoles:
    """colour -> Counter of observed Roles (written by learn.TouchLearner)."""

    def __init__(self) -> None:
        self.evidence: dict[int, Counter] = {}

    def add(self, colours: set, role: Role) -> None:
        for c in colours:
            self.evidence.setdefault(c, Counter())[role] += 1

    def forget(self, colour: int) -> None:
        self.evidence.pop(colour, None)

    def role_of(self, colour: int) -> Role | None:
        """The colour's dominant role once it has enough evidence, else None."""
        roles = self.evidence.get(colour)
        if not roles:
            return None
        role, count = roles.most_common(1)[0]
        return role if count >= ROLE_MIN_EVIDENCE and count >= ROLE_MIN_SHARE * sum(roles.values()) else None


@dataclass(slots=True)
class Kind:
    """A distinct item type: one colour + one exact shape (written by look.FirstLook)."""
    name: str
    colour: int
    size: int
    purpose: str = "?"  # "?" until play reveals it; "background?" for very large items
    seen_in_levels: set[int] = field(default_factory=set)


@dataclass(slots=True)
class Pattern:
    """Something a first look noticed."""
    type: str  # "repeated" (one kind several times) or "same_shape" (one shape in several colours)
    kinds: list[str]
    count: int
    level: int = -1


@dataclass(slots=True)
class Item:
    """One thing on a level's first screen."""
    kind: KindKey
    colour: int
    size: int
    centre: tuple[int, int]  # (y, x)
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)


class ActionBeliefs:
    """What each action tends to do (written by learn.ActionLearner)."""

    def __init__(self) -> None:
        self.changed_screen: dict[int, Tally] = {}  # action_id -> did it change the screen?
        self.click_colour_changed_screen: dict[int, Tally] = {}  # clicked colour -> Tally
        self.undid_last_move: dict[int, Tally] = {}  # action_id -> went back to the screen before last?
        self.reversed_after: dict[tuple[int, int], Tally] = {}  # (previous_id, action_id) -> Tally
        self.no_op_at: dict[tuple, int] = {}  # (avatar position, action_id) -> count
        self.died_at: dict[tuple, int] = {}  # (avatar position, action_id) -> count


@dataclass(slots=True)
class Rules:
    steps_per_life: dict[int, int] = field(default_factory=dict)  # level -> step budget of one life


@dataclass(slots=True)
class Goal:
    """A hypothesis about what finishes a level: `feature` moves in `direction`."""
    feature: tuple  # goals.ScreenFeatures name, e.g. ("touch", 4, 9)
    direction: int  # +1 or -1
    support: int
    against: int
    source: str
    prior: float | None = None  # confidence before any evidence (first-look cues)

    def confidence(self) -> float:
        if self.prior is not None and self.support + self.against == 0:
            return self.prior
        return (self.support + 1) / (self.support + self.against + 2)


@dataclass(slots=True)
class Death:
    level: int
    life_length: int
    screen: Screen
    action: Action
    avatar_at: tuple[int, int] | None


# ── the per-level map ───────────────────────────────────────────────────────


class Screen:
    """One remembered screen (identified by what matters on it)."""
    __slots__ = ("id", "candidates", "results", "avatar_at", "snapshot", "click_scans", "features")

    def __init__(self, screen_id: int, candidates: dict[Action, ClickInfo], avatar_at, snapshot: bytes) -> None:
        self.id = screen_id
        self.candidates = candidates  # action -> ClickInfo, best first
        self.results: dict[Action, Result] = {}  # action -> next screen id | DIED | LEVEL_UP
        self.avatar_at = avatar_at  # avatar position here (or None)
        self.snapshot = snapshot  # compressed raw picture, to re-identify after the counter mask grows
        self.click_scans = 0  # whole-screen click grids added so far
        self.features: np.ndarray | None = None  # goals.ScreenFeatures vector

    def untried(self) -> list[Action]:
        return [a for a in self.candidates if a not in self.results]


class LevelMemory:
    """The map of one level: screens and what each action did on them."""

    def __init__(self) -> None:
        self.items: list[Item] = []  # first-look inventory
        self.screens: dict[int, Screen] = {}
        self.start_screen: int | None = None
        self.contradictions = 0  # same screen + same action -> different result
        self.has_solution = False

    def record_result(self, screen_id: int, action: Action, result: Result) -> bool:
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
        distance = {start: 0}
        came_from: dict[int, tuple[int, Action] | None] = {start: None}
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
    def route_to(came_from: dict, target: int) -> list[tuple[Action, Result]]:
        """[(action, screen it should lead to), ...] from the search start to target."""
        route: list[tuple[Action, Result]] = []
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


# ── short-term memory (one life / one level) ────────────────────────────────


@dataclass(slots=True)
class WorkingMemory:
    """What just happened and what is planned - the Brain's short-term memory, not beliefs."""
    level: int | None = None
    last_grid: np.ndarray | None = None  # picture before last_action
    last_screen_id: int | None = None
    screen_before_last_id: int | None = None
    last_action: Action | None = None
    came_by: Action | None = None  # the action that brought us to this screen
    route: list = field(default_factory=list)  # planned [(action, screen it should lead to), ...]
    expected_screen: Result | None = None  # where the route's last step should have taken us
    at_level_start: bool = False
    life_length: int = 0  # actions since the last reset / level start
    pixel_scans: dict[int, int] = field(default_factory=dict)  # level -> screens given a 1-pixel click scan
    plan: list = field(default_factory=list)  # the LLM advisor's remaining actions
    last_from_advisor: bool = False
    ask_again_at: int = 0  # after an empty answer the brain plays a few steps first
    recent: deque = field(default_factory=deque)  # "ACTIONn -> what happened", for the advisor
    # why the current action was chosen (for traces and debugging; never read by decisions)
    why: str = ""
    route_reason: str = ""  # why the current route was planned
    last_outcome: str = ""  # what the previous action did
    theory: str = ""  # the advisor's latest theory of the game
    plan_size: int = 0  # length of the advisor's current plan when it arrived


# ── the root ────────────────────────────────────────────────────────────────


class World:
    def __init__(self, shape: tuple[int, int] = (64, 64)) -> None:
        self.counters = CounterMemory(shape)
        self.avatar = AvatarBelief()
        self.kinds: dict[KindKey, Kind] = {}
        self.colour_roles = ColourRoles()
        self.patterns: list[Pattern] = []
        self.actions = ActionBeliefs()
        self.rules = Rules()
        self.levels: dict[int, LevelMemory] = {}
        self.goals: list[Goal] = []
        self.deaths: list[Death] = []
        self.mode: GameMode | None = None  # from the offered actions (learn.ActionLearner.note_available)


def compress_grid(grid: np.ndarray) -> bytes:
    return zlib.compress(grid.tobytes(), 1)


def decompress_grid(blob: bytes, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(shape)
