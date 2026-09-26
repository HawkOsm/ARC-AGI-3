"""BRAIN HINTS — our brain's findings, handed to the Duck harness (vendor/duck) as the `brain_hints` variable of its
Python tool (switch: DUCK_BRAIN_HINTS=1). The Duck's model can read them or ignore them; nothing here acts.

For one game session, BrainHints replays the Duck's history into our own pieces and reports:
    control_modes            which object each key moves, and which click / key switches that (control.ControlMap)
    possible_goals           objects that belong together by shape, ranked (goals.GoalBoard, shapes.ShapeRelations)
    clicks_that_did_nothing  objects whose click changed nothing on this level
    earlier_levels           how finished levels ended; new_or_changed: what differs on this level (lessons.GameLessons)
    objects                  every object id used above -> colour letter and row / col box (the Duck's coordinates)
Updated incrementally: each call only feeds the frames added since the last one.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.brain.control import ControlMap
from agent.brain.goals import GoalBoard
from agent.brain.learn import CounterLearner
from agent.brain.lessons import GameLessons
from agent.brain.scene import COLOUR_NAMES, SceneObject, SceneTracker
from agent.brain.world import Action, World

ARC_CHARS = "WwgGcBMPRbSYOrNp"  # the Duck's letter per colour (inference/utils/grid_utils.py)
ARC_NAMES = ["white", "light gray", "gray", "dark gray", "charcoal", "black", "magenta", "pink", "red", "blue",
             "sky blue", "yellow", "orange", "dark red", "light green", "purple"]
KEYS = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5}
MOUSE = re.compile(r"MOUSE\(row=(\d+),\s*col=(\d+)\)", re.IGNORECASE)
MAX_LINES = 8


def to_action(name: str) -> Action | None:
    """The Duck's action label ("UP", "MOUSE(row=4, col=7)") as our Action; None for RESET / unknown."""
    name = name.strip().upper()
    if name in KEYS:
        return Action(KEYS[name])
    m = MOUSE.fullmatch(name)
    return Action(6, int(m.group(2)), int(m.group(1))) if m else None


@dataclass(slots=True)
class BrainHints:
    """Our brain's view of one game, fed from the Duck's history. Step counters and timer bars are learned
    (learn.CounterLearner) and hidden, so a bar ticking down is not taken for a switch or a move."""
    world: World = field(default_factory=World)
    tracker: SceneTracker = field(init=False)
    counters: CounterLearner = field(init=False)
    last_grid: np.ndarray | None = None
    controls: dict[int, ControlMap] = field(default_factory=dict)
    lessons: GameLessons = field(default_factory=GameLessons)
    fed: int = 0  # history entries already processed
    level: int | None = None
    level_actions: int = 0

    def __post_init__(self) -> None:
        self.tracker, self.counters = SceneTracker(self.world), CounterLearner(self.world)

    def feed(self, history: list[tuple[str, np.ndarray, int]]) -> None:
        """New (action label, grid, level) entries since the last call (entry 0 is the start frame)."""
        if len(history) < self.fed:  # the history was cut (a new play): start over
            self.world = World()
            self.__post_init__()
            self.controls, self.lessons, self.fed, self.level, self.last_grid = {}, GameLessons(), 0, None, None
        for label, grid, level in history[self.fed:]:
            if self.last_grid is not None:
                self.counters.learn(self.last_grid, grid)
            self.last_grid = grid
            if self.level is not None and level != self.level:
                self._level_done(self.level)
                self.tracker.update(level, grid, None)
            else:
                action = to_action(label) if label else None
                observation = self.tracker.update(level, grid, action)
                if observation is not None:
                    self.controls.setdefault(level, ControlMap()).observe(self.tracker.scene(level), observation)
                self.level_actions += 1 if action is not None else 0
            self.level = level
        self.fed = len(history)

    def _level_done(self, level: int) -> None:
        scene = self.tracker.scene(level)
        self.lessons.end_level(level, scene, self.controls.setdefault(level, ControlMap()), self.level_actions,
                               None, None, "", {})
        self.level_actions = 0

    def report(self) -> dict[str, Any]:
        if self.level is None:
            return {}
        scene = self.tracker.scene(self.level)
        controls = self.controls.setdefault(self.level, ControlMap())
        board = GoalBoard()
        board.won_pairs = self.lessons.won_pairs
        goals = [g.describe() for g in board.possible_goals(scene)][:MAX_LINES]
        objects = {o.id: o for o in scene.objects if scene.root(o.id) == o.id}
        dead = [f"#{i}" for i in sorted(controls.dead_clicks) if i in objects]
        out: dict[str, Any] = {
            "about": "Computed by the agent's own analysis code from every frame so far. Object ids (#n) are this "
                     "analysis' ids, NOT segmentation ids: look them up in `objects`. Evidence, not orders.",
            "control_modes": _names(controls.describe(scene)),
            "possible_goals": _names(goals),
            "clicks_that_did_nothing": dead,
            "earlier_levels": _names(self.lessons.describe_wins()),
            "new_or_changed": _names(self.lessons.news(scene, controls)),
        }
        used = {int(i) for v in out.values() if isinstance(v, list) for line in v for i in re.findall(r"#(\d+)", line)}
        out["objects"] = {f"#{i}": _where(objects[i]) for i in sorted(used) if i in objects}
        return {k: v for k, v in out.items() if v}


def _where(o: SceneObject) -> str:
    y0, x0, y1, x1 = o.bbox
    return f"{ARC_CHARS[o.colour]} ({ARC_NAMES[o.colour]}), {o.size} cells, rows {y0}-{y1}, cols {x0}-{x1}"


_COLOUR_INDEX = {name: i for i, name in enumerate(COLOUR_NAMES)}
_COLOUR_WORDS = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(COLOUR_NAMES, key=len, reverse=True))
                           + r")\b")


def _names(lines: list[str]) -> list[str]:
    """Our colour words -> the Duck's letter names (one pass: "light blue" is not renamed twice);
    ACTION1-5 -> UP DOWN LEFT RIGHT SPACE."""
    out = []
    for line in lines[:MAX_LINES]:
        line = _COLOUR_WORDS.sub(lambda m: f"{ARC_NAMES[_COLOUR_INDEX[m.group(1)]]} "
                                           f"({ARC_CHARS[_COLOUR_INDEX[m.group(1)]]})", line)
        for label, key in KEYS.items():
            line = line.replace(f"ACTION{key}", label)
        out.append(line)
    return out


_SESSIONS: dict[str, BrainHints] = {}
_LOCK = threading.Lock()


def hints_for(session: str, history: list[tuple[str, np.ndarray, int]]) -> dict[str, Any]:
    """The hints for one game session (keyed by its state file path); never raises."""
    try:
        with _LOCK:
            hints = _SESSIONS.setdefault(session, BrainHints())
        hints.feed(history)
        return hints.report()
    except Exception as exc:  # hints are optional: a bug here must never stop the game
        return {"error": f"brain hints unavailable: {type(exc).__name__}"}
