"""PLAYER — play mode (ARC3_LLM_MODE=play): the LLM drives, the way the top Kaggle entries do (Milestone #1: Reki,
forge — see docs/research/LLM_VS_ALGO.md). The code around it is kept small on purpose:

    each call    the last few screens as images (with a coordinate ruler; the last click ringed), the actions that led
                 to them and whether each changed anything, the model's own notes and last plan
    the answer   {"changed": ..., "plan": ..., "notes": ..., "actions": [1-4 actions]}
    helpers      only the ones Reki kept: legal actions only; a click is moved onto the nearest object cell; never
                 click again a kind of object whose clicks never changed anything this level ("dead signature")
The brain stays underneath: it handles RESET and lives, and acts when the model gives nothing usable.
"""
from __future__ import annotations

import base64
import json
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .llm import Advisor, parse_plan
from .look import ScreenReader
from .render import FramePainter
from .settings import LLM_PLAY_FRAMES, LLM_PLAY_MAX_ACTIONS, LLM_PLAY_NOTES, LLM_PLAY_SCALE, LLM_PLAY_TEXT_GRID
from .world import Action

PLAY_SYSTEM_PROMPT = (
    "You are playing an unknown turn-based puzzle game on a 64x64 grid of coloured cells. Nobody tells you the rules "
    "or the goal: find them out by playing. Finishing a level shows the next one; the step counter or lives shown on "
    "screen running out loses the life and restarts the level. Every action counts: finish each level in as few "
    "actions as possible.\n"
    "- ACTION1..ACTION5 are simple inputs (often up, down, left, right, use/select). ACTION6 x y clicks the cell at "
    "column x, row y (0-63; read them off the rulers: the numbers mark every 8th cell). ACTION7 is often undo. Only "
    "the listed actions exist in this game.\n"
    "- You see the most recent screens, oldest first; the last one is the current screen. A magenta ring marks the "
    "cell of the last click.\n"
    "Answer with one JSON object: changed = what the last actions did; plan = your short plan; notes = what you have "
    "learned about this game so far (kept for you and shown again next time); actions = the next 1 to 4 actions."
)

PLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "changed": {"type": "string", "maxLength": 300},
        "plan": {"type": "string", "maxLength": 300},
        "notes": {"type": "string", "maxLength": 500},
        "actions": {"type": "array", "minItems": 1, "maxItems": LLM_PLAY_MAX_ACTIONS,
                    "items": {"type": "string", "pattern": "^ACTION[1-7]( [0-9]{1,2} [0-9]{1,2})?$"}},
    },
    "required": ["changed", "plan", "notes", "actions"],
    "additionalProperties": False,
}


class PlayAnswer(NamedTuple):
    changed: str
    plan: str
    notes: str
    actions: list[Action]


def parse_play(text: str, available: list[int]) -> PlayAnswer | None:
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        actions = parse_plan(json.dumps({"plan": data["actions"]}), available)
        return PlayAnswer(str(data.get("changed", "")), str(data.get("plan", "")), str(data.get("notes", "")),
                          actions[:LLM_PLAY_MAX_ACTIONS])
    except (ValueError, KeyError, TypeError):
        return None


@dataclass(slots=True)
class Seen:
    grid: np.ndarray
    action: Action | None  # the action that led to this screen
    changed: bool
    step: int


class ClickHelper:
    """Reki's click helpers (a utility class)."""

    @staticmethod
    def kind_at(grid: np.ndarray, x: int, y: int) -> tuple | None:
        """(colour, shape) of the object under a cell; None outside the screen."""
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
            return None
        for blob in ScreenReader.find_objects(grid):
            if blob.bbox[0] <= y <= blob.bbox[2] and blob.bbox[1] <= x <= blob.bbox[3] \
                    and bool(((blob.cells[:, 0] == y) & (blob.cells[:, 1] == x)).any()):
                return blob.colour, ScreenReader.shape_of(blob)
        return None

    @staticmethod
    def snap(grid: np.ndarray, x: int, y: int, radius: int = 2) -> tuple[int, int]:
        """A click on the background next to a small object lands on the object (vision models miss by a cell or
        two); anything else stays where it is."""
        counts = np.bincount(grid.ravel(), minlength=16)
        background = int(counts.argmax())
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] != background:
            return x, y
        best: tuple[int, int, int] | None = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1] and grid[yy, xx] != background:
                    d = abs(dy) + abs(dx)
                    if best is None or d < best[0]:
                        best = (d, xx, yy)
        return (best[1], best[2]) if best else (x, y)


@dataclass(slots=True)
class Player:
    """The model's side of play mode: what it has seen, its notes and plan, the actions it queued."""
    frames: deque = field(default_factory=lambda: deque(maxlen=LLM_PLAY_FRAMES))
    notes: str = ""
    plan: str = ""
    changed: str = ""
    queue: list[Action] = field(default_factory=list)
    dead_kinds: set = field(default_factory=set)  # (colour, shape) never changed anything when clicked (this level)
    live_kinds: set = field(default_factory=set)  # ... and those that did
    step: int = 0
    last_why: str = ""

    def observe(self, grid: np.ndarray, action: Action | None, before: np.ndarray | None, changed: bool) -> None:
        """The screen after `action`; learn which clicked kinds are dead."""
        self.step += 1
        self.frames.append(Seen(grid, action, changed, self.step))
        if action is not None and action[0] == 6 and before is not None:
            kind = ClickHelper.kind_at(before, action[1], action[2])
            if kind is not None:
                if changed:
                    self.live_kinds.add(kind)
                    self.dead_kinds.discard(kind)
                elif kind not in self.live_kinds:
                    self.dead_kinds.add(kind)

    def new_level(self) -> None:
        self.queue, self.plan, self.dead_kinds, self.live_kinds = [], "", set(), set()

    def drop_plan(self) -> None:
        self.queue = []

    def legal(self, action: Action, grid: np.ndarray, available: list[int]) -> Action | None:
        """The action as it will be sent (clicks snapped onto objects), or None: not offered, or a dead click."""
        if action[0] not in available:
            return None
        if action[0] != 6:
            return action
        x, y = ClickHelper.snap(grid, action[1], action[2])
        if ClickHelper.kind_at(grid, x, y) in self.dead_kinds:
            return None
        return Action(6, x, y)

    def parts(self, level: int, lives_lost: int, available: list[int]) -> list[dict]:
        """The user message: text, then each recent screen as an image (oldest first)."""
        names = ", ".join("ACTION6 x y" if a == 6 else f"ACTION{a}" for a in sorted(available) if a != 0)
        lines = [f"Level {level + 1}. Lives lost so far: {lives_lost}. Actions in this game: {names}."]
        if LLM_PLAY_NOTES and self.notes:
            lines.append(f"Your notes so far: {self.notes}")
        if self.plan:
            lines.append(f"Your last plan: {self.plan}")
        if self.dead_kinds:
            lines.append(f"{len(self.dead_kinds)} kind(s) of object never changed anything when clicked on this "
                         f"level; clicks on them are skipped.")
        lines.append("Recent screens, oldest first (the last one is now):")
        parts: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
        frames = list(self.frames)
        for i, seen in enumerate(frames):
            label = "the level start" if seen.action is None else (
                f"after {_name(seen.action)} ({'the screen changed' if seen.changed else 'NOTHING changed'})")
            label = f"Screen {i + 1} of {len(frames)}{' (NOW)' if i == len(frames) - 1 else ''}: {label}"
            parts.append({"type": "text", "text": label})
            mark = (seen.action[1], seen.action[2]) if seen.action is not None and seen.action[0] == 6 else None
            png = FramePainter.png(seen.grid, LLM_PLAY_SCALE, mark)
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
        if LLM_PLAY_TEXT_GRID and frames:
            rows = "\n".join(f"{y:02d} " + "".join(f"{v:x}" for v in row) for y, row in enumerate(frames[-1].grid))
            parts.append({"type": "text", "text": "The current screen as hex digits (row, then x=0..63):\n" + rows})
        return parts

    def ask(self, adviser: Advisor, level: int, lives_lost: int, available: list[int], tag: str) -> bool:
        """One call: the model sees the recent screens and queues its next actions; False if no usable answer."""
        text = adviser.ask_parts(self.parts(level, lives_lost, available), tag, PLAY_SYSTEM_PROMPT, PLAY_SCHEMA)
        answer = parse_play(text, available) if text else None
        if answer is None or not answer.actions:
            return False
        self.take(answer)
        return True

    def take(self, answer: PlayAnswer) -> None:
        self.changed, self.plan = answer.changed, answer.plan
        if LLM_PLAY_NOTES and answer.notes:
            self.notes = answer.notes
        self.queue = list(answer.actions)
        self.last_why = f"LLM plays: {answer.plan}"[:200]


def _name(action: Action) -> str:
    return f"ACTION6 {action[1]} {action[2]}" if action[0] == 6 else f"ACTION{action[0]}"
