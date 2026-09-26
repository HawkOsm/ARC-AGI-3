"""CONTROL — what the keys move right now, and what changes that (selection).

Many games let the keys move one object at a time; clicking an object, or a switch key such as ACTION5, picks
which one (ar25 level 2: the keys move the mirror line until ACTION5 or a click selects the grey shape).
ControlMap learns this from the level's observations:
    mode      one state of "what the keys move": for each object, the step each key gives it
    switch    a click or non-direction key (ACTION5..7) that changed something - usually only small marks (a
              selection marker moved): what the keys control may have changed. The next key press shows the mode.
Nothing here knows any game: a game without selection simply has one mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .scene import COLOUR_NAMES, LevelScene, Observation, SceneObject
from .shapes import mask_of
from .world import Action

MARK_SIZE = 2  # objects this small (selection dots, markers) never count as things the keys move
KEYS = (1, 2, 3, 4)


@dataclass(slots=True)
class ControlMode:
    steps: dict[int, dict[int, tuple[int, int]]] = field(default_factory=dict)  # object -> key -> (dy, dx)
    entered_by: list[Action] = field(default_factory=list)  # switch actions seen to lead into this mode
    clicked: list[int] = field(default_factory=list)  # objects whose click led into this mode

    def keys_for(self, object_id: int) -> dict[int, tuple[int, int]]:
        return self.steps.get(object_id, {})


class ControlMap:
    """The level's control modes and switches (per level; written only by `observe`)."""

    def __init__(self) -> None:
        self.modes: list[ControlMode] = []
        self.current: int | None = None  # None: unknown (no key pressed yet, or right after a switch)
        self.pending: Action | None = None  # the switch whose effect the next key press will show
        self.pending_click: int | None = None
        self.switch_ids: set[int] = set()  # action ids seen acting as switches (5, 6, ...)
        self.dead_clicks: set[int] = set()  # objects whose click changed nothing (not selectable)

    # ── learning ───────────────────────────────────────────────────────────

    @staticmethod
    def real_moves(scene: LevelScene, obs: Observation) -> dict[int, tuple[int, int]]:
        """Objects that really moved: bigger than a mark, and one consistent step (a composite whose parts
        jumped different ways was re-matched, not moved)."""
        sizes: dict[int, int] = {}
        for o in scene.objects:
            root = scene.root(o.id)
            sizes[root] = sizes.get(root, 0) + o.size
        steps: dict[int, set[tuple[int, int]]] = {}
        for c in obs.changes:
            if c.kind == "moved":
                steps.setdefault(c.object_id, set()).add((c.dy, c.dx))
        return {i: next(iter(s)) for i, s in steps.items() if len(s) == 1 and sizes.get(i, 0) > MARK_SIZE}

    @staticmethod
    def is_switch(scene: LevelScene, obs: Observation) -> bool:
        """Something changed, but no real object moved: a selection (or other state) switch."""
        return bool(obs.changes) and not ControlMap.real_moves(scene, obs)

    def observe(self, scene: LevelScene, obs: Observation) -> str:
        """Learn from one observation; returns "key", "switch" or ""."""
        action = obs.action
        moves = self.real_moves(scene, obs)
        if moves and action[0] in KEYS:
            index = self._mode_for(action[0], moves)
            mode = self.modes[index]
            for obj, step in moves.items():
                mode.steps.setdefault(obj, {})[action[0]] = step
            if self.pending is not None:
                if self.pending not in mode.entered_by:
                    mode.entered_by.append(self.pending)
                if self.pending_click is not None and self.pending_click not in mode.clicked:
                    mode.clicked.append(self.pending_click)
                self.pending, self.pending_click = None, None
            self.current = index
            return "key"
        if action[0] == 6 and not obs.changes and obs.clicked is not None:
            self.dead_clicks.add(obs.clicked)
        if obs.changes and action[0] not in KEYS:  # a click or switch key that changed anything may have switched
            self.switch_ids.add(action[0])            # (marker changes can look like jumps: the next key decides)
            self.pending, self.pending_click, self.current = action, obs.clicked, None
            return "switch"
        return ""

    def _mode_for(self, key: int, moves: dict[int, tuple[int, int]]) -> int:
        """The mode these moves belong to. With no switch since the last key press it is the current mode (unless
        a key now moves an object the opposite way); after a switch, a known mode must match exactly - the same
        key moving the same object the same way, or the same set of objects moving - else it is a new mode."""
        if self.pending is None and self.current is not None \
                and not self._conflicts(self.modes[self.current], key, moves):
            return self.current
        best, best_score = -1, 0
        for index, mode in enumerate(self.modes):
            if self._conflicts(mode, key, moves):
                continue
            score = sum(2 for obj in moves if key in mode.keys_for(obj))
            if set(moves) == set(mode.steps):
                score += 2
            if score > best_score:
                best, best_score = index, score
        if best >= 0 and best_score >= 2:
            return best
        self.modes.append(ControlMode())
        return len(self.modes) - 1

    @staticmethod
    def _conflicts(mode: ControlMode, key: int, moves: dict[int, tuple[int, int]]) -> bool:
        for obj, (dy, dx) in moves.items():
            known = mode.keys_for(obj).get(key)
            if known is not None and (_sign(known[0]), _sign(known[1])) != (_sign(dy), _sign(dx)):
                return True
        return False

    # ── using it ───────────────────────────────────────────────────────────

    def best_mode(self, object_ids: list[int]) -> int | None:
        """The mode in which the keys move one of these objects in the most directions."""
        best, most = None, 0
        for index, mode in enumerate(self.modes):
            for obj in object_ids:
                n = len({(_sign(dy), _sign(dx)) for dy, dx in mode.keys_for(obj).values()})
                if n > most or (n == most and n and index == self.current):
                    best, most = index, n
        return best

    def steps(self, mode: int, object_ids: list[int]) -> dict[int, tuple[int, int]]:
        found: dict[int, tuple[int, int]] = {}
        for obj in object_ids:
            for key, step in self.modes[mode].keys_for(obj).items():
                found.setdefault(key, step)
        return found

    def how_to_enter(self, mode: int, scene: LevelScene, mover: int, available: list[int]) -> Action | None:
        """The action most likely to make `mode` the current one, in order: click an object whose click led there
        before; a switch key that led there before; if clicks select what they click, click one of the mode's
        movers (never one whose click did nothing); any switch key seen (it may cycle through the modes)."""
        objects = {o.id: o for o in scene.objects if scene.root(o.id) == o.id}
        target = self.modes[mode]
        if 6 in available:
            for obj in target.clicked:
                if obj in objects:
                    return click_on(objects[obj])
        for action in target.entered_by:
            if action[0] != 6 and action[0] in available:
                return action
        if 6 in available and self.clicks_select():
            for obj in [mover] + sorted(target.steps, key=lambda i: -objects[i].size if i in objects else 0):
                if obj in objects and obj not in self.dead_clicks and objects[obj].size > MARK_SIZE:
                    return click_on(objects[obj])
        for action_id in sorted(self.switch_ids):
            if action_id != 6 and action_id in available:
                return Action(action_id)
        return None

    def clicks_select(self) -> bool:
        """Has clicking an object ever made the keys move that very object?"""
        return any(obj in mode.steps for mode in self.modes for obj in mode.clicked)

    def describe(self, scene: LevelScene) -> list[str]:
        """For the LLM: what the keys move in each mode, and how each mode was reached."""
        if not self.modes:
            return []
        colour = {o.id: COLOUR_NAMES[o.colour] for o in scene.objects}
        lines = []
        if len(self.modes) > 1 or self.switch_ids:
            lines.append(f"The keys move different things at different times ({len(self.modes)} control mode(s) "
                         f"seen); a switch changed only small marks on screen.")
        for index, mode in enumerate(self.modes):
            parts = []
            for obj, keys in sorted(mode.steps.items()):
                moves = ", ".join(f"ACTION{k} {_direction(dy, dx)}" for k, (dy, dx) in sorted(keys.items()))
                parts.append(f"#{obj} ({colour.get(obj, 'gone')}): {moves}")
            how = [f"clicking #{o}" for o in mode.clicked] + \
                [f"ACTION{a[0]}" for a in mode.entered_by if a[0] != 6]
            head = f"mode {index + 1}" + (" (NOW)" if index == self.current else "") + \
                (f", reached by {' or '.join(how)}" if how else "")
            lines.append(f"{head}: " + "; ".join(parts))
        if self.clicks_select():
            lines.append("Clicking an object selects it: the keys then move the clicked object.")
        return lines


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _direction(dy: int, dx: int) -> str:
    parts = []
    if dy:
        parts.append(f"{'down' if dy > 0 else 'up'} {abs(dy)}")
    if dx:
        parts.append(f"{'right' if dx > 0 else 'left'} {abs(dx)}")
    return " ".join(parts) or "stays"


def click_on(obj: SceneObject) -> Action:
    """Click a cell of the object itself (a box centre can be empty for an L or a ring)."""
    mask = mask_of(obj)
    y0, x0 = obj.bbox[0], obj.bbox[1]
    h, w = mask.shape
    cells = [(abs(y - h // 2) + abs(x - w // 2), y, x) for y in range(h) for x in range(w) if mask[y, x]]
    _d, y, x = min(cells) if cells else (0, h // 2, w // 2)
    return Action(6, x0 + x, y0 + y)
