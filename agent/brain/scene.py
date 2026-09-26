"""SCENE — the objects on screen, with ids that stay the same during a level, and what each action did to them.

Used by the LLM advisor's object questions (llm.PromptWriter.object_question): instead of 64 rows of hex digits
the model gets numbered objects, relations the brain can compute itself, and what actions did to which object.

    SceneObject      one object: id, colour, size, box, shape
    Change           what one action did to one object: moved (dx, dy), appeared, disappeared, recoloured
    SceneTracker     per level: numbers the objects, matches them across screens, records changes per action
    RelationFinder   facts the brain can see itself: same size in different colours, mirror images across a line
Nothing here decides anything; it only describes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .look import ScreenReader
from .settings import BACKGROUND_FRACTION
from .world import Action, World

COMPOSITE_MIN_MOVES = 2  # parts must move together (touching) this often, and never apart, to count as one object

COLOUR_NAMES = ["white", "light grey", "grey", "dark grey", "darker grey", "black", "magenta", "pink", "red",
                "blue", "light blue", "yellow", "orange", "maroon", "green", "purple"]


@dataclass(slots=True)
class SceneObject:
    id: int
    colour: int
    size: int
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
    shape: tuple  # ScreenReader.shape_of: (height, width, mask bytes)
    background: bool = False  # covers a large part of the screen

    @property
    def top_left(self) -> tuple[int, int]:
        return self.bbox[0], self.bbox[1]

    def describe(self) -> str:
        y0, x0, y1, x1 = self.bbox
        where = f"x{x0}-{x1} y{y0}-{y1}" if (x0, y0) != (x1, y1) else f"x{x0} y{y0}"
        size = f"{x1 - x0 + 1}x{y1 - y0 + 1}"
        return f"#{self.id} {COLOUR_NAMES[self.colour]}, {self.size} cells, {size} box at {where}" \
            + (" (background)" if self.background else "")


@dataclass(slots=True)
class Change:
    object_id: int
    kind: str  # "moved" | "appeared" | "disappeared" | "recoloured" | "reshaped"
    dx: int = 0
    dy: int = 0
    note: str = ""

    def describe(self) -> str:
        if self.kind == "moved":
            parts = []
            if self.dx:
                parts.append(f"{'right' if self.dx > 0 else 'left'} {abs(self.dx)}")
            if self.dy:
                parts.append(f"{'down' if self.dy > 0 else 'up'} {abs(self.dy)}")
            return f"#{self.object_id} moved {' and '.join(parts)}"
        return f"#{self.object_id} {self.kind}{(' ' + self.note) if self.note else ''}"


@dataclass(slots=True)
class Observation:
    """One action and what it did to the objects."""
    action: Action
    changes: list[Change]
    clicked: int | None = None  # id of the object under a click

    def describe(self) -> str:
        a = self.action
        name = f"ACTION6 at x{a[1]} y{a[2]}" + (f" (on #{self.clicked})" if self.clicked is not None else "") \
            if a[0] == 6 else f"ACTION{a[0]}"
        return f"{name}: {self.effects()}"

    def effects(self) -> str:
        """The changes, with objects that moved the same way grouped: "#7 #8 #9 moved right 3"."""
        if not self.changes:
            return "no object changed"
        groups: dict[tuple, list[int]] = {}
        for c in self.changes:
            groups.setdefault((c.kind, c.dx, c.dy, c.note), []).append(c.object_id)
        parts = []
        for (kind, dx, dy, note), ids in groups.items():
            text = Change(ids[0], kind, dx, dy, note).describe()
            parts.append(" ".join(f"#{i}" for i in ids) + text[len(f"#{ids[0]}"):])
        return ", ".join(parts)


@dataclass(slots=True)
class LevelScene:
    objects: list[SceneObject] = field(default_factory=list)
    next_id: int = 0
    observations: deque = field(default_factory=lambda: deque(maxlen=40))
    parent: dict[int, int] = field(default_factory=dict)  # objects that move as one: member id -> other id
    together: dict[tuple[int, int], int] = field(default_factory=dict)  # pair -> times moved by the same step
    apart: dict[tuple[int, int], int] = field(default_factory=dict)  # pair -> times one moved differently

    def root(self, object_id: int) -> int:
        """The id that stands for the whole composite object (the smallest id of the group)."""
        while self.parent.get(object_id, object_id) != object_id:
            object_id = self.parent[object_id]
        return object_id

    def join(self, a: int, b: int) -> None:
        ra, rb = self.root(a), self.root(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def rebuild_composites(self) -> None:
        """Parts of one object: moved by the same step, touching, at least twice, and never apart."""
        self.parent = {}
        for (a, b), n in self.together.items():
            if n >= COMPOSITE_MIN_MOVES and not self.apart.get((a, b)):
                self.join(a, b)

    def members(self, root: int) -> list[int]:
        ids = {o.id for o in self.objects} | set(self.parent)
        return sorted(i for i in ids if i != root and self.root(i) == root)


class SceneTracker:
    """Numbers the objects of a level and records what each action did to them (writes only its own state)."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.levels: dict[int, LevelScene] = {}

    def scene(self, level: int) -> LevelScene:
        return self.levels.setdefault(level, LevelScene())

    def objects_in(self, grid: np.ndarray) -> list[SceneObject]:
        """Objects on this screen (counter pixels hidden), largest first; ids not assigned yet (-1)."""
        picture = self.world.counters.hide(grid)
        visible = ~self.world.counters.ignore
        objs = []
        for blob in ScreenReader.find_objects(picture, visible=visible):
            objs.append(SceneObject(-1, blob.colour, blob.size, blob.bbox, ScreenReader.shape_of(blob),
                                    background=blob.size > BACKGROUND_FRACTION * grid.size))
        objs.sort(key=lambda o: (-o.size, o.bbox))
        return objs

    def update(self, level: int, grid: np.ndarray, last_action: Action | None) -> Observation | None:
        """Match this screen's objects to the level's known ones; returns what `last_action` changed."""
        scene = self.scene(level)
        now = self.objects_in(grid)
        before = scene.objects
        if not before:
            for obj in now:
                obj.id = scene.next_id
                scene.next_id += 1
            scene.objects = now
            return None
        changes = self._match(scene, before, now)
        scene.objects = now
        if last_action is None or last_action[0] == 0:
            return None
        self._join_moving_together(scene, changes)
        changes = self._as_composites(scene, changes)
        clicked = self._object_at(before, last_action) if last_action[0] == 6 else None
        if clicked is not None:
            clicked = scene.root(clicked)
        observation = Observation(last_action, changes, clicked)
        scene.observations.append(observation)
        return observation

    def _match(self, scene: LevelScene, before: list[SceneObject], now: list[SceneObject]) -> list[Change]:
        """Give every object on the new screen its id, and list what changed."""
        changes: list[Change] = []
        used_now, used_before = self._pair_same_objects(before, now, changes)
        unmatched_before = [b for j, b in enumerate(before) if j not in used_before]
        reshaped: set[int] = set()
        for obj in (o for i, o in enumerate(now) if i not in used_now):
            self._match_changed(scene, obj, unmatched_before, changes, reshaped)
        for prev in unmatched_before:
            if not prev.background and prev.id not in reshaped:
                changes.append(Change(prev.id, "disappeared"))
        return changes

    @staticmethod
    def _join_moving_together(scene: LevelScene, changes: list[Change]) -> None:
        """Count, for touching objects, how often they moved by the same step and how often not (one moved, the
        other stood still or went elsewhere). Objects split into pieces (partly covered) never count as moving."""
        pieces: dict[int, int] = {}
        for o in scene.objects:
            pieces[o.id] = pieces.get(o.id, 0) + 1
        whole = {o.id: o for o in scene.objects if pieces[o.id] == 1 and not o.background}
        step = {c.object_id: (c.dx, c.dy) for c in changes if c.kind == "moved" and c.object_id in whole}
        if not step:
            return
        ids = sorted(whole)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if (a not in step and b not in step) or not _touch(whole[a].bbox, whole[b].bbox):
                    continue
                pair = (a, b)
                if step.get(a) == step.get(b):
                    scene.together[pair] = scene.together.get(pair, 0) + 1
                else:
                    scene.apart[pair] = scene.apart.get(pair, 0) + 1
        scene.rebuild_composites()

    @staticmethod
    def _as_composites(scene: LevelScene, changes: list[Change]) -> list[Change]:
        """Report each composite object once, under its root id."""
        out: list[Change] = []
        seen: set[tuple] = set()
        for c in changes:
            key = (scene.root(c.object_id), c.kind, c.dx, c.dy)
            if key in seen:
                continue
            seen.add(key)
            out.append(Change(key[0], c.kind, c.dx, c.dy, c.note))
        return out

    @staticmethod
    def _pair_same_objects(before: list[SceneObject], now: list[SceneObject],
                           changes: list[Change]) -> tuple[set[int], set[int]]:
        """Same colour + shape = same object. Identical twins move together more often than not: within each
        group, pairs following the group's most common displacement are matched first (five dots that all moved
        down 3 pair with themselves even where one lands on another's old spot), then the nearest remaining."""
        groups: dict[tuple, tuple[list[int], list[int]]] = {}
        for i, obj in enumerate(now):
            groups.setdefault((obj.colour, obj.shape), ([], []))[0].append(i)
        for j, prev in enumerate(before):
            if (prev.colour, prev.shape) in groups:
                groups[(prev.colour, prev.shape)][1].append(j)
        pairs = []
        for now_ids, before_ids in groups.values():
            deltas = [((now[i].bbox[0] - before[j].bbox[0], now[i].bbox[1] - before[j].bbox[1]), i, j)
                      for i in now_ids for j in before_ids]
            counts: dict[tuple[int, int], int] = {}
            for d, _i, _j in deltas:
                counts[d] = counts.get(d, 0) + 1
            common = max(counts, key=lambda d: (counts[d], -abs(d[0]) - abs(d[1]))) if counts else (0, 0)
            pairs += [(d != common, abs(d[0]) + abs(d[1]), i, j) for d, i, j in deltas]
        pairs.sort()
        used_now: set[int] = set()
        used_before: set[int] = set()
        for _not_common, _dist, i, j in pairs:
            if i in used_now or j in used_before:
                continue
            used_now.add(i)
            used_before.add(j)
            obj, prev = now[i], before[j]
            obj.id = prev.id
            dy, dx = obj.bbox[0] - prev.bbox[0], obj.bbox[1] - prev.bbox[1]
            if (dx or dy) and not obj.background:
                changes.append(Change(obj.id, "moved", dx, dy))
        return used_now, used_before

    @staticmethod
    def _match_changed(scene: LevelScene, obj: SceneObject, unmatched_before: list[SceneObject],
                       changes: list[Change], reshaped: set[int]) -> None:
        """An object with no identical partner: recoloured in place, a background region that changed shape,
        the same object partly covered or split into pieces (same colour, overlapping its old place), or new."""
        recoloured = next((b for b in unmatched_before if b.bbox == obj.bbox and b.colour != obj.colour), None)
        if recoloured is not None:
            unmatched_before.remove(recoloured)
            obj.id = recoloured.id
            if not obj.background:
                changes.append(Change(obj.id, "recoloured",
                                      note=f"{COLOUR_NAMES[recoloured.colour]} -> {COLOUR_NAMES[obj.colour]}"))
            return
        region = next((b for b in unmatched_before if b.background and obj.background and b.colour == obj.colour),
                      None)
        if region is not None:
            unmatched_before.remove(region)
            obj.id = region.id
            return
        # same colour where it used to be: the same object, partly covered or split (pieces share its id)
        old = [b for b in unmatched_before + [b for b in scene.objects if b.id in reshaped]
               if b.colour == obj.colour and not b.background and _touch(b.bbox, obj.bbox, margin=4)]
        if old:
            obj.id = max(old, key=lambda b: _overlap(b.bbox, obj.bbox)).id
            if obj.id not in reshaped:
                reshaped.add(obj.id)
                changes.append(Change(obj.id, "reshaped", note="(partly covered, cut or grown)"))
            return
        obj.id = scene.next_id
        scene.next_id += 1
        if not obj.background:
            changes.append(Change(obj.id, "appeared", note=f"({obj.describe()})"))

    @staticmethod
    def _object_at(objects: list[SceneObject], action: Action) -> int | None:
        x, y = action[1], action[2]
        hits = [o for o in objects if o.bbox[1] <= x <= o.bbox[3] and o.bbox[0] <= y <= o.bbox[2]]
        return min(hits, key=lambda o: o.size).id if hits else None


def _touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    """Boxes (y0, x0, y1, x1) overlap or come within `margin` cells of each other."""
    return a[0] <= b[2] + margin and b[0] <= a[2] + margin and a[1] <= b[3] + margin and b[1] <= a[3] + margin


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    h = min(a[2], b[2]) - max(a[0], b[0]) + 1
    w = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return max(0, h) * max(0, w)


class RelationFinder:
    """Relations between objects the brain can compute itself (utility class)."""

    @staticmethod
    def find(objects: list[SceneObject], height: int, width: int) -> list[str]:
        facts: list[str] = []
        things = [o for o in objects if not o.background and o.size > 1]
        axes = [o for o in objects if RelationFinder._is_vertical_line(o, height)
                or RelationFinder._is_horizontal_line(o, width)]
        for axis in axes:
            for a in things:
                if a is axis:
                    continue
                for b in things:
                    if b is axis or b.id <= a.id or not RelationFinder._mirrored(a, b, axis, height):
                        continue
                    facts.append(f"#{b.id} is where #{a.id} would be if mirrored across #{axis.id}")
        for i, a in enumerate(things):
            for b in things[i + 1:]:
                ha, wa = a.bbox[2] - a.bbox[0], a.bbox[3] - a.bbox[1]
                hb, wb = b.bbox[2] - b.bbox[0], b.bbox[3] - b.bbox[1]
                if a.colour != b.colour and a.size == b.size and (ha, wa) == (hb, wb):
                    same = "same shape" if a.shape == b.shape else "same size and box, different shape"
                    facts.append(f"#{a.id} and #{b.id}: {same}, different colours")
        return facts[:12]

    @staticmethod
    def _is_vertical_line(o: SceneObject, height: int) -> bool:
        y0, x0, y1, x1 = o.bbox
        return x1 - x0 <= 3 and y1 - y0 + 1 >= 0.8 * height and o.size >= 0.8 * (x1 - x0 + 1) * (y1 - y0 + 1)

    @staticmethod
    def _is_horizontal_line(o: SceneObject, width: int) -> bool:
        y0, x0, y1, x1 = o.bbox
        return y1 - y0 <= 3 and x1 - x0 + 1 >= 0.8 * width and o.size >= 0.8 * (x1 - x0 + 1) * (y1 - y0 + 1)

    @staticmethod
    def _mirrored(a: SceneObject, b: SceneObject, axis: SceneObject, height: int) -> bool:
        ay0, ax0, ay1, ax1 = axis.bbox
        y0, x0, y1, x1 = a.bbox
        by0, bx0, by1, bx1 = b.bbox
        if RelationFinder._is_vertical_line(axis, height):
            twice = ax0 + ax1  # mirror x -> twice - x
            return (abs(by0 - y0) <= 1 and abs(by1 - y1) <= 1
                    and abs(bx0 - (twice - x1)) <= 1 and abs(bx1 - (twice - x0)) <= 1)
        twice = ay0 + ay1
        return (abs(bx0 - x0) <= 1 and abs(bx1 - x1) <= 1
                and abs(by0 - (twice - y1)) <= 1 and abs(by1 - (twice - y0)) <= 1)
