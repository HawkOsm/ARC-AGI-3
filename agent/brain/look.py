"""LOOK — everything about reading the screen.

Reading helpers (pure: they never change what the agent believes):
    settled_grid(frame)      the final picture after an action (animations have many layers)
    find_objects(grid)       every same-colour connected blob: colour, size, cells, centre cell
    changed_blobs(diff)      connected groups of changed cells between two pictures
    shape_of(object)         an object's exact shape, independent of where it is

The first look at a level (writes what it saw into the world):
    first_look(world, level, grid)
        * every distinct thing on screen becomes an item of some KIND (colour + exact shape),
          catalogued in world.kinds with purpose "?" until play reveals it
        * very large items are probably background
        * patterns worth remembering go to world.patterns:
            "repeated"    the same kind appears several times (tiles, buttons, coins, walls ...)
            "same_shape"  one shape in several colours (key/lock, target/source, before/after ...)
        * the level's inventory is stored in world.levels[level].items
      Nothing here is game-specific: it only describes what is visible, so later learning
      (learn.learn_what_touching_does) can attach purposes to kinds.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .settings import BACKGROUND_FRACTION, MAX_PATTERNS_PER_LEVEL, PATTERN_MIN_ITEM_CELLS
from .world import World


def settled_grid(frame_layers) -> np.ndarray | None:
    """The last layer of a (possibly animated) frame, as a uint8 64x64 array."""
    if frame_layers is None or len(frame_layers) == 0:
        return None
    return np.asarray(frame_layers[-1], dtype=np.uint8)


def find_objects(grid: np.ndarray, visible: np.ndarray | None = None) -> list[dict]:
    """Same-colour 4-connected blobs (row runs + union-find, ~2 ms on real frames).

    Each object: {"colour", "size", "cells" (N x 2 array of (y, x)), "centre_cell"
    (the member cell nearest the centroid - always ON the object, so safe to click),
    "bbox" (y0, x0, y1, x1)}. Cells where `visible` is False are skipped.
    """
    runs, runs_per_row = _row_runs(grid, visible)
    members_of: dict[int, list[int]] = {}
    for i, root in enumerate(_join_touching_runs(runs, runs_per_row)):
        members_of.setdefault(root, []).append(i)
    objects = []
    for members in members_of.values():
        ys = np.concatenate([np.full(runs[i][2] - runs[i][1], runs[i][0]) for i in members])
        xs = np.concatenate([np.arange(runs[i][1], runs[i][2]) for i in members])
        cells = np.stack([ys, xs], axis=1)
        cy, cx = cells.mean(axis=0)
        nearest = int(np.argmin((cells[:, 0] - cy) ** 2 + (cells[:, 1] - cx) ** 2))
        objects.append(dict(colour=runs[members[0]][3], size=len(cells), cells=cells,
                            centre_cell=(int(cells[nearest, 0]), int(cells[nearest, 1])),
                            bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))))
    return objects


def changed_blobs(changed: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected groups of True cells in a boolean 'changed' mask."""
    ys, xs = np.nonzero(changed)
    todo = set(zip(ys.tolist(), xs.tolist(), strict=True))
    blobs = []
    while todo:
        start = todo.pop()
        blob, queue = [start], deque([start])
        while queue:
            y, x = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (y + dy, x + dx)
                    if n in todo:
                        todo.remove(n)
                        blob.append(n)
                        queue.append(n)
        blobs.append(blob)
    return blobs


# ── the first look at a level ───────────────────────────────────────────────


def first_look(world: World, level: int, grid: np.ndarray) -> None:
    ignore = world.counters["ignore"]  # counter pixels are not items
    picture = grid.copy()
    picture[ignore] = 255
    objects = find_objects(picture, visible=~ignore)
    items = []
    for obj in objects:
        kind_key = (obj["colour"], shape_of(obj))
        kind = world.kinds.get(kind_key)
        if kind is None:
            y0, x0, y1, x1 = obj["bbox"]
            kind = world.kinds[kind_key] = {
                "name": f"colour{obj['colour']}-{y1 - y0 + 1}x{x1 - x0 + 1}-#{len(world.kinds)}",
                "colour": obj["colour"],
                "size": obj["size"],
                "purpose": "background?" if obj["size"] > BACKGROUND_FRACTION * grid.size else "?",
                "seen_in_levels": set(),
            }
        kind["seen_in_levels"].add(level)
        items.append({"kind": kind_key, "colour": obj["colour"], "size": obj["size"],
                      "centre": obj["centre_cell"], "bbox": obj["bbox"]})
    world.levels[level].items = items
    _notice_patterns(world, level, items)


def shape_of(obj: dict) -> tuple:
    """Exact shape, independent of position: (height, width, cell mask bytes)."""
    y0, x0, y1, x1 = obj["bbox"]
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    mask[obj["cells"][:, 0] - y0, obj["cells"][:, 1] - x0] = True
    return mask.shape[0], mask.shape[1], mask.tobytes()


def _notice_patterns(world: World, level: int, items: list[dict]) -> None:
    per_kind: dict[tuple, int] = defaultdict(int)
    colours_per_shape: dict[tuple, set] = defaultdict(set)
    for item in items:
        if item["size"] < PATTERN_MIN_ITEM_CELLS or world.kinds[item["kind"]]["purpose"] == "background?":
            continue
        per_kind[item["kind"]] += 1
        colours_per_shape[item["kind"][1]].add(item["colour"])
    found = []
    for kind_key, count in per_kind.items():
        if count >= 2:
            found.append({"type": "repeated", "kinds": [world.kinds[kind_key]["name"]], "count": count})
    for shape, colours in colours_per_shape.items():
        if len(colours) >= 2:
            names = [world.kinds[(c, shape)]["name"] for c in sorted(colours)]
            found.append({"type": "same_shape", "kinds": names, "count": len(names)})
    # few, informative patterns first: a pair beats a 60-tile floor
    found.sort(key=lambda p: (p["type"] != "same_shape", p["count"]))
    for p in found[:MAX_PATTERNS_PER_LEVEL]:
        p["level"] = level
        world.patterns.append(p)


# ── helpers for find_objects ────────────────────────────────────────────────

def _row_runs(grid: np.ndarray, visible: np.ndarray | None):
    """Maximal same-colour horizontal runs (y, x0, x1_exclusive, colour), split at
    invisible cells, plus the run indices of each row."""
    h, w = grid.shape
    runs, runs_per_row = [], []
    for y in range(h):
        row = grid[y]
        cut = np.flatnonzero(row[1:] != row[:-1]) + 1
        these = []
        for x0, x1 in zip(np.concatenate(([0], cut)).tolist(), np.concatenate((cut, [w])).tolist(), strict=True):
            if visible is None or visible[y, x0:x1].all():
                parts = [(x0, x1)]
            else:
                xs = np.flatnonzero(visible[y, x0:x1]) + x0
                breaks = np.flatnonzero(np.diff(xs) > 1) + 1
                parts = [(int(q[0]), int(q[-1]) + 1) for q in np.split(xs, breaks) if len(q)]
            for a, b in parts:
                these.append(len(runs))
                runs.append((y, a, b, int(row[x0])))
        runs_per_row.append(these)
    return runs, runs_per_row


def _join_touching_runs(runs: list, runs_per_row: list) -> list[int]:
    """Union-find over vertically overlapping same-colour runs; returns each run's root."""
    parent = list(range(len(runs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(1, len(runs_per_row)):
        above = runs_per_row[y - 1]
        j = 0
        for i in runs_per_row[y]:
            _, a0, a1, colour = runs[i]
            while j < len(above) and runs[above[j]][2] <= a0:
                j += 1
            k = j
            while k < len(above) and runs[above[k]][1] < a1:
                if runs[above[k]][3] == colour:
                    ri, rk = root(i), root(above[k])
                    if ri != rk:
                        parent[ri] = rk
                k += 1
    return [root(i) for i in range(len(runs))]
