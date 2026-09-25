"""Frame -> state: settled grid, volatile-cell masking, state keys, components."""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import (
    AVATAR_MIN_CONSISTENCY,
    AVATAR_MIN_OBS,
    BORDER_BAND,
    FREQ_MASK_MIN_TRANSITIONS,
    FREQ_MASK_RATE,
    TICK_LINE_MIN_CELLS,
    TICK_MAX_CELLS,
)


def settled_grid(frame_layers) -> np.ndarray | None:
    """Last layer of a (possibly animated) multi-layer frame, as uint8."""
    if frame_layers is None or len(frame_layers) == 0:
        return None
    return np.asarray(frame_layers[-1], dtype=np.uint8)


def _row_runs(grid: np.ndarray, valid: np.ndarray | None):
    """Maximal same-colour horizontal runs, split at invalid cells.
    Returns runs [(y, x0, x1_exclusive, colour)] and per-row run indices."""
    h, w = grid.shape
    runs, row_runs = [], []
    for y in range(h):
        row = grid[y]
        cut = np.flatnonzero(row[1:] != row[:-1]) + 1
        these = []
        for x0, x1 in zip(np.concatenate(([0], cut)).tolist(), np.concatenate((cut, [w])).tolist(), strict=True):
            if valid is None or valid[y, x0:x1].all():
                parts = [(x0, x1)]
            else:
                xs = np.flatnonzero(valid[y, x0:x1]) + x0
                brk = np.flatnonzero(np.diff(xs) > 1) + 1
                parts = [(int(q[0]), int(q[-1]) + 1) for q in np.split(xs, brk) if len(q)]
            for a, b in parts:
                these.append(len(runs))
                runs.append((y, a, b, int(row[x0])))
        row_runs.append(these)
    return runs, row_runs


def _union_runs(runs: list, row_runs: list) -> list[int]:
    """Union-find over vertically overlapping same-colour runs; returns roots."""
    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(1, len(row_runs)):
        above = row_runs[y - 1]
        j = 0
        for i in row_runs[y]:
            _, a0, a1, c = runs[i]
            while j < len(above) and runs[above[j]][2] <= a0:
                j += 1
            k = j
            while k < len(above) and runs[above[k]][1] < a1:
                if runs[above[k]][3] == c:
                    ri, rk = find(i), find(above[k])
                    if ri != rk:
                        parent[ri] = rk
                k += 1
    return [find(i) for i in range(len(runs))]


def components(grid: np.ndarray, valid: np.ndarray | None = None) -> list[dict]:
    """4-connected same-colour components, via row runs + union-find (~2 ms on
    real 64x64 frames, vs ~11 ms for a per-cell BFS). Returns dicts with
    colour, size, bbox, `cells` (N x 2 array of (y, x)) and `rep` = the member
    cell nearest the centroid (a click point guaranteed to be on the
    component). Cells where `valid` is False are skipped."""
    runs, row_runs = _row_runs(grid, valid)
    groups: dict[int, list[int]] = {}
    for i, root in enumerate(_union_runs(runs, row_runs)):
        groups.setdefault(root, []).append(i)
    out = []
    for members in groups.values():
        ys = np.concatenate([np.full(runs[i][2] - runs[i][1], runs[i][0]) for i in members])
        xs = np.concatenate([np.arange(runs[i][1], runs[i][2]) for i in members])
        cells = np.stack([ys, xs], axis=1)
        cy, cx = cells.mean(axis=0)
        r = int(np.argmin((cells[:, 0] - cy) ** 2 + (cells[:, 1] - cx) ** 2))
        out.append(dict(color=runs[members[0]][3], size=len(cells), cells=cells,
                        rep=(int(cells[r, 0]), int(cells[r, 1])),
                        bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))))
    return out


class VolatileMask:
    """Learns which cells to ignore when deciding state identity.

    Two mechanisms (see AGENT.md):
      * HUD ticks: small diff components near the frame border are counter
        evidence; a border line with enough evidence is masked in full.
      * Frequency: cells changing in most transitions (blinkers) are masked.
    `version` increments whenever the mask grows, so callers can invalidate
    keys computed under the old mask.
    """

    def __init__(self, shape=(64, 64)) -> None:
        self.shape = shape
        self.mask = np.zeros(shape, dtype=bool)  # True = ignored
        self.tick_cells = np.zeros(shape, dtype=bool)
        self.change_counts = np.zeros(shape, dtype=np.int32)
        self.transitions = 0
        self.version = 0

    def _ensure_shape(self, shape) -> None:
        if shape != self.shape:
            self.__init__(shape)
            self.version += 1

    def observe(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Update from one transition. Returns True if the mask changed."""
        self._ensure_shape(after.shape)
        if before.shape != after.shape:
            return False
        diff = before != after
        self.transitions += 1
        self.change_counts += diff
        if not diff.any():
            return False
        old = self.mask.copy()
        h, w = diff.shape
        band = np.zeros_like(diff)
        band[:BORDER_BAND, :] = band[-BORDER_BAND:, :] = True
        band[:, :BORDER_BAND] = band[:, -BORDER_BAND:] = True
        for comp in _diff_components(diff):
            if len(comp) <= TICK_MAX_CELLS and all(band[y, x] for y, x in comp):
                for y, x in comp:
                    self.tick_cells[y, x] = True
        self.mask |= self.tick_cells
        for r in list(range(BORDER_BAND)) + list(range(h - BORDER_BAND, h)):
            if self.tick_cells[r].sum() >= TICK_LINE_MIN_CELLS:
                self.mask[r, :] = True
        for c in list(range(BORDER_BAND)) + list(range(w - BORDER_BAND, w)):
            if self.tick_cells[:, c].sum() >= TICK_LINE_MIN_CELLS:
                self.mask[:, c] = True
        if self.transitions >= FREQ_MASK_MIN_TRANSITIONS:
            self.mask |= self.change_counts >= FREQ_MASK_RATE * self.transitions
        if not np.array_equal(old, self.mask):
            self.version += 1
            return True
        return False

    def masked(self, grid: np.ndarray) -> np.ndarray:
        self._ensure_shape(grid.shape)
        g = grid.copy()
        g[self.mask] = 255
        return g

    def key(self, grid: np.ndarray) -> int:
        return hash(self.masked(grid).tobytes())


def _diff_components(diff: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected components of a boolean diff mask (only visits True cells)."""
    ys, xs = np.nonzero(diff)
    todo = set(zip(ys.tolist(), xs.tolist(), strict=True))
    comps = []
    while todo:
        start = todo.pop()
        comp, q = [start], deque([start])
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (y + dy, x + dx)
                    if n in todo:
                        todo.remove(n)
                        comp.append(n)
                        q.append(n)
        comps.append(comp)
    return comps


class AvatarTracker:
    """Identifies the controllable avatar colour: the colour whose centroid
    displacement depends on the chosen action (different actions -> different
    consistent shifts). Colours that shift the same way regardless of action
    are environmental (gravity, conveyors) and score low."""

    def __init__(self) -> None:
        # colour -> action -> list of (dy, dx) centroid shifts (rounded)
        self.shifts: dict[int, dict[int, list[tuple[int, int]]]] = {}
        self.color: int | None = None
        self.n_obs = 0

    def observe(self, before: np.ndarray, after: np.ndarray, action_id: int) -> None:
        if action_id in (0, 6):
            return
        changed = before != after
        if not changed.any():
            return
        self.n_obs += 1
        for c in np.unique(np.concatenate([before[changed], after[changed]])).tolist():
            # Centroid of the colour's CHANGED cells only (vacated vs newly
            # occupied): immune to static same-colour pixels elsewhere. For a
            # rigid mover the value is constant per action (not the true
            # displacement, but consistency is all we need).
            yb, xb = np.nonzero(changed & (before == c))
            ya, xa = np.nonzero(changed & (after == c))
            if len(yb) == 0 or len(ya) == 0 or abs(len(yb) - len(ya)) > max(2, len(yb) // 5):
                continue
            shift = (round(ya.mean() - yb.mean()), round(xa.mean() - xb.mean()))
            self.shifts.setdefault(c, {}).setdefault(action_id, []).append(shift)
        if self.n_obs >= AVATAR_MIN_OBS:
            self._refit()

    def _refit(self) -> None:
        best, best_score = None, 0.0
        for c, per_action in self.shifts.items():
            modes, cons, n = set(), [], 0
            for shifts in per_action.values():
                n += len(shifts)
                mode = max(set(shifts), key=shifts.count)
                if mode != (0, 0):
                    modes.add(mode)
                cons.append(shifts.count(mode) / len(shifts))
            if n < AVATAR_MIN_OBS or len(modes) < 2:
                continue
            score = sum(cons) / len(cons) * min(1.0, len(modes) / 2)
            if score > best_score:
                best, best_score = c, score
        self.color = best if best_score >= AVATAR_MIN_CONSISTENCY else None

    def position(self, grid: np.ndarray) -> tuple[int, int] | None:
        if self.color is None:
            return None
        ys, xs = np.nonzero(grid == self.color)
        if len(ys) == 0:
            return None
        return int(round(ys.mean() * 4)), int(round(xs.mean() * 4))  # quarter-cell resolution
