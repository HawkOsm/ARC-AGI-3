"""The policy: systematic graph exploration with a learned action model.

Framework-independent on purpose (takes plain frame fields, returns an action
tuple) so it can be unit-tested and profiled without the ARC framework.

Loop per step:
  1. Perceive: settled grid -> volatile mask update -> masked state key.
  2. Record the transition (prev_key, prev_act) -> key | DEATH | LEVELUP.
  3. Verify the plan: if the key differs from what the plan expected, drop it.
  4. Decide: if the level is already solved, shortest path to its LEVELUP edge;
     else the cheapest frontier (distance + lambda * (1 - p)) over untried
     actions; else a random legal action.
"""
from __future__ import annotations

import random
import zlib

import numpy as np

from .config import (
    FRONTIER_LAMBDA,
    LARGE_COMPONENT,
    LATTICE_MIN_SIZE,
    LATTICE_PER_COMPONENT,
    LATTICE_PRIOR_FACTOR,
    LATTICE_STRIDE,
    LOCAL_GREEDY_P,
    MAX_CLICK_TARGETS,
    PRIOR_CLICK_LARGE,
    PRIOR_CLICK_SINGLE_PIXEL,
    PRIOR_CLICK_SMALL,
    PRIOR_SIMPLE,
    PRIOR_UNDO_CANDIDATE,
    REFINE_STRIDES,
    REPLAY_ENABLED,
)
from .memory import DEATH, LEVELUP, ActionModel, LevelGraph, Node, pack_grid
from .perception import AvatarTracker, VolatileMask, components, settled_grid

RESET: tuple = (0, -1, -1)


class Explorer:
    def __init__(self, seed: int = 0, replay: bool = REPLAY_ENABLED) -> None:
        self.rng = random.Random(seed)
        self.replay = replay
        self.mask = VolatileMask()
        self.avatar = AvatarTracker()
        self.model = ActionModel()
        self.graphs: dict[int, LevelGraph] = {}
        self.level: int | None = None
        self.prev_grid: np.ndarray | None = None  # raw settled grid before prev_act
        self.prev_key: int | None = None
        self.prev_prev_key: int | None = None
        self.prev_act: tuple | None = None
        self.arrived_by: tuple | None = None
        self.plan: list = []  # [(act, expected_key), ...]
        self.expected = None  # key the last planned action should produce
        self.at_level_start = False
        # per-play accounting (a full reset starts a new scored play)
        self.play_actions: dict[int, int] = {}
        self.best_solution_len: dict[int, int] = {}
        self.want_full_reset = False
        self.finished = False
        self.stats = dict(actions=0, deaths=0, plan_mismatch=0, conflicts=0, mask_changes=0,
                          fallback=0, levelups=0, full_resets=0, replays=0, nodes=0)

    # ─────────────────────────── public API ───────────────────────────

    def act(self, state: str, frame_layers, levels_completed: int, available: list[int],
            full_reset: bool = False) -> tuple:
        act = self._decide(state, frame_layers, levels_completed, available, full_reset)
        self.stats["actions"] += 1
        # Every action (RESET included) counts toward this play's current level,
        # except the RESET that starts a new play.
        if self.level is not None and not (act == RESET and self.want_full_reset):
            self.play_actions[self.level] = self.play_actions.get(self.level, 0) + 1
        return act

    def forget_last(self) -> None:
        """The last action never reached the engine: record nothing for it."""
        self.prev_act = None
        self.plan, self.expected = [], None

    def wants_more(self, state: str) -> bool:
        """Called on WIN: keep playing only to run a replay play."""
        if state != "WIN":
            return True
        return self.replay and not self.finished and self._replay_gain() > 0

    # ─────────────────────────── decision ───────────────────────────

    def _decide(self, state, frame_layers, lvl, available, full_reset) -> tuple:
        grid = settled_grid(frame_layers)
        if state == "NOT_PLAYED" or (grid is None and state != "GAME_OVER"):
            return self._reset()
        if state == "GAME_OVER":
            self._on_death()
            return self._reset()
        if full_reset:
            self._on_new_play()
        if self.level is None:
            self.level = lvl
            self.at_level_start = True
        elif lvl > self.level or state == "WIN":
            self._on_levelup(grid, lvl if lvl > self.level else self.level + 1)
            if self.replay and self._replay_gain() > 0:
                self.want_full_reset = True
                self.stats["full_resets"] += 1
                return self._reset()  # action count is 0 here -> engine does a FULL reset
            if state == "WIN":
                self.finished = True
                return self._reset()
        elif lvl < self.level:
            self._on_new_play()
            self.level = lvl
            self.at_level_start = True

        g = self.graphs.setdefault(self.level, LevelGraph())
        self._observe_transition(g, grid)
        key = self.mask.key(grid)
        node = self._ensure_node(g, key, grid, available)
        if self.at_level_start:
            if g.start_key is None:
                g.start_key = key
            self.at_level_start = False
        if self.prev_act is not None and self.prev_act != RESET:
            self._record(g, key)
        if self.plan and self.expected is not None and self.expected != key:
            self.stats["plan_mismatch"] += 1
            self.plan = []
        act = self._choose(g, node, available)
        self.prev_prev_key = self.prev_key if self.prev_act != RESET else None
        self.prev_grid, self.prev_key, self.prev_act = grid, key, act
        return act

    def _choose(self, g: LevelGraph, node: Node, available: list[int]) -> tuple:
        if g.solved() and not self.plan:
            path = g.path_to_levelup(node.key)
            if path:
                self.plan = path
        if not self.plan:
            self.plan = self._frontier_plan(g, node, available)
        if self.plan:
            act, self.expected = self.plan.pop(0)
            if act[0] in available:
                return act
            self.plan = []
        self.expected = None
        self.stats["fallback"] += 1
        return self._random_action(node, available)

    def _frontier_plan(self, g: LevelGraph, node: Node, available: list[int]) -> list:
        local = self._best_untried(node, self.arrived_by, available)
        if local is not None and local[0] >= LOCAL_GREEDY_P:
            return [(local[1], None)]
        best_cost = float("inf") if local is None else FRONTIER_LAMBDA * (1 - local[0])
        best = None if local is None else (node.key, local[1])
        dist, parent = g.bfs(node.key, stop_at=best_cost)
        for k, d in dist.items():
            if d == 0 or d >= best_cost:
                continue
            via = parent[k][1] if parent[k] else None
            cand = self._best_untried(g.nodes[k], via, available)
            if cand is None:
                continue
            cost = d + FRONTIER_LAMBDA * (1 - cand[0]) + self.rng.random() * 1e-3
            if cost < best_cost:
                best_cost, best = cost, (k, cand[1])
        if best is None:
            if 6 in available and self._refine_clicks(g, node):
                return self._frontier_plan(g, node, available)
            return []
        k, act = best
        path = [] if k == node.key else LevelGraph.unroll(parent, k)
        return path + [(act, None)]

    def _best_untried(self, node: Node, arrived_by, available):
        best = None
        for a in node.untried():
            if a[0] not in available:
                continue
            p = self.model.score(node, a, arrived_by) + self.rng.random() * 1e-6
            if best is None or p > best[0]:
                best = (p, a)
        return best

    def _random_action(self, node: Node, available: list[int]) -> tuple:
        acts = [a for a in node.actions if a[0] in available] or [(a, -1, -1) for a in available if a != 6]
        if not acts:
            return (6, self.rng.randrange(64), self.rng.randrange(64)) if 6 in available else RESET
        return self.rng.choice(acts)

    # ─────────────────────────── bookkeeping ───────────────────────────

    def _reset(self) -> tuple:
        self.plan, self.expected = [], None
        self.prev_act, self.prev_grid, self.arrived_by = RESET, None, None
        self.at_level_start = True
        return RESET

    def _observe_transition(self, g: LevelGraph, grid: np.ndarray) -> None:
        if self.prev_grid is None or self.prev_act in (None, RESET):
            return
        before = self.mask.masked(self.prev_grid)
        if self.mask.observe(self.prev_grid, grid):
            self.stats["mask_changes"] += 1
            for lg in self.graphs.values():
                lg.rekey(self._key_of_raw)
            self.prev_key = self.mask.key(self.prev_grid)
            self.prev_prev_key = None
            self.plan, self.expected = [], None
            before = self.mask.masked(self.prev_grid)
        self.avatar.observe(before, self.mask.masked(grid), self.prev_act[0])

    def _key_of_raw(self, raw: bytes) -> int:
        grid = np.frombuffer(zlib.decompress(raw), dtype=np.uint8).reshape(self.mask.shape)
        return self.mask.key(grid)

    def _record(self, g: LevelGraph, key: int) -> None:
        prev = g.nodes.get(self.prev_key)
        if prev is None:
            return
        if not g.record(self.prev_key, self.prev_act, key):
            self.stats["conflicts"] += 1
            self.plan = []
        changed = key != self.prev_key
        reverted = None
        if self.prev_prev_key is not None and self.prev_prev_key != self.prev_key:
            reverted = key == self.prev_prev_key
        self.model.observe(self.prev_act, prev.meta.get(self.prev_act, {}), changed, False,
                           prev.avatar, reverted, self.arrived_by)
        self.arrived_by = self.prev_act if changed else self.arrived_by

    def _on_death(self) -> None:
        self.stats["deaths"] += 1
        g = self.graphs.get(self.level)
        if g is not None and self.prev_key is not None and self.prev_act not in (None, RESET):
            g.record(self.prev_key, self.prev_act, DEATH)
            prev = g.nodes.get(self.prev_key)
            if prev is not None:
                self.model.observe(self.prev_act, prev.meta.get(self.prev_act, {}), True, True,
                                   prev.avatar, None, None)

    def _on_levelup(self, grid, new_level: int) -> None:
        self.stats["levelups"] += 1
        g = self.graphs.get(self.level)
        if g is not None and self.prev_key is not None and self.prev_act not in (None, RESET):
            g.record(self.prev_key, self.prev_act, LEVELUP)
            sol = g.path_to_levelup(g.start_key) if g.start_key is not None else None
            if sol:
                n = len(sol)
                self.best_solution_len[self.level] = min(n, self.best_solution_len.get(self.level, n))
        self.level = new_level
        self.prev_act, self.prev_grid, self.arrived_by = None, None, None
        self.plan, self.expected = [], None
        self.at_level_start = True

    def _on_new_play(self) -> None:
        if self.want_full_reset:
            self.stats["replays"] += 1
        self.want_full_reset = False
        self.play_actions = {}
        self.level = 0
        self.at_level_start = True

    def _replay_gain(self) -> int:
        """Actions a fresh play would save on levels already solved."""
        return sum(max(0, self.play_actions.get(lv, 0) - n) for lv, n in self.best_solution_len.items()
                   if lv in self.play_actions)

    def _ensure_node(self, g: LevelGraph, key: int, grid: np.ndarray, available: list[int]) -> Node:
        node = g.nodes.get(key)
        if node is not None:
            return node
        actions, meta = [], {}
        for aid in sorted(available):
            if aid == 6:
                continue
            a = (aid, -1, -1)
            actions.append(a)
            meta[a] = dict(prior=PRIOR_UNDO_CANDIDATE if aid == 7 else PRIOR_SIMPLE)
        if 6 in available:
            targets = []  # (prior, size, (x, y), colour)
            for c in components(self.mask.masked(grid), valid=~self.mask.mask):
                prior = _prior_of(c)
                targets.append((prior, c["size"], (c["rep"][1], c["rep"][0]), c["color"]))
                if c["size"] > LATTICE_MIN_SIZE:
                    cells = c["cells"]
                    lat = cells[(cells[:, 0] % LATTICE_STRIDE == LATTICE_STRIDE // 2)
                                & (cells[:, 1] % LATTICE_STRIDE == LATTICE_STRIDE // 2)]
                    if len(lat) > LATTICE_PER_COMPONENT:
                        lat = lat[np.linspace(0, len(lat) - 1, LATTICE_PER_COMPONENT).astype(int)]
                    for y, x in lat.tolist():
                        targets.append((prior * LATTICE_PRIOR_FACTOR, c["size"], (x, y), c["color"]))
            targets.sort(key=lambda t: (-t[0], t[1]))
            for prior, size, (x, y), color in targets[:MAX_CLICK_TARGETS]:
                a = (6, x, y)
                if a in meta:
                    continue
                actions.append(a)
                meta[a] = dict(prior=prior, color=color, size=size)
        node = Node(key, actions, meta, self.avatar.position(self.mask.masked(grid)), pack_grid(grid))
        g.nodes[key] = node
        self.stats["nodes"] += 1
        return node


    def _refine_clicks(self, g: LevelGraph, node: Node) -> bool:
        """Add the next whole-grid click lattice to an exhausted node."""
        if node.refine >= len(REFINE_STRIDES):
            return False
        stride = REFINE_STRIDES[node.refine]
        node.refine += 1
        grid = np.frombuffer(zlib.decompress(node.raw), dtype=np.uint8).reshape(self.mask.shape)
        added = 0
        for y in range(stride // 2, grid.shape[0], stride):
            for x in range(stride // 2, grid.shape[1], stride):
                a = (6, x, y)
                if a in node.meta or self.mask.mask[y, x]:
                    continue
                node.actions.append(a)
                node.meta[a] = dict(prior=PRIOR_CLICK_LARGE, color=int(grid[y, x]), size=0)
                added += 1
        self.stats["refines"] = self.stats.get("refines", 0) + 1
        return added > 0 or self._refine_clicks(g, node)


def _prior_of(c: dict) -> float:
    if c["size"] > LARGE_COMPONENT:
        return PRIOR_CLICK_LARGE
    if c["size"] == 1:
        return PRIOR_CLICK_SINGLE_PIXEL
    return PRIOR_CLICK_SMALL
