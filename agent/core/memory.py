"""Per-level state graph + per-game learned action model."""
from __future__ import annotations

import zlib
from collections import deque

from .config import (
    P_DEATH,
    P_DEMOTED,
    PAIR_UNDO_MIN_TRIES,
    PAIR_UNDO_RATE,
    PRIOR_WEIGHT,
    UNDO_MIN_TRIES,
    UNDO_RATE,
)

DEATH = "DEATH"
LEVELUP = "LEVELUP"

# An action is (action_id, x, y); simple actions use x = y = -1.
Act = tuple


class Node:
    __slots__ = ("key", "actions", "meta", "edges", "avatar", "raw", "refine")

    def __init__(self, key: int, actions: list, meta: dict, avatar, raw: bytes) -> None:
        self.key = key
        self.actions = actions  # candidate actions, in prior order
        self.meta = meta  # act -> dict(prior=..., color=..., size=...)
        self.edges: dict = {}  # act -> next key | DEATH | LEVELUP
        self.avatar = avatar  # avatar position in this state (or None)
        self.raw = raw  # zlib-compressed raw grid, for re-keying on mask change
        self.refine = 0  # whole-grid click refinements applied so far

    def untried(self) -> list:
        return [a for a in self.actions if a not in self.edges]


def pack_grid(grid) -> bytes:
    return zlib.compress(grid.tobytes(), 1)


class LevelGraph:
    """Deterministic transition graph for one level, keyed by masked state."""

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.start_key: int | None = None
        self.conflicts = 0  # observed outcome != recorded outcome
        self.has_levelup = False

    def record(self, key: int, act: Act, outcome) -> bool:
        """Store an edge. Returns False if it contradicts a recorded edge."""
        node = self.nodes.get(key)
        if node is None:
            return True
        prev = node.edges.get(act)
        node.edges[act] = outcome
        if outcome == LEVELUP:
            self.has_levelup = True
        if prev is not None and prev != outcome:
            self.conflicts += 1
            return False
        return True

    def rekey(self, key_of_raw) -> None:
        """Recompute every key after the volatile mask changed. Nodes that now
        share a key are merged (their edges unioned)."""
        remap = {k: key_of_raw(n.raw) for k, n in self.nodes.items()}
        merged: dict[int, Node] = {}
        for old, node in self.nodes.items():
            new = remap[old]
            node.key = new
            node.edges = {a: (remap.get(t, t) if isinstance(t, int) else t) for a, t in node.edges.items()}
            tgt = merged.get(new)
            if tgt is None:
                merged[new] = node
                continue
            for a in node.actions:
                if a not in tgt.meta:
                    tgt.actions.append(a)
                    tgt.meta[a] = node.meta[a]
            for a, t in node.edges.items():
                if a in tgt.edges and tgt.edges[a] != t:
                    self.conflicts += 1
                tgt.edges.setdefault(a, t)
        self.nodes = merged
        if self.start_key is not None:
            self.start_key = remap.get(self.start_key, self.start_key)

    def bfs(self, src: int, stop_at: float | None = None):
        """Distances and parent pointers over known non-terminal edges."""
        dist, parent = {src: 0}, {src: None}
        q = deque([src])
        while q:
            k = q.popleft()
            if stop_at is not None and dist[k] > stop_at:
                break
            node = self.nodes.get(k)
            if node is None:
                continue
            for act, nxt in node.edges.items():
                if isinstance(nxt, int) and nxt not in dist and nxt in self.nodes:
                    dist[nxt] = dist[k] + 1
                    parent[nxt] = (k, act)
                    q.append(nxt)
        return dist, parent

    @staticmethod
    def unroll(parent: dict, target: int) -> list:
        """[(act, expected_next_key), ...] from BFS source to target."""
        path = []
        k = target
        while parent[k] is not None:
            pk, act = parent[k]
            path.append((act, k))
            k = pk
        path.reverse()
        return path

    def path_to_levelup(self, src: int) -> list | None:
        """Shortest known src -> LEVELUP path, as [(act, expected), ...]."""
        if src not in self.nodes:
            return None
        dist, parent = self.bfs(src)
        best = None
        for k, d in dist.items():
            for act, nxt in self.nodes[k].edges.items():
                if nxt == LEVELUP and (best is None or d + 1 < best[0]):
                    best = (d + 1, k, act)
        if best is None:
            return None
        return self.unroll(parent, best[1]) + [(best[2], LEVELUP)]

    def solved(self) -> bool:
        return self.has_levelup


class Rate:
    __slots__ = ("tries", "hits")

    def __init__(self) -> None:
        self.tries = 0
        self.hits = 0

    def add(self, hit: bool) -> None:
        self.tries += 1
        self.hits += int(hit)

    def p(self, prior: float) -> float:
        return (self.hits + PRIOR_WEIGHT * prior) / (self.tries + PRIOR_WEIGHT)


class ActionModel:
    """Generalises across states: which actions/click colours tend to change
    the state, which are undo-like, and which are no-ops or deadly from a
    given avatar position (walls, pits)."""

    def __init__(self) -> None:
        self.simple: dict[int, Rate] = {}
        self.click_color: dict[int, Rate] = {}
        self.undo: dict[int, Rate] = {}
        self.pair_undo: dict[tuple, Rate] = {}
        self.avatar_noop: dict[tuple, int] = {}
        self.avatar_death: dict[tuple, int] = {}

    def observe(self, act: Act, meta: dict, changed: bool, died: bool, avatar,
                reverted: bool | None, prev_act: Act | None) -> None:
        aid = act[0]
        if aid == 6:
            self.click_color.setdefault(meta.get("color", -1), Rate()).add(changed)
        else:
            self.simple.setdefault(aid, Rate()).add(changed)
        if reverted is not None:
            self.undo.setdefault(aid, Rate()).add(reverted)
            if prev_act is not None and prev_act[0] != 6 and aid != 6:
                self.pair_undo.setdefault((prev_act[0], aid), Rate()).add(reverted)
        if avatar is not None and aid != 6:
            if died:
                self.avatar_death[(avatar, aid)] = self.avatar_death.get((avatar, aid), 0) + 1
            elif not changed:
                self.avatar_noop[(avatar, aid)] = self.avatar_noop.get((avatar, aid), 0) + 1

    def is_undo(self, aid: int) -> bool:
        r = self.undo.get(aid)
        return r is not None and r.tries >= UNDO_MIN_TRIES and r.hits / r.tries >= UNDO_RATE

    def reverts_after(self, prev_aid: int, aid: int) -> bool:
        r = self.pair_undo.get((prev_aid, aid))
        return r is not None and r.tries >= PAIR_UNDO_MIN_TRIES and r.hits / r.tries >= PAIR_UNDO_RATE

    def score(self, node: Node, act: Act, arrived_by: Act | None) -> float:
        """Estimated probability that trying `act` at `node` reaches a new state."""
        aid = act[0]
        meta = node.meta.get(act, {})
        prior = meta.get("prior", 0.5)
        if aid != 6:
            if node.avatar is not None:
                if self.avatar_death.get((node.avatar, aid)):
                    return P_DEATH
                if self.avatar_noop.get((node.avatar, aid)):
                    return P_DEMOTED
            if self.is_undo(aid):
                return P_DEMOTED
            if arrived_by is not None and arrived_by[0] != 6 and self.reverts_after(arrived_by[0], aid):
                return P_DEMOTED
            r = self.simple.get(aid)
            return r.p(prior) if r else prior
        r = self.click_color.get(meta.get("color", -1))
        return r.p(prior) if r else prior
