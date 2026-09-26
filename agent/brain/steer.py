"""STEER — move a controlled object onto its goal (hypothesis mode, step 2).

    Controls.of(scene, object_id)          what each key did to this object so far: {action_id: (dy, dx)}
    Navigator.plan(start, target, ...)     shortest key sequence from one position to (as near as possible) another,
                                           searched over the positions the learned steps can reach
    Prober                                 the level's first actions: each key once, then one click per object kind
                                           (cheap, informative: what moves, what reacts to clicks)
    Steerer                                follows a plan toward the picked goal, checks every move against the
                                           prediction, replans when blocked, and gives the goal up when arriving
                                           does not finish the level
Positions are an object's top-left cell (y, x); goals align the moving object's centre with the target's centre
(for identical shapes that is exact coverage).
"""
from __future__ import annotations

import re
from collections import deque

from .goals import GoalCandidate
from .scene import LevelScene, SceneObject
from .world import Action

Position = tuple[int, int]
MAX_SEARCH = 5000  # positions explored per plan
PROBE_CLICKS = 6  # clicks on distinct object kinds at a level start
UNLOCK_TRIES = 4  # switches tried per level when a needed direction is missing


class Controls:
    """What the keys do to one object, learned from the level's observations (a utility class)."""

    @staticmethod
    def of(scene: LevelScene, object_id: int) -> dict[int, tuple[int, int]]:
        """The LATEST effect of each key on the object: what a key controls can change (ar25 level 2: ACTION5 or a
        click switches the keys from moving the mirror line to moving the shapes)."""
        latest: dict[int, tuple[int, int]] = {}
        for obs in scene.observations:
            if obs.action[0] in (0, 6):
                continue
            mine = [c for c in obs.changes if scene.root(c.object_id) == object_id]
            if any(c.kind != "moved" for c in mine):
                continue  # reshaped / hidden / recoloured this time: no evidence about what the key does
            latest[obs.action[0]] = (mine[0].dy, mine[0].dx) if mine else (0, 0)
        return {a: step for a, step in latest.items() if step != (0, 0)}


class Navigator:
    """Breadth-first search over positions reachable with the learned steps (a utility class)."""

    @staticmethod
    def plan(start: Position, target: Position, steps: dict[int, tuple[int, int]], size: tuple[int, int],
             screen: tuple[int, int], blocked: set[tuple[Position, int]]) -> list[int]:
        """Actions from `start` to `target`, or to the reachable position nearest it; [] if already there."""
        h, w = size
        came: dict[Position, tuple[Position, int] | None] = {start: None}
        queue = deque([start])
        best = start
        while queue and len(came) < MAX_SEARCH:
            here = queue.popleft()
            if _distance(here, target) < _distance(best, target):
                best = here
            if here == target:
                break
            for action, (dy, dx) in steps.items():
                there = (here[0] + dy, here[1] + dx)
                if (here, action) in blocked or there in came:
                    continue
                if not (0 <= there[0] <= screen[0] - h and 0 <= there[1] <= screen[1] - w):
                    continue
                came[there] = (here, action)
                queue.append(there)
        route: list[int] = []
        node = best
        while came[node] is not None:
            previous, action = came[node]  # type: ignore[misc]
            route.append(action)
            node = previous
        route.reverse()
        return route


def _distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Prober:
    """The first actions of a level in hypothesis mode: every key once, then one click per object kind."""

    @staticmethod
    def actions(scene: LevelScene, available: list[int]) -> list[tuple[Action, str]]:
        probes = [(Action(a), f"probe: press ACTION{a} once to see what it moves") for a in sorted(available)
                  if a in (1, 2, 3, 4, 5)]
        if 6 in available:
            kinds: dict[tuple, SceneObject] = {}
            for o in scene.objects:
                if not o.background and scene.root(o.id) == o.id:
                    kinds.setdefault((o.colour, o.shape), o)
            for o in sorted(kinds.values(), key=lambda o: -o.size)[:PROBE_CLICKS]:
                y0, x0, y1, x1 = o.bbox
                probes.append((Action(6, (x0 + x1) // 2, (y0 + y1) // 2),
                               f"probe: click #{o.id} once to see what it does"))
        return probes


class Steerer:
    """Moves the controlled object of the picked goal onto its partner, one checked step at a time."""

    def __init__(self) -> None:
        self.goal: GoalCandidate | None = None
        self.mover = -1
        self.target = -1
        self.plan: list[int] = []
        self.expected: Position | None = None
        self.last: tuple[Position, int] | None = None
        self.blocked: set[tuple[Position, int]] = set()
        self.arrived = False
        self.failed: set[tuple[str, int, int]] = set()  # goals reached without finishing the level
        self.note = ""
        self.stuck = False  # no known key moves the mover along a needed direction: look for a switch
        self.unlock_tries = 0
        self.switches_tried: set[Action] = set()
        # what the mover and target looked like when steering began: while they overlap, both outlines change
        self.mover_colour = -1
        self.mover_size = (0, 0)
        self.target_box: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self.goal, self.plan, self.expected, self.last, self.arrived = None, [], None, None, False
        self.blocked = set()
        self.stuck = False

    def start(self, goal: GoalCandidate, scene: LevelScene) -> bool:
        """Pick which of the two objects we can move; True if steering can begin."""
        if goal.relation == "clear" or goal.key in self.failed:
            return False
        for mover, target in ((goal.a, goal.b), (goal.b, goal.a)):
            if len(Controls.of(scene, mover)) >= 2:
                objects = {o.id: o for o in scene.objects}
                if mover not in objects or target not in objects:
                    continue
                self.reset()
                self.goal, self.mover, self.target = goal, mover, target
                m, t = objects[mover], objects[target]
                self.mover_colour, self.mover_size = m.colour, (m.bbox[2] - m.bbox[0], m.bbox[3] - m.bbox[1])
                self.target_box = t.bbox
                return True
        return False

    def next_action(self, scene: LevelScene, screen: tuple[int, int], available: list[int]) -> Action | None:
        """The next step toward the goal, or None (arrived, lost the objects, or no way to get closer)."""
        if self.goal is None:
            return None
        mover = self._find_mover(scene)
        if mover is None or self.target_box is None:
            self.note = "lost sight of the objects; steering stops"
            self.reset()
            return None
        here = mover.top_left
        if (mover.bbox[2] - mover.bbox[0], mover.bbox[3] - mover.bbox[1]) != self.mover_size \
                and self.expected is not None and _near(mover.top_left, self.expected, 1 + max(self.mover_size)):
            here = self.expected  # partly hidden (on or under the target): trust where the move should have put it
        if self.last is not None and here == self.last[0]:
            self.blocked.add(self.last)  # it did not move at all: a wall, or the key does something else now
            self.plan = []
        elif self.expected is not None and here != self.expected:
            self.plan = []  # moved, but not as predicted (or touching the target changed its outline): plan again
        goal_pos = _aligned_box(self.mover_size, self.target_box)
        if here == goal_pos:
            self.arrived = True
            self.note = f"#{self.mover} is on #{self.target}"
            return None
        steps = {a: s for a, s in Controls.of(scene, self.mover).items() if a in available}
        if any(a not in steps for a in self.plan):  # what a planned key does has changed since: plan again
            self.plan = []
        if not self.plan:
            size = (self.mover_size[0] + 1, self.mover_size[1] + 1)
            self.plan = Navigator.plan(here, goal_pos, steps, size, screen, self.blocked)
            if not self.plan:
                gap_y, gap_x = goal_pos[0] - here[0], goal_pos[1] - here[1]
                missing = (gap_y and not any(dy for dy, _ in steps.values())) \
                    or (gap_x and not any(dx for _, dx in steps.values()))
                self.note = (f"stuck: no known key moves #{self.mover} {'vertically' if gap_y else 'sideways'}"
                             if missing else f"no way around what blocks #{self.mover}")
                self.stuck = bool(missing)  # a whole direction is missing: something may switch the controls
                if not missing:
                    self.reset()
                return None
        action = self.plan.pop(0)
        dy, dx = steps[action]
        self.expected, self.last = (here[0] + dy, here[1] + dx), (here, action)
        self.note = (f"steer: move #{self.mover} toward #{self.target} "
                     f"({len(self.plan) + 1} step(s) planned, {_distance(here, goal_pos)} cells to go)")
        return Action(action)

    def unlock_probes(self, scene: LevelScene, available: list[int]) -> list[tuple[Action, str]]:
        """Stuck: try ONE thing that might change what the keys control - ACTION5, or clicking an object involved
        (the mover, the target, the mover's mirror partner) - then every key once to relearn them. Steering resumes
        after; if the direction is still missing, the next switch is tried."""
        self.unlock_tries += 1
        self.stuck = False
        keys = [a for a in sorted(available) if a in (1, 2, 3, 4)]
        switches: list[tuple[Action, str]] = []
        if 5 in available:
            switches.append((Action(5), "ACTION5"))
        if 6 in available:
            partners = {self.mover, self.target}
            for fact in scene.relations:  # "#7 is where #5 would be if mirrored across #2": all three may switch
                ids = [int(i) for i in re.findall(r"#(\d+)", fact)]
                if "mirrored" in fact and self.mover in ids:
                    partners.update(ids)
            for o in scene.objects:
                if o.id in partners and scene.root(o.id) == o.id:
                    y0, x0, y1, x1 = o.bbox
                    switches.append((Action(6, (x0 + x1) // 2, (y0 + y1) // 2), f"clicking #{o.id}"))
        for switch, name in switches:
            if switch in self.switches_tried:
                continue
            self.switches_tried.add(switch)
            return [(switch, f"unlock: try {name} - maybe it changes what the keys move")] + \
                [(Action(k), f"unlock: re-test ACTION{k} after {name}") for k in keys]
        return []

    def _find_mover(self, scene: LevelScene) -> SceneObject | None:
        """The mover by id; if its id is gone (its outline changed), the object of its colour nearest where it
        should be."""
        objects = {o.id: o for o in scene.objects}
        if self.mover in objects:
            return objects[self.mover]
        same_colour = [o for o in scene.objects if o.colour == self.mover_colour and not o.background]
        if not same_colour:
            return None
        anchor = self.expected or (self.last[0] if self.last else None)
        if anchor is None:
            return max(same_colour, key=lambda o: o.size)
        found = min(same_colour, key=lambda o: _distance(o.top_left, anchor) - o.size / 100)
        self.mover = found.id
        return found

    def give_up(self) -> None:
        """Arrived and the level did not end: this goal is wrong."""
        if self.goal is not None:
            self.failed.add(self.goal.key)
        self.reset()


def _aligned_box(mover_size: tuple[int, int], target: tuple[int, int, int, int]) -> Position:
    """Top-left position that puts the mover's centre on the target box's centre."""
    ty, tx = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    return round(ty - mover_size[0] / 2), round(tx - mover_size[1] / 2)


def _near(a: Position, b: Position, margin: int) -> bool:
    return abs(a[0] - b[0]) <= margin and abs(a[1] - b[1]) <= margin
