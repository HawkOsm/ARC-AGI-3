"""LEARN — the learners: the only code that turns what the agent saw into beliefs in `world`.

    CounterLearner(world).learn(before, after)        find HUD step-counter pixels to ignore
    AvatarLearner(world).learn(before, after, key)    which colour moves because of our keys?
    TouchLearner(world).learn(before, after, key)     floor / wall / hazard / goal / pushable / collectible
    ActionLearner(world).record(...)                  what an action just did (world.actions)
    ActionLearner(world).note_available(available)    the game mode the offered actions imply (world.mode)
    DeathJudge(world).is_timeout(level, life_length)  was that death just the step counter running out?

Read-only questions live on the beliefs themselves: world.counters.hide(grid),
world.avatar.position(grid) / .body(picture) / .usual_step(key), world.colour_roles.role_of(colour).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .look import ScreenReader
from .settings import (
    AVATAR_MAX_CELLS,
    AVATAR_MIN_CONSISTENCY,
    AVATAR_MIN_OBSERVATIONS,
    AVATAR_PART_MIN_MOVES,
    AVATAR_PART_RATE,
    AVATAR_REFIT_EVERY,
    BLINKER_CHANGE_RATE,
    BLINKER_MIN_TRANSITIONS,
    COUNTER_BORDER_BAND,
    COUNTER_TICK_MAX_CELLS,
    COUNTER_TRACK_MAX_GAP,
    COUNTER_TRACK_MIN_POSITIONS,
    TIMEOUT_MIN_LIFE,
    TIMEOUT_TOLERANCE,
    TIMEOUT_TOLERANCE_FRACTION,
)
from .world import DIED, Action, ClickInfo, GameMode, Outcome, Role, Tally, World

# ── Counters / HUD ──────────────────────────────────────────────────────────


class CounterLearner:
    def __init__(self, world: World) -> None:
        self.world = world

    def learn(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Learn counter pixels from one transition. Returns True if the ignore-mask grew
        (screen ids computed before are then stale and must be recomputed)."""
        c = self.world.counters
        c.fit(after.shape)
        if before.shape != after.shape:
            return False
        changed = before != after
        c.transitions += 1
        c.change_counts += changed
        if not changed.any():
            return False
        ignore_before = c.ignore.copy()
        near_edge = np.zeros_like(changed)
        b = COUNTER_BORDER_BAND
        near_edge[:b, :] = near_edge[-b:, :] = True
        near_edge[:, :b] = near_edge[:, -b:] = True
        ticked = False
        for blob in ScreenReader.changed_blobs(changed):
            if len(blob) <= COUNTER_TICK_MAX_CELLS and all(near_edge[y, x] for y, x in blob):
                ticked = True
                for y, x in blob:
                    c.tick_cells[y, x] = True
                cy = round(sum(y for y, _ in blob) / len(blob))
                cx = round(sum(x for _, x in blob) / len(blob))
                for y in {y for y, _ in blob}:
                    c.ticks_along_row[y].add(cx)
                for x in {x for _, x in blob}:
                    c.ticks_along_col[x].add(cy)
        c.ignore |= c.tick_cells
        if ticked:  # a line whose ticks advance along it is a counter track: ignore all of it
            for r, positions in enumerate(c.ticks_along_row):
                if self._is_counter_track(positions):
                    c.ignore[r, :] = True
            for col, positions in enumerate(c.ticks_along_col):
                if self._is_counter_track(positions):
                    c.ignore[:, col] = True
        if c.transitions >= BLINKER_MIN_TRANSITIONS:
            c.ignore |= c.change_counts >= BLINKER_CHANGE_RATE * c.transitions
        if not np.array_equal(ignore_before, c.ignore):
            c.version += 1
            return True
        return False

    @staticmethod
    def _is_counter_track(positions: set) -> bool:
        """Enough distinct tick positions forming one near-contiguous run (a counter
        advances along its track; two separate counters crossing a line do not)."""
        if len(positions) < COUNTER_TRACK_MIN_POSITIONS:
            return False
        p = sorted(positions)
        return all(b - a <= COUNTER_TRACK_MAX_GAP for a, b in zip(p, p[1:], strict=False))


# ── The avatar ──────────────────────────────────────────────────────────────


class AvatarLearner:
    def __init__(self, world: World) -> None:
        self.world = world

    def learn(self, before: np.ndarray, after: np.ndarray, action_id: int) -> None:
        """The avatar is the colour whose movement depends on which key we press
        (different actions -> different consistent shifts). Colours that shift the
        same way whatever we press are the environment (gravity, conveyors)."""
        if action_id in (0, 6):
            return
        changed = before != after
        if not changed.any():
            return
        av = self.world.avatar
        av.moves_seen += 1
        for colour in np.unique(np.concatenate([before[changed], after[changed]])).tolist():
            # The background mirrors every move (cells it loses/regains), which looks just as
            # "action-dependent" as the avatar: a colour covering this much can't be the avatar
            # (wa30/ka59/re86/tr87 picked their background before this rule).
            if (after == colour).sum() > AVATAR_MAX_CELLS:
                continue
            # centroid of the colour's changed cells only (left vs newly covered): ignores
            # static pixels of the same colour; constant per action for a rigid mover
            yb, xb = np.nonzero(changed & (before == colour))
            ya, xa = np.nonzero(changed & (after == colour))
            if len(yb) == 0 or len(ya) == 0 or abs(len(yb) - len(ya)) > max(2, len(yb) // 5):
                continue
            shift = (round(ya.mean() - yb.mean()), round(xa.mean() - xb.mean()))
            av.shifts.setdefault(colour, {}).setdefault(action_id, Counter())[shift] += 1
        if av.moves_seen >= AVATAR_MIN_OBSERVATIONS and av.moves_seen % AVATAR_REFIT_EVERY == 0:
            self._refit()

    def _refit(self) -> None:
        av = self.world.avatar
        best, best_score = None, 0.0
        for colour, per_action in av.shifts.items():
            distinct_moves, consistency, n = set(), [], 0
            for shifts in per_action.values():
                total = sum(shifts.values())
                n += total
                usual, count = shifts.most_common(1)[0]
                if usual != (0, 0):
                    distinct_moves.add(usual)
                consistency.append(count / total)
            if n < AVATAR_MIN_OBSERVATIONS or len(distinct_moves) < 2:
                continue
            score = sum(consistency) / len(consistency) * min(1.0, len(distinct_moves) / 2)
            if score > best_score:
                best, best_score = colour, score
        av.colour = best if best_score >= AVATAR_MIN_CONSISTENCY else None


# ── Actions ─────────────────────────────────────────────────────────────────


class ActionLearner:
    def __init__(self, world: World) -> None:
        self.world = world

    def note_available(self, available: list[int]) -> bool:
        """The offered actions set the game mode. Returns True if the mode changed."""
        mode = GameMode.from_actions(available)
        if mode == self.world.mode:
            return False
        self.world.mode = mode
        return True

    def record(self, action: Action, info: ClickInfo, changed: bool, died: bool, avatar_at,
               undid_last_move: bool | None, previous_action: Action | None) -> None:
        """Update world.actions with what `action` just did."""
        a = self.world.actions
        action_id = action[0]
        if action_id == 6:
            a.click_colour_changed_screen.setdefault(info.colour, Tally()).add(changed)
        else:
            a.changed_screen.setdefault(action_id, Tally()).add(changed)
        if undid_last_move is not None:
            a.undid_last_move.setdefault(action_id, Tally()).add(undid_last_move)
            if previous_action is not None and previous_action[0] != 6 and action_id != 6:
                a.reversed_after.setdefault((previous_action[0], action_id), Tally()).add(undid_last_move)
        if avatar_at is not None and action_id != 6:
            if died:
                a.died_at[(avatar_at, action_id)] = a.died_at.get((avatar_at, action_id), 0) + 1
            elif not changed:
                a.no_op_at[(avatar_at, action_id)] = a.no_op_at.get((avatar_at, action_id), 0) + 1


# ── Touching things ─────────────────────────────────────────────────────────


class TouchLearner:
    """What happened where the avatar tried to go? Evidence per colour in world.colour_roles:
        moved onto it                        -> FLOOR
        key usually moves us, didn't         -> the colour in the way is a WALL
        died stepping into it                -> HAZARD       (outcome=DIED, `after` unused)
        level finished stepping into it      -> GOAL         (outcome=LEVEL_UP)
        moved the same way we did (sometimes)-> PUSHABLE     (every time -> part of the avatar)
        gone when we stepped off it          -> COLLECTIBLE
    Pictures must already have counters hidden."""

    def __init__(self, world: World) -> None:
        self.world = world

    def learn(self, before: np.ndarray, after: np.ndarray | None, action_id: int,
              outcome: Outcome | None = None) -> None:
        world = self.world
        av = world.avatar
        if av.colour is None or action_id in (0, 6):
            return
        was = av.body(before)
        if was is None:
            return
        if outcome is not None:
            step = av.usual_step(action_id)
            if step is not None:
                world.colour_roles.add(self._colours_at(before, _shift(was, step) & ~was),
                                       Role.HAZARD if outcome == DIED else Role.GOAL)
            return
        now = av.body(after) if after is not None else None
        if now is None or np.array_equal(was, now):
            step = av.usual_step(action_id)
            if step is not None:
                world.colour_roles.add(self._colours_at(before, _shift(was, step) & ~was), Role.WALL)
            return
        step = self._moved_piece_step(was, now)
        if step is None:
            return
        av.steps.setdefault(action_id, Counter())[step] += 1
        av.moves += 1
        covered = now & ~was
        world.colour_roles.add(self._colours_at(before, covered), Role.FLOOR)
        self._learn_what_moved_along(before, after, was | now, step)
        self._learn_what_was_collected(before, after, was, now)

    @staticmethod
    def _moved_piece_step(was: np.ndarray, now: np.ndarray) -> tuple[int, int] | None:
        """(dy, dx) of the avatar piece that moved: compare the connected piece holding the
        vacated cells with the one holding the newly covered cells. (The whole-mask bounding
        box fails when the avatar colour also sits still elsewhere - tu93 read (0, 0) always.)"""
        def piece(mask, seeds):
            for blob in ScreenReader.changed_blobs(mask):
                if any(seeds[y, x] for y, x in blob):
                    return np.array(blob)
            return None
        left, arrived = piece(was, was & ~now), piece(now, now & ~was)
        if left is None or arrived is None:
            return None
        return int(arrived[:, 0].min() - left[:, 0].min()), int(arrived[:, 1].min() - left[:, 1].min())

    def _learn_what_moved_along(self, before, after, avatar_cells, step) -> None:
        """Colours (outside the avatar) whose changed cells moved in our direction."""
        world = self.world
        av = world.avatar
        changed = (before != after) & ~avatar_cells
        for c in np.unique(np.concatenate([before[changed], after[changed]])).tolist():
            if c in (av.colour, 255):
                continue
            yb, xb = np.nonzero(changed & (before == c))
            ya, xa = np.nonzero(changed & (after == c))
            if len(yb) == 0 or len(ya) == 0 or abs(len(yb) - len(ya)) > max(2, len(yb) // 5):
                continue
            dy, dx = ya.mean() - yb.mean(), xa.mean() - xb.mean()
            if np.sign(round(dy)) == np.sign(step[0]) and np.sign(round(dx)) == np.sign(step[1]):
                av.moved_with[c] += 1
                if c not in av.parts:
                    world.colour_roles.add({c}, Role.PUSHABLE)
        if av.moves >= AVATAR_PART_MIN_MOVES:
            for c, n in av.moved_with.items():
                if n / av.moves >= AVATAR_PART_RATE and c not in av.parts:
                    av.parts.add(c)
                    world.colour_roles.forget(c)  # its "roles" were just the avatar moving

    def _learn_what_was_collected(self, before, after, was, now) -> None:
        """Remember what each cell showed before the avatar covered it; when the avatar
        steps off and something else shows, that thing was collected."""
        av = self.world.avatar
        if av.under is None or av.under.shape != before.shape:
            av.under = np.full(before.shape, 255, dtype=np.uint8)
        under = av.under
        newly_covered = now & ~was
        under[newly_covered] = before[newly_covered]
        revealed = was & ~now & (under != 255)
        for c in np.unique(under[revealed]).tolist():
            cells = revealed & (under == c)
            if (after[cells] != c).all() and c not in av.parts and c != av.colour:
                self.world.colour_roles.add({c}, Role.COLLECTIBLE)
        under[was & ~now] = 255

    def _colours_at(self, picture: np.ndarray, where: np.ndarray) -> set:
        """Colours found at `where`, excluding the avatar's own colours and ignored cells."""
        own = self.world.avatar.own_colours()
        return {c for c in np.unique(picture[where]).tolist() if c not in own}


def _shift(mask: np.ndarray, step: tuple[int, int]) -> np.ndarray:
    """Mask moved by (dy, dx); cells pushed off-screen are dropped."""
    dy, dx = step
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    ys, xs = ys + dy, xs + dx
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    out[ys[keep], xs[keep]] = True
    return out


# ── Deaths ──────────────────────────────────────────────────────────────────


class DeathJudge:
    def __init__(self, world: World) -> None:
        self.world = world

    def is_timeout(self, level: int, life_length: int) -> bool:
        """Was this death just the step counter running out (not the last action's fault)?
        Two deaths at the same life length reveal the step budget; the earlier death is
        then un-blamed too."""
        world = self.world
        tolerance = max(TIMEOUT_TOLERANCE, round(TIMEOUT_TOLERANCE_FRACTION * life_length))
        budget = world.rules.steps_per_life.get(level)
        if budget is not None:
            return life_length >= budget - tolerance
        if life_length < TIMEOUT_MIN_LIFE:
            return False
        same = [d for d in world.deaths if d.level == level and abs(d.life_length - life_length) <= tolerance]
        if not same:
            return False
        world.rules.steps_per_life[level] = min([life_length] + [d.life_length for d in same])
        for d in same:
            if d.screen.results.get(d.action) == DIED:
                del d.screen.results[d.action]
            key = (d.avatar_at, d.action[0])
            if world.actions.died_at.get(key):
                world.actions.died_at[key] -= 1
            world.deaths.remove(d)
        return True
