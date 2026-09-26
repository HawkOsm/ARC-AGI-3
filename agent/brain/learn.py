"""LEARN — functions that turn what the agent saw into beliefs in `world`.

    update_counters(world, before, after)          find HUD step-counter pixels to ignore
    hide_counters(world, grid)                     the picture with counter pixels blanked out
    update_avatar(world, before, after, action)    which colour moves because of our actions?
    avatar_position(world, grid)                   where the avatar is (or None)
    record_action_result(world, ...)               what an action just did (world.actions)
    learn_what_touching_does(world, ...)           floor / wall / hazard / goal / pushable / collectible per colour
    usual_step(world, action_id)                   how far a key usually moves the avatar
    is_timeout_death(world, level, life_length)    was that death just the step counter running out?
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .look import changed_blobs
from .settings import (
    AVATAR_MAX_CELLS,
    AVATAR_MIN_CONSISTENCY,
    AVATAR_MIN_OBSERVATIONS,
    AVATAR_PART_MIN_MOVES,
    AVATAR_PART_RATE,
    AVATAR_PART_SEARCH_MARGIN,
    AVATAR_REFIT_EVERY,
    BLINKER_CHANGE_RATE,
    BLINKER_MIN_TRANSITIONS,
    COUNTER_BORDER_BAND,
    COUNTER_TICK_MAX_CELLS,
    COUNTER_TRACK_MAX_GAP,
    COUNTER_TRACK_MIN_POSITIONS,
    STEP_MIN_OBSERVATIONS,
    ROLE_MIN_EVIDENCE,
    ROLE_MIN_SHARE,
    TIMEOUT_MIN_LIFE,
    TIMEOUT_TOLERANCE,
    TIMEOUT_TOLERANCE_FRACTION,
)
from .world import DIED, Tally, World, new_counter_memory

# ── Counters / HUD ──────────────────────────────────────────────────────────


def update_counters(world: World, before: np.ndarray, after: np.ndarray) -> bool:
    """Learn counter pixels from one transition. Returns True if the ignore-mask grew
    (screen ids computed before are then stale and must be recomputed)."""
    c = _counters_for(world, after.shape)
    if before.shape != after.shape:
        return False
    changed = before != after
    c["transitions"] += 1
    c["change_counts"] += changed
    if not changed.any():
        return False
    ignore_before = c["ignore"].copy()
    near_edge = np.zeros_like(changed)
    b = COUNTER_BORDER_BAND
    near_edge[:b, :] = near_edge[-b:, :] = True
    near_edge[:, :b] = near_edge[:, -b:] = True
    ticked = False
    for blob in changed_blobs(changed):
        if len(blob) <= COUNTER_TICK_MAX_CELLS and all(near_edge[y, x] for y, x in blob):
            ticked = True
            for y, x in blob:
                c["tick_cells"][y, x] = True
            cy = round(sum(y for y, _ in blob) / len(blob))
            cx = round(sum(x for _, x in blob) / len(blob))
            for y in {y for y, _ in blob}:
                c["ticks_along_row"][y].add(cx)
            for x in {x for _, x in blob}:
                c["ticks_along_col"][x].add(cy)
    c["ignore"] |= c["tick_cells"]
    if ticked:  # a line whose ticks advance along it is a counter track: ignore all of it
        for r, positions in enumerate(c["ticks_along_row"]):
            if _is_counter_track(positions):
                c["ignore"][r, :] = True
        for col, positions in enumerate(c["ticks_along_col"]):
            if _is_counter_track(positions):
                c["ignore"][:, col] = True
    if c["transitions"] >= BLINKER_MIN_TRANSITIONS:
        c["ignore"] |= c["change_counts"] >= BLINKER_CHANGE_RATE * c["transitions"]
    if not np.array_equal(ignore_before, c["ignore"]):
        c["version"] += 1
        return True
    return False


def hide_counters(world: World, grid: np.ndarray) -> np.ndarray:
    """Copy of the picture with ignored (counter) pixels set to 255."""
    c = _counters_for(world, grid.shape)
    g = grid.copy()
    g[c["ignore"]] = 255
    return g


def _counters_for(world: World, shape) -> dict:
    if world.counters["shape"] != shape:  # frame size changed: start over
        version = world.counters["version"]
        world.counters = new_counter_memory(shape)
        world.counters["version"] = version + 1
    return world.counters


def _is_counter_track(positions: set) -> bool:
    """Enough distinct tick positions forming one near-contiguous run (a counter
    advances along its track; two separate counters crossing a line do not)."""
    if len(positions) < COUNTER_TRACK_MIN_POSITIONS:
        return False
    p = sorted(positions)
    return all(b - a <= COUNTER_TRACK_MAX_GAP for a, b in zip(p, p[1:], strict=False))


# ── The avatar ──────────────────────────────────────────────────────────────


def update_avatar(world: World, before: np.ndarray, after: np.ndarray, action_id: int) -> None:
    """The avatar is the colour whose movement depends on which key we press
    (different actions -> different consistent shifts). Colours that shift the
    same way whatever we press are the environment (gravity, conveyors)."""
    if action_id in (0, 6):
        return
    changed = before != after
    if not changed.any():
        return
    av = world.avatar
    av["moves_seen"] += 1
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
        av["shifts"].setdefault(colour, {}).setdefault(action_id, Counter())[shift] += 1
    if av["moves_seen"] >= AVATAR_MIN_OBSERVATIONS and av["moves_seen"] % AVATAR_REFIT_EVERY == 0:
        _refit_avatar(av)


def _refit_avatar(av: dict) -> None:
    best, best_score = None, 0.0
    for colour, per_action in av["shifts"].items():
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
    new = best if best_score >= AVATAR_MIN_CONSISTENCY else None
    av["colour"] = new


def avatar_position(world: World, grid: np.ndarray) -> tuple[int, int] | None:
    """Centroid of the avatar colour at quarter-cell resolution, or None."""
    colour = world.avatar["colour"]
    if colour is None:
        return None
    ys, xs = np.nonzero(grid == colour)
    if len(ys) == 0:
        return None
    return int(round(ys.mean() * 4)), int(round(xs.mean() * 4))


# ── Actions ─────────────────────────────────────────────────────────────────


def record_action_result(world: World, action: tuple, info: tuple, changed: bool, died: bool, avatar_at,
                         undid_last_move: bool | None, previous_action: tuple | None) -> None:
    """Update world.actions with what `action` just did."""
    a = world.actions
    action_id = action[0]
    if action_id == 6:
        a["click_colour_changed_screen"].setdefault(info[1], Tally()).add(changed)
    else:
        a["changed_screen"].setdefault(action_id, Tally()).add(changed)
    if undid_last_move is not None:
        a["undid_last_move"].setdefault(action_id, Tally()).add(undid_last_move)
        if previous_action is not None and previous_action[0] != 6 and action_id != 6:
            a["reversed_after"].setdefault((previous_action[0], action_id), Tally()).add(undid_last_move)
    if avatar_at is not None and action_id != 6:
        if died:
            a["died_at"][(avatar_at, action_id)] = a["died_at"].get((avatar_at, action_id), 0) + 1
        elif not changed:
            a["no_op_at"][(avatar_at, action_id)] = a["no_op_at"].get((avatar_at, action_id), 0) + 1


# ── Touching things ─────────────────────────────────────────────────────────


def learn_what_touching_does(world: World, before: np.ndarray, after: np.ndarray | None, action_id: int,
                             outcome: str | None = None) -> None:
    """What happened where the avatar tried to go? Evidence per colour in world.colour_roles:
        moved onto it                        -> "floor"
        key usually moves us, didn't         -> the colour in the way is a "wall"
        died stepping into it                -> "hazard"      (outcome="died", `after` unused)
        level finished stepping into it      -> "goal"        (outcome="level_up")
        moved the same way we did (sometimes)-> "pushable"    (every time -> part of the avatar)
        gone when we stepped off it          -> "collectible"
    Pictures must already have counters hidden."""
    if world.avatar["colour"] is None or action_id in (0, 6):
        return
    was = avatar_body(world, before)
    if was is None:
        return
    if outcome is not None:
        step = usual_step(world, action_id)
        if step is not None:
            _add_roles(world, _colours_at(world, before, _shift(was, step) & ~was),
                       "hazard" if outcome == "died" else "goal")
        return
    now = avatar_body(world, after)
    if now is None or np.array_equal(was, now):
        step = usual_step(world, action_id)
        if step is not None:
            _add_roles(world, _colours_at(world, before, _shift(was, step) & ~was), "wall")
        return
    av = world.avatar
    step = _moved_piece_step(was, now)
    if step is None:
        return
    av["steps"].setdefault(action_id, Counter())[step] += 1
    av["moves"] += 1
    covered = now & ~was
    _add_roles(world, _colours_at(world, before, covered), "floor")
    _learn_what_moved_along(world, before, after, was | now, step)
    _learn_what_was_collected(world, before, after, was, now)


def _moved_piece_step(was: np.ndarray, now: np.ndarray) -> tuple[int, int] | None:
    """(dy, dx) of the avatar piece that moved: compare the connected piece holding the
    vacated cells with the one holding the newly covered cells. (The whole-mask bounding
    box fails when the avatar colour also sits still elsewhere - tu93 read (0, 0) always.)"""
    def piece(mask, seeds):
        for blob in changed_blobs(mask):
            if any(seeds[y, x] for y, x in blob):
                return np.array(blob)
        return None
    left, arrived = piece(was, was & ~now), piece(now, now & ~was)
    if left is None or arrived is None:
        return None
    return int(arrived[:, 0].min() - left[:, 0].min()), int(arrived[:, 1].min() - left[:, 1].min())


def avatar_body(world: World, picture: np.ndarray) -> np.ndarray | None:
    """Cells of the avatar: its main colour plus 'part' colours touching it
    (multi-colour sprites). None if unknown, absent or implausibly large."""
    av = world.avatar
    main = picture == av["colour"]
    if not main.any():
        return None
    body = main
    if av["parts"]:  # grow from the main colour into touching part-colour cells
        allowed = main | np.isin(picture, list(av["parts"]))
        for _ in range(AVATAR_PART_SEARCH_MARGIN):
            grown = body.copy()
            grown[1:, :] |= body[:-1, :]
            grown[:-1, :] |= body[1:, :]
            grown[:, 1:] |= body[:, :-1]
            grown[:, :-1] |= body[:, 1:]
            grown &= allowed
            if np.array_equal(grown, body):
                break
            body = grown
    return None if body.sum() > AVATAR_MAX_CELLS else body


def _learn_what_moved_along(world: World, before, after, avatar_cells, step) -> None:
    """Colours (outside the avatar) whose changed cells moved in our direction."""
    av = world.avatar
    changed = (before != after) & ~avatar_cells
    for c in np.unique(np.concatenate([before[changed], after[changed]])).tolist():
        if c in (av["colour"], 255):
            continue
        yb, xb = np.nonzero(changed & (before == c))
        ya, xa = np.nonzero(changed & (after == c))
        if len(yb) == 0 or len(ya) == 0 or abs(len(yb) - len(ya)) > max(2, len(yb) // 5):
            continue
        dy, dx = ya.mean() - yb.mean(), xa.mean() - xb.mean()
        if np.sign(round(dy)) == np.sign(step[0]) and np.sign(round(dx)) == np.sign(step[1]):
            av["moved_with"][c] += 1
            if c not in av["parts"]:
                _add_roles(world, {c}, "pushable")
    if av["moves"] >= AVATAR_PART_MIN_MOVES:
        for c, n in av["moved_with"].items():
            if n / av["moves"] >= AVATAR_PART_RATE and c not in av["parts"]:
                av["parts"].add(c)
                world.colour_roles.pop(c, None)  # its "roles" were just the avatar moving


def _learn_what_was_collected(world: World, before, after, was, now) -> None:
    """Remember what each cell showed before the avatar covered it; when the avatar
    steps off and something else shows, that thing was collected."""
    av = world.avatar
    if av["under"] is None or av["under"].shape != before.shape:
        av["under"] = np.full(before.shape, 255, dtype=np.uint8)
    under = av["under"]
    newly_covered = now & ~was
    under[newly_covered] = before[newly_covered]
    revealed = was & ~now & (under != 255)
    for c in np.unique(under[revealed]).tolist():
        cells = revealed & (under == c)
        if (after[cells] != c).all() and c not in av["parts"] and c != av["colour"]:
            _add_roles(world, {c}, "collectible")
    under[was & ~now] = 255


def usual_step(world: World, action_id: int) -> tuple[int, int] | None:
    """The avatar's usual (dy, dx) for a key, once seen STEP_MIN_OBSERVATIONS times."""
    steps = world.avatar["steps"].get(action_id)
    if not steps:
        return None
    step, count = steps.most_common(1)[0]
    return step if count >= STEP_MIN_OBSERVATIONS else None


def role_of(world: World, colour: int) -> str | None:
    """The colour's dominant touch role once it has enough evidence, else None."""
    roles = world.colour_roles.get(colour)
    if not roles:
        return None
    role, count = roles.most_common(1)[0]
    return role if count >= ROLE_MIN_EVIDENCE and count >= ROLE_MIN_SHARE * sum(roles.values()) else None


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


def _colours_at(world: World, picture: np.ndarray, where: np.ndarray) -> set:
    """Colours found at `where`, excluding the avatar's own colours and ignored cells."""
    own = {world.avatar["colour"], 255} | world.avatar["parts"]
    return {c for c in np.unique(picture[where]).tolist() if c not in own}


def _add_roles(world: World, colours: set, role: str) -> None:
    for c in colours:
        world.colour_roles.setdefault(c, Counter())[role] += 1


# ── Deaths ──────────────────────────────────────────────────────────────────


def is_timeout_death(world: World, level: int, life_length: int) -> bool:
    """Was this death just the step counter running out (not the last action's fault)?
    Two deaths at the same life length reveal the step budget; the earlier death is
    then un-blamed too."""
    tolerance = max(TIMEOUT_TOLERANCE, round(TIMEOUT_TOLERANCE_FRACTION * life_length))
    budget = world.rules["steps_per_life"].get(level)
    if budget is not None:
        return life_length >= budget - tolerance
    if life_length < TIMEOUT_MIN_LIFE:
        return False
    same = [d for d in world.deaths if d[0] == level and abs(d[1] - life_length) <= tolerance]
    if not same:
        return False
    world.rules["steps_per_life"][level] = min([life_length] + [d[1] for d in same])
    for d in same:
        _, _, screen, action, avatar_at = d
        if screen.results.get(action) == DIED:
            del screen.results[action]
        key = (avatar_at, action[0])
        if world.actions["died_at"].get(key):
            world.actions["died_at"][key] -= 1
        world.deaths.remove(d)
    return True
