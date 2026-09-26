"""KAGGLE AGENT — plugs the Brain into the ARC-AGI-3 framework (the `MyAgent` class Kaggle runs).

The framework calls choose_action() / is_done() once per step. We hand the frame
to the Brain and send back its action. Everything else here is protection against
framework hazards we verified (eval/interface_checks.py):

  * GameAction values are shared by every game thread; ACTION6.set_data() in one
    game changes the click of another. We pass THIS game's click straight to the
    engine instead.
  * A failed step (network error/timeout on Kaggle) returns None and the base
    class then crashes this game's thread. We swallow it, back off and retry.
  * An exception inside choose_action would also kill the thread; we fall back
    to a random legal action.
  * The base class keeps every frame forever (~GBs over long games); we keep a few.
  * Wall-clock and memory guards stop a game before Kaggle kills the whole run.
"""
from __future__ import annotations

import logging
import os
import random
import resource
import time
import traceback
import zlib
from typing import Any, Optional

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

from .brain import Brain
from .settings import (
    KEEP_FRAMES,
    MAX_ACTIONS,
    MAX_CONSECUTIVE_ERRORS,
    MEM_GUARD_MB,
    MEM_GUARD_SCREENS,
    TIME_BUDGET_S,
)

_PROCESS_START = time.time()
_log = logging.getLogger(__name__)
# Local-only overrides (eval/kaggle_sim.py); never set on Kaggle.
_MAX_ACTIONS = int(os.environ.get("ARC3_MAX_ACTIONS", MAX_ACTIONS))
_TIME_BUDGET_S = float(os.environ.get("ARC3_TIME_BUDGET_S", TIME_BUDGET_S))


class MyAgent(Agent):
    MAX_ACTIONS = _MAX_ACTIONS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        base = os.environ.get("ARC3_SEED")
        seed = int(base) if base is not None else int(time.time() * 1e6)
        seed = (seed + zlib.crc32(self.game_id.encode())) % (2**31)
        self.rng = random.Random(seed)
        self.brain = Brain(seed=seed, name=self.game_id)
        self._click: dict = {}  # this game's ACTION6 coordinates
        self.errors = 0
        self.consecutive_errors = 0

    @property
    def name(self) -> str:
        return f"{super().name}.brain"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if time.time() - _PROCESS_START > _TIME_BUDGET_S:
            return True
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return True
        if self.action_counter % 500 == 0 and self._memory_exceeded():
            return True
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            action_id, x, y = self.brain.next_action(
                latest_frame.state.value, latest_frame.frame, latest_frame.levels_completed,
                list(latest_frame.available_actions or []), bool(latest_frame.full_reset))
        except Exception:
            self.errors += 1
            _log.warning("choose_action fallback for %s:\n%s", self.game_id, traceback.format_exc(limit=3))
            legal = [a for a in (latest_frame.available_actions or []) if a != 0] or [0]
            action_id = self.rng.choice(legal)
            x, y = self.rng.randrange(64), self.rng.randrange(64)
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                action_id = 0
        action = GameAction.from_id(action_id)
        self._click = {"x": int(x), "y": int(y)} if action.is_complex() else {}
        return action

    # ── framework overrides (hardening) ─────────────────────────────────────

    def do_action_request(self, action: GameAction) -> FrameData:
        raw = self.arc_env.step(action, data=dict(self._click), reasoning=None)
        return self._convert_raw_frame_data(raw)

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        try:
            frame = super().take_action(action)
        except Exception:
            frame = None
        if frame is None:
            self.errors += 1
            self.consecutive_errors += 1
            self.brain.forget_last_action()
            time.sleep(min(5.0, 0.05 * self.consecutive_errors))
        else:
            self.consecutive_errors = 0
        return frame

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > KEEP_FRAMES:
            del self.frames[: len(self.frames) - KEEP_FRAMES]
        if frame.guid:
            self.guid = frame.guid

    def _memory_exceeded(self) -> bool:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux
        if rss_mb < MEM_GUARD_MB:
            return False
        screens = sum(len(m.screens) for m in self.brain.world.levels.values())
        if screens > MEM_GUARD_SCREENS:
            _log.warning("memory guard: stopping %s (%d screens, RSS %.0f MB)", self.game_id, screens, rss_mb)
            return True
        return False

    # ── diagnostics for eval/run.py (not used by the framework) ─────────────

    def eval_stats(self) -> dict:
        world = self.brain.world
        s = dict(self.brain.stats)
        s["errors"] = self.errors
        s["avatar_colour"] = world.avatar.colour
        s["ignored_cells"] = int(world.counters.ignore.sum())
        s["levels_mapped"] = len(world.levels)
        s["steps_per_life"] = dict(world.rules.steps_per_life)
        return s
