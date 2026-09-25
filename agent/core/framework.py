"""MyAgent: the ARC-AGI-3-Agents framework adapter around Explorer.

Hardening against verified framework hazards (eval/interface_checks.py):
  * GameAction members are process-global singletons; set_data() on ACTION6
    is visible to every Swarm thread. We keep this game's click coordinates on
    the instance and pass them straight to arc_env.step().
  * A failed step (HTTP error/timeout on Kaggle's gateway) returns None and the
    base class then raises, killing this game's thread. We swallow it and retry.
  * An exception inside choose_action also kills the thread; we fall back to a
    legal random action instead.
  * The base class keeps every FrameData forever (~GBs over long games); we
    keep only the last KEEP_FRAMES.
"""
from __future__ import annotations

import logging
import os
import random
import time
import traceback
import zlib
from typing import Any, Optional

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

from .config import KEEP_FRAMES, MAX_ACTIONS, MAX_CONSECUTIVE_ERRORS, TIME_BUDGET_S
from .explorer import Explorer

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
        self.explorer = Explorer(seed=seed)
        self._pending: dict = {}
        self.errors = 0
        self.consecutive_errors = 0

    @property
    def name(self) -> str:
        return f"{super().name}.explorer"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if time.time() - _PROCESS_START > _TIME_BUDGET_S:
            return True
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return True
        if latest_frame.state is GameState.WIN:
            return not self.explorer.wants_more("WIN")
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            aid, x, y = self.explorer.act(
                latest_frame.state.value, latest_frame.frame, latest_frame.levels_completed,
                list(latest_frame.available_actions or []), bool(latest_frame.full_reset))
        except Exception:
            self.errors += 1
            _log.warning("choose_action fallback for %s:\n%s", self.game_id, traceback.format_exc(limit=3))
            avail = [a for a in (latest_frame.available_actions or []) if a != 0] or [0]
            aid = self.rng.choice(avail)
            x, y = self.rng.randrange(64), self.rng.randrange(64)
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                aid = 0
        action = GameAction.from_id(aid)
        self._pending = {"x": int(x), "y": int(y)} if action.is_complex() else {}
        return action

    def do_action_request(self, action: GameAction) -> FrameData:
        # Use this instance's own coordinates, never the shared enum's action_data.
        raw = self.arc_env.step(action, data=dict(self._pending), reasoning=None)
        return self._convert_raw_frame_data(raw)

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        try:
            frame = super().take_action(action)
        except Exception:
            frame = None
        if frame is None:
            self.errors += 1
            self.consecutive_errors += 1
            self.explorer.forget_last()
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

    def eval_stats(self) -> dict:
        """Diagnostics for eval/run.py (not used by the framework)."""
        s = dict(self.explorer.stats)
        s["errors"] = self.errors
        s["avatar_color"] = self.explorer.avatar.color
        s["mask_cells"] = int(self.explorer.mask.mask.sum())
        s["levels_graphed"] = len(self.explorer.graphs)
        return s
