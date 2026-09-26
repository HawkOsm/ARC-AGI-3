"""TRACE — optional step-by-step recording of a game, for eval/trace_viewer.py (watch the agent play).

Off unless the environment variable ARC3_TRACE_DIR names a folder. Then every Brain writes
<ARC3_TRACE_DIR>/<game>.jsonl, one JSON line per action:
    i          step number (0 = the first action)
    state      the game state the agent saw (NOT_FINISHED, GAME_OVER, ...)
    level      levels completed so far
    grid       the screen it saw: base64 of zlib(64x64 bytes); left out when identical to the previous step
    action     [id, x, y]
    why        who chose the action and why (brain explorer / known solution / LLM plan step / reset ...)
    outcome    what the PREVIOUS action did ("ACTION3 -> 12 cells changed ...", "... LEVEL COMPLETE ...")
    theory     the LLM's theory, only on the step where a new one arrived
    check      hypothesis mode: the verdict on the LLM's last test, on the step where it was judged
    life, deaths, screens, avatar, mode    the brain's state at that moment
Nothing in here is read back by the agent; it only writes.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import zlib
from pathlib import Path

import numpy as np

from .world import Action, WorkingMemory, World


class Tracer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", buffering=1)  # line-buffered: a crash still leaves a readable trace
        self._lock = threading.Lock()
        self._last_grid: bytes | None = None
        self._last_theory = ""
        self._step = 0

    @staticmethod
    def for_game(name: str) -> Tracer | None:
        folder = os.environ.get("ARC3_TRACE_DIR")
        if not folder or not name:
            return None
        return Tracer(Path(folder) / f"{name.split('-')[0]}.jsonl")

    def record(self, state: str, grid: np.ndarray | None, action: Action, world: World, memory: WorkingMemory,
               stats: dict, check: str = "") -> None:
        record: dict = {"i": self._step, "state": state, "level": memory.level, "action": list(action),
                        "why": memory.why, "outcome": memory.last_outcome, "life": memory.life_length,
                        "deaths": stats.get("deaths", 0), "avatar": world.avatar.colour,
                        "mode": world.mode.value if world.mode else None}
        level_map = world.levels.get(memory.level) if memory.level is not None else None
        record["screens"] = len(level_map.screens) if level_map else 0
        if grid is not None:
            raw = np.ascontiguousarray(grid, dtype=np.uint8).tobytes()
            if raw != self._last_grid:
                record["grid"] = base64.b64encode(zlib.compress(raw, 6)).decode()
                self._last_grid = raw
        if check:
            record["check"] = check
        if memory.theory and memory.theory != self._last_theory:
            record["theory"] = memory.theory
            self._last_theory = memory.theory
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
        self._step += 1
