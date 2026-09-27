"""MODEL — the whole agent in one file: an LLM that plays ARC-AGI-3 by writing Python.

Idea: Tufa Labs' "Duck" harness (ARC Prize 2026 Milestone #1 winner, MIT; outline in docs/research/DUCK_HARNESS.md).
This is our own small rewrite of it, not their code.

    one game = many TURNS; one turn:
      1. user message: what the last actions did, level / step, valid actions, the model's own notes, the screen image
      2. the model thinks and calls the `python` tool; its code sees the game (below) and calls action([...]) itself
      3. tool results go back to the model; the turn ends once actions have run (or after MAX_TOOL_CALLS)
    what the model's code can read (sandbox, a fresh process per call, stdlib only):
      current_frame / previous_frame   .ascii (letter grid), .level, .step, .shape, .segmentation (objects + adjacency)
      history, transitions, last_action_result, valid_actions, action(actions)
    memory: the model writes "World model: / Goal model: / Action model: / Plan: / Cross-level notes:" lines; they are
      carried into every later prompt. Old messages are dropped when the context fills; only the newest image is kept.

Run locally:  .venv/bin/python -m agent.model --games ar25 --minutes 10
              (start the model server first: make llm-local)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── settings ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Config:
    """Everything tunable, read from the environment (AGENT_*) with the original Duck's defaults."""
    base_url: str = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000/v1")
    model: str = os.environ.get("AGENT_MODEL", "qwen")
    context_tokens: int = int(os.environ.get("AGENT_CONTEXT", "32768"))
    thinking: bool = os.environ.get("AGENT_THINKING", "1") == "1"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    request_timeout_s: float = float(os.environ.get("AGENT_REQUEST_TIMEOUT", "900"))
    max_tokens: int = int(os.environ.get("AGENT_MAX_TOKENS", "0"))  # per reply, thinking included; 0 = no cap
    tool_timeout_s: int = 30  # one python call
    tool_output_chars: int = 4000  # what the model sees of one call's output
    max_tool_calls: int = 12  # per turn, before the turn is cut
    screen: str = os.environ.get("AGENT_SCREEN", "image")  # the screen in each message: image | text | both | none
    image_upscale: int = int(os.environ.get("AGENT_IMAGE_UPSCALE", "4"))  # pixels per cell in the image
    trace_dir: str = os.environ.get("AGENT_TRACE_DIR", "")


ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE", 7: "UNDO"}
NAME_TO_ID = {name: i for i, name in ACTION_NAMES.items()}
COLOR_CHARS = "WwgGcBMPRbSYOrNp"
COLOR_LEGEND = ("W=white w=light gray g=gray G=dark gray c=charcoal B=black M=magenta P=pink R=red b=blue S=sky blue "
                "Y=yellow O=orange r=dark red N=light green p=purple")
PALETTE = [(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0),
           (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
           (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214)]
SCREEN_TEXT_HEADER = "The current screen, one letter per cell"  # older messages lose this part (like old images)
NOTE_KEYS = ["World model", "Goal model", "Action model", "Recent findings", "Open questions", "Plan",
             "Cross-level notes"]

# ── the prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a coding agent playing an unknown turn-based puzzle game on a 64x64 colour grid.
Nobody tells you the rules or the goal: find them by experiment. Clearing a level shows the next one (levels often
reuse earlier mechanics, with changes). Every in-game action counts: clear every level in as few actions as possible.

You act only through the `python` tool. Each call runs in a fresh process (stdlib only: collections, heapq,
itertools, math, re, json, functools, statistics, copy, random, string, bisect, operator, fractions; 30 s limit;
long output is cut). These variables are already defined in every call (do not import them):
- `current_frame` (the board now) and `previous_frame` (before the last action), each with:
  `.rows` (list of 64 strings, one letter per cell: `current_frame.rows[r][c]`), `.at(r, c)` (one letter),
  `.ascii` (ONE string, rows joined by newlines), `.level`, `.step`, `.shape` (rows, cols), `.segmentation`.
  Colours: {COLOR_LEGEND}.
- `.segmentation` = {{'nodes': [...], 'adjacency': [[i, j], ...]}}. Each node is one 4-connected same-colour object:
  `id`, `color` (letter), `hash` (colour + shape, position-free: equal hash = same object), `size` (cell count),
  `pixels` (list of its [row, col] cells), `bbox` [row0, col0, row1, col1], `children` (ids of objects it fully
  encloses). Use it as your main view.
- `history` (list of entries with `.action` (a string) and `.frame`; history[-1].frame is the current frame),
  `transitions` (`.action`, `.before`, `.after`, `.result`), `last_action_result` (dict: board_changed,
  level_completed, game_over, run_complete, levels_completed), `valid_actions` (list of names).
- `changes()` (or `changes(frame_a, frame_b)`): what changed between two frames by object: moved (with from/to
  bbox and d_row/d_col), appeared, disappeared; the step counter is ignored. Use it after every action.
- `known`: what the agent remembers: `tried_here` (actions already tried on this exact screen and their outcome:
  changed / no change / game over / level completed), `step_counter` (cells that count steps; not puzzle pieces),
  `actions_this_life`, `earlier_lives_lasted` (actions per life that ended in game over: your step budget).
- `memory`: a dict you may fill (JSON values only); it is kept between your python calls. Store what you learned
  (action effects, the goal, coordinates) so you do not recompute it.
- Already imported for you: collections, Counter, defaultdict, deque, heapq, itertools, math. Files are not available.
- `action(actions)` executes real game actions and refreshes every variable above. Pass a list:
  ['UP'], ['LEFT', 'LEFT'], or [{{'action': 'MOUSE', 'row': 12, 'col': 30}}]. UP DOWN LEFT RIGHT SPACE MOUSE UNDO.
  It returns a dict (executed, outcome, board_changed, level_completed, game_over, run_complete, levels_completed).
  An action that already gave "no change" or "game over" on this exact screen is not sent again (it would only cost
  an action); add 'force': True to a dict action to send it anyway.
- Inspect and act in the SAME call when you can: every call costs time.

How to play well:
- Keep a compact world model: what the objects are, what each action does, what the goal seems to be, what is
  unsure. Before acting, write it as lines starting `World model:`, `Goal model:`, `Action model:`,
  `Recent findings:`, `Open questions:`, `Plan:`, `Cross-level notes:`; they are shown back to you every turn.
- Diff `previous_frame` and `current_frame` in code after actions. Separate real game changes from HUD changes:
  a bar or strip of cells along a screen edge that changes every step is usually a step counter, not a puzzle piece.
- When unsure, program ONE discriminating probe. When the goal is understood, write a search (BFS, flood fill,
  shortest path, small beam search) over the state you inferred and batch the resulting actions in one call.
- Print compact summaries (object lists, diffs, counts), never whole boards.
- Stop acting inside your code as soon as a result says level_completed, game_over or run_complete; look again first.
"""


# ── the sandbox (runs the model's code in a separate process) ───────────────

SANDBOX = r'''
import builtins, contextlib, hashlib, io, json, resource, signal, sys
ALLOWED = {"bisect", "collections", "copy", "fractions", "functools", "hashlib", "heapq", "itertools", "json",
           "math", "operator", "random", "re", "statistics", "string"}
_real_import = builtins.__import__
def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {name!r} is not allowed")
    return _real_import(name, globals, locals, fromlist, level)
CHARS = "WwgGcBMPRbSYOrNp"

def segment(grid):
    h, w = len(grid), len(grid[0]) if grid else 0
    comp = [[-1] * w for _ in range(h)]
    nodes, cells_of = [], []
    for r0 in range(h):
        for c0 in range(w):
            if comp[r0][c0] != -1:
                continue
            v, cid, stack, cells = grid[r0][c0], len(nodes), [(r0, c0)], []
            comp[r0][c0] = cid
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and comp[nr][nc] == -1 and grid[nr][nc] == v:
                        comp[nr][nc] = cid
                        stack.append((nr, nc))
            rs, cs = [p[0] for p in cells], [p[1] for p in cells]
            top, left = min(rs), min(cs)
            shape = sorted((r - top, c - left) for r, c in cells)
            signature = hashlib.sha1(repr((v, shape)).encode()).hexdigest()[:12]
            nodes.append({"id": cid, "color": CHARS[v], "hash": signature, "size": len(cells),
                          "pixels": [[r, c] for r, c in sorted(cells)],
                          "bbox": [top, left, max(rs), max(cs)], "children": []})
            cells_of.append(cells)
    adj = set()
    for r in range(h):
        for c in range(w):
            a = comp[r][c]
            if r + 1 < h and comp[r + 1][c] != a:
                adj.add((min(a, comp[r + 1][c]), max(a, comp[r + 1][c])))
            if c + 1 < w and comp[r][c + 1] != a:
                adj.add((min(a, comp[r][c + 1]), max(a, comp[r][c + 1])))
    # containment: a node touching no screen edge, all of whose neighbours are one other node, is its child
    neighbours = {}
    for a, b in adj:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    for n in nodes:
        r0, c0, r1, c1 = n["bbox"]
        outer = neighbours.get(n["id"], set())
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1 and len(outer) == 1:
            parent = next(iter(outer))
            if nodes[parent]["size"] > n["size"]:
                nodes[parent]["children"].append(n["id"])
    return {"nodes": nodes, "adjacency": sorted([a, b] for a, b in adj)}

class Frame:
    def __init__(self, d):
        self.grid, self.level, self.step = d["grid"], d["level"], d["step"]
        self.shape = (len(self.grid), len(self.grid[0]) if self.grid else 0)
        self._seg = None
    @property
    def rows(self):
        return ["".join(CHARS[v] for v in row) for row in self.grid]
    @property
    def ascii(self):
        return "\n".join(self.rows)
    def at(self, row, col):
        return CHARS[self.grid[row][col]]
    @property
    def segmentation(self):
        if self._seg is None:
            self._seg = segment(self.grid)
        return self._seg
    def __repr__(self):
        return f"<Frame level={self.level} step={self.step}>"

class Entry:
    def __init__(self, action, frame):
        self.action, self.frame = action, frame
    def __getattr__(self, name):  # entry.ascii etc. mean the entry's frame
        return getattr(self.__dict__["frame"], name)
    def __repr__(self):
        return f"<Entry {self.action} -> {self.frame!r}>"

class Transition:
    def __init__(self, action, before, after, result):
        self.action, self.before, self.after, self.result = action, before, after, result

_proto_out, _proto_in = sys.__stdout__, sys.__stdin__
def _send(msg):
    _proto_out.write(json.dumps(msg) + "\n"); _proto_out.flush()
def _recv():
    return json.loads(_proto_in.readline())

def _no_files(*args, **kwargs):
    raise PermissionError("files are not available here: keep what you need in variables")
import collections, heapq, itertools, math
env = {"__builtins__": dict(vars(builtins), __import__=_import, open=_no_files),
       # preloaded: the model's code uses these without importing them (63 NameErrors on Kaggle, run C)
       "collections": collections, "Counter": collections.Counter, "defaultdict": collections.defaultdict,
       "deque": collections.deque, "heapq": heapq, "itertools": itertools, "math": math}
def changes(before=None, after=None):
    """What changed between two frames (default: previous_frame -> current_frame), by object: objects with the same
    hash in both but somewhere else moved; the rest appeared or disappeared. Ignores cells the step counter uses."""
    before = before if before is not None else env["previous_frame"]
    after = after if after is not None else env["current_frame"]
    if before is None or after is None:
        return {"moved": [], "appeared": [], "disappeared": []}
    hud = {tuple(c) for c in env["known"].get("hud_cells", [])}
    def objects(frame):
        return [n for n in frame.segmentation["nodes"] if not all(tuple(p) in hud for p in n["pixels"])]
    old, new = objects(before), objects(after)
    old_at = {(n["hash"], tuple(n["bbox"])) for n in old}
    new_at = {(n["hash"], tuple(n["bbox"])) for n in new}
    gone = [n for n in old if (n["hash"], tuple(n["bbox"])) not in new_at]
    came = [n for n in new if (n["hash"], tuple(n["bbox"])) not in old_at]
    moved = []
    for n in list(came):
        match = next((o for o in gone if o["hash"] == n["hash"]), None)
        if match is not None:
            gone.remove(match); came.remove(n)
            moved.append({"color": n["color"], "hash": n["hash"], "from": match["bbox"], "to": n["bbox"],
                          "d_row": n["bbox"][0] - match["bbox"][0], "d_col": n["bbox"][1] - match["bbox"][1]})
    brief = lambda n: {"color": n["color"], "hash": n["hash"], "bbox": n["bbox"], "size": n["size"]}
    return {"moved": moved, "appeared": [brief(n) for n in came], "disappeared": [brief(n) for n in gone]}

def _load(state):
    entries = [Entry(e["action"], Frame(e["frame"])) for e in state["history"]]
    history = (env.get("history", []) + entries)[-40:] if state.get("append") else entries
    env["history"] = history
    env["known"] = state.get("known", {})
    env["current_frame"] = history[-1].frame if history else None
    env["previous_frame"] = history[-2].frame if len(history) > 1 else None
    env["transitions"] = [Transition(history[i].action, history[i - 1].frame, history[i].frame,
                                     state["last_action_result"] if i == len(history) - 1 else {})
                          for i in range(1, len(history))]
    env["last_transition"] = env["transitions"][-1] if env["transitions"] else None
    env["last_action_result"] = state["last_action_result"]
    env["valid_actions"] = state["valid_actions"]
def action(actions):
    if isinstance(actions, (str, dict)):
        actions = [actions]
    _send({"type": "action", "actions": list(actions)})
    reply = _recv()
    _load(reply["state"])
    return reply["result"]
env["action"] = action
env["changes"] = changes

def _timeout(signum, frame):
    raise TimeoutError("the python call hit its time limit")

request = _recv()
_load(request["state"])
env["memory"] = request["state"].get("memory") or {}
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(request["timeout"])
out, error = io.StringIO(), None
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        exec(compile(request["code"], "<python>", "exec"), env)
except BaseException as e:
    error = f"{type(e).__name__}: {e}"
signal.alarm(0)
result = env.get("result")
try:
    memory = json.loads(json.dumps(env.get("memory", {})))
except (TypeError, ValueError):
    memory, error = None, (error or "") + " | `memory` must hold JSON values only (dicts, lists, str, numbers)"
_send({"type": "done", "stdout": out.getvalue(), "error": error, "memory": memory,
       "result": None if result is None else repr(result)[:2000]})
'''


class Sandbox:
    """Runs one piece of model code in a fresh `python -I -S` process; its action() calls come back to `act`."""

    @staticmethod
    def run(code: str, state: dict[str, Any], act: Callable[[list[Any]], tuple[dict[str, Any], dict[str, Any]]],
            timeout_s: int) -> dict[str, Any]:
        workdir = tempfile.mkdtemp(prefix="agent_sandbox_")  # its own folder, removed after the call
        process = subprocess.Popen([sys.executable, "-I", "-S", "-c", SANDBOX], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                                   cwd=workdir)
        assert process.stdin is not None and process.stdout is not None
        killer = threading.Timer(timeout_s + 10, process.kill)  # alarm failed to stop it: kill the process
        killer.start()
        actions_run: list[dict[str, Any]] = []
        try:
            process.stdin.write(json.dumps({"code": code, "state": state, "timeout": timeout_s}) + "\n")
            process.stdin.flush()
            while True:
                line = process.stdout.readline()
                if not line:
                    return {"error": "the python process ended unexpectedly (time or memory limit?)",
                            "actions": actions_run}
                message = json.loads(line)
                if message["type"] == "done":
                    message["actions"] = actions_run
                    return message
                result, new_state = act(message["actions"])
                actions_run.append(result)
                process.stdin.write(json.dumps({"result": result, "state": new_state}) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            return {"error": f"sandbox failure: {e}", "actions": actions_run}
        finally:
            killer.cancel()
            process.kill()
            shutil.rmtree(workdir, ignore_errors=True)


# ── what the agent remembers for the model (pieces of the old brain, git 4eab2e7^: learn.CounterLearner,
#    LevelMemory's screen map, learn.DeathJudge) ────────────────────────────────

COUNTER_TICK_MAX_CELLS = 4  # a change this small ...
COUNTER_BORDER_BAND = 3  # ... this close to the screen edge is a counter "tick"
COUNTER_TRACK_MIN_POSITIONS = 3  # a line with ticks at this many places, close together, is a counter track
COUNTER_TRACK_MAX_GAP = 2


@dataclass(slots=True)
class Hud:
    """Step counters and timer bars: small changes at the screen edge that advance along a line. Hidden when
    screens are compared, so a ticking counter is not taken for a real change (the old brain's CounterLearner)."""
    ignore: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=bool))
    along_row: list[set[int]] = field(default_factory=lambda: [set() for _ in range(64)])
    along_col: list[set[int]] = field(default_factory=lambda: [set() for _ in range(64)])

    def learn(self, before: np.ndarray, after: np.ndarray) -> None:
        if before.shape != after.shape or self.ignore.shape != after.shape:
            self.ignore = np.zeros(after.shape, dtype=bool)
            self.along_row = [set() for _ in range(after.shape[0])]
            self.along_col = [set() for _ in range(after.shape[1])]
            return
        edge = np.zeros(after.shape, dtype=bool)
        b = COUNTER_BORDER_BAND
        edge[:b, :] = edge[-b:, :] = edge[:, :b] = edge[:, -b:] = True
        for blob in _blobs(before != after):
            if len(blob) <= COUNTER_TICK_MAX_CELLS and all(edge[y, x] for y, x in blob):
                for y, x in blob:
                    self.ignore[y, x] = True
                cy, cx = round(sum(y for y, _ in blob) / len(blob)), round(sum(x for _, x in blob) / len(blob))
                for y in {y for y, _ in blob}:
                    self.along_row[y].add(cx)
                for x in {x for _, x in blob}:
                    self.along_col[x].add(cy)
        for r, positions in enumerate(self.along_row):
            if _is_track(positions):
                self.ignore[r, :] = True
        for c, positions in enumerate(self.along_col):
            if _is_track(positions):
                self.ignore[:, c] = True

    def hide(self, grid: np.ndarray) -> np.ndarray:
        picture = grid.copy()
        if picture.shape == self.ignore.shape:
            picture[self.ignore] = -1
        return picture

    def summary(self) -> str:
        if not self.ignore.any():
            return "none found yet"
        rows = [r for r in range(self.ignore.shape[0]) if self.ignore[r].all()]
        cols = [c for c in range(self.ignore.shape[1]) if self.ignore[:, c].all()]
        parts = [f"row {r}" for r in rows] + [f"col {c}" for c in cols]
        return ", ".join(parts) if parts else f"{int(self.ignore.sum())} cells at the screen edge"


def _is_track(positions: set[int]) -> bool:
    if len(positions) < COUNTER_TRACK_MIN_POSITIONS:
        return False
    p = sorted(positions)
    return all(b - a <= COUNTER_TRACK_MAX_GAP for a, b in zip(p, p[1:], strict=False))


def _blobs(changed: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected groups of True cells."""
    todo = set(zip(*(axis.tolist() for axis in np.nonzero(changed)), strict=True))
    blobs = []
    while todo:
        start = todo.pop()
        blob, queue = [start], [start]
        while queue:
            y, x = queue.pop()
            for n in ((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                if n in todo:
                    todo.remove(n)
                    blob.append(n)
                    queue.append(n)
        blobs.append(blob)
    return blobs


@dataclass(slots=True)
class Memory:
    """What happened so far, for the model and for not wasting actions: every (screen, action) tried and its
    outcome (the old brain's screen map), step counters (Hud), and how long each life lasted (its DeathJudge)."""
    hud: Hud = field(default_factory=Hud)
    tried: dict[tuple[int, int], dict[str, str]] = field(default_factory=dict)  # (level, screen) -> action -> outcome
    life_actions: int = 0  # actions since the level (re)started
    life_lengths: list[int] = field(default_factory=list)  # actions in lives that ended in GAME_OVER

    def screen(self, grid: np.ndarray) -> int:
        return hash(self.hud.hide(grid).tobytes())

    def known(self, level: int, grid: np.ndarray) -> dict[str, str]:
        return self.tried.get((level, self.screen(grid)), {})

    def record(self, level: int, before: np.ndarray, label: str, after: np.ndarray, outcome: str) -> None:
        self.hud.learn(before, after)
        self.tried.setdefault((level, self.screen(before)), {})[label] = outcome

    def summary(self, level: int, grid: np.ndarray) -> dict[str, Any]:
        return {"tried_here": self.known(level, grid), "step_counter": self.hud.summary(),
                "hud_cells": [[int(r), int(c)] for r, c in zip(*np.nonzero(self.hud.ignore), strict=True)],
                "actions_this_life": self.life_actions, "earlier_lives_lasted": self.life_lengths[-5:]}


# ── the game ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Game:
    """One game through arc_agi: its frames, and actions in the names the model uses."""
    env: Any
    game_id: str
    frames: list[tuple[str, list[list[int]], int, int]] = field(default_factory=list)  # (action, grid, level, step)
    state: str = "NOT_PLAYED"
    levels: int = 0
    valid: list[int] = field(default_factory=list)
    actions_taken: int = 0
    memory: Memory = field(default_factory=Memory)

    def start(self) -> None:
        self._take(self.env.reset(), "")

    def _take(self, frame: Any, label: str) -> None:
        self.state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
        self.levels = int(frame.levels_completed)
        self.valid = [int(a) for a in (frame.available_actions or [])]
        if frame.frame:
            grid = np.asarray(frame.frame[-1], dtype=np.int64).tolist()
            self.frames.append((label, grid, self.levels + 1, self.actions_taken))

    def do(self, action: Any) -> dict[str, Any]:
        """One action; a result dict the model reads (level_completed, game_over, board_changed, ...). An action
        that already did nothing (or lost the game) on this exact screen is not repeated unless forced: every action
        counts, and the games are deterministic."""
        from arcengine import GameAction
        name, data, label = self._parse(action)
        if name not in NAME_TO_ID or NAME_TO_ID[name] not in self.valid:
            return {"executed": False, "error": f"{label} is not a valid action now; valid: {self.valid_names()}"}
        before, level = np.array(self.frames[-1][1]), self.levels + 1
        known = self.memory.known(level, before).get(label)
        if known in ("no change", "game over") and not (isinstance(action, dict) and action.get("force")):
            return {"executed": False, "action": label, "skipped": f"not sent: on this exact screen {label} already "
                    f"gave '{known}' (pass {{'action': ..., 'force': True}} to send it anyway)"}
        levels_before = self.levels
        self.actions_taken += 1
        self.memory.life_actions += 1
        self._take(self.env.step(GameAction.from_id(NAME_TO_ID[name]), data=data), label)
        after = np.array(self.frames[-1][1])
        game_over, completed = self.state == "GAME_OVER", self.levels > levels_before
        changed = bool((self.memory.hud.hide(before) != self.memory.hud.hide(after)).any())
        outcome = ("level completed" if completed else "game over" if game_over
                   else "changed" if changed else "no change")
        self.memory.record(level, before, label, after, outcome)
        if game_over:
            self.memory.life_lengths.append(self.memory.life_actions)
        if completed or game_over:
            self.memory.life_actions = 0
        return {"executed": True, "action": label, "outcome": outcome, "board_changed": changed,
                "level_completed": completed, "levels_completed": self.levels, "game_over": game_over,
                "run_complete": self.state == "WIN", "valid_actions": self.valid_names()}

    @staticmethod
    def _parse(action: Any) -> tuple[str, dict[str, int], str]:
        if isinstance(action, dict):
            name = str(action.get("action", "")).upper()
            if name == "MOUSE":
                row, col = int(action.get("row", -1)), int(action.get("col", -1))
                return name, {"x": col, "y": row}, f"MOUSE(row={row}, col={col})"
            return name, {}, name
        name = str(action).strip().upper()
        return name, {}, name

    def reset(self) -> None:
        from arcengine import GameAction
        self._take(self.env.step(GameAction.RESET), "RESET")

    def valid_names(self) -> list[str]:
        return [ACTION_NAMES[a] for a in self.valid if a in ACTION_NAMES]

    def sandbox_state(self, last_result: dict[str, Any], since: int = 0) -> dict[str, Any]:
        """What the model's code sees: the frames after index `since` (at most the last 40), the last result, the
        valid actions, and what the memory knows about the current screen."""
        start = max(since, len(self.frames) - 40)
        history = [{"action": a, "frame": {"grid": g, "level": lv, "step": s}} for a, g, lv, s in self.frames[start:]]
        known = self.memory.summary(self.levels + 1, np.array(self.frames[-1][1])) if self.frames else {}
        return {"history": history, "append": since > 0, "last_action_result": last_result,
                "valid_actions": self.valid_names(), "known": known}


# ── the model ───────────────────────────────────────────────────────────────

PYTHON_TOOL = {"type": "function", "function": {
    "name": "python", "description": "Run Python that inspects the game and may call action([...]).",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
TOOL_MARKUP = re.compile(r"<function=python>\s*<parameter=code>\s*(.*?)\s*</parameter>", re.S)


class LLM:
    """OpenAI-compatible chat with the python tool (vLLM: --tool-call-parser qwen3_coder --reasoning-parser qwen3)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        c = self.config
        body = {"model": c.model, "messages": messages, "tools": [PYTHON_TOOL], "tool_choice": "auto",
                "temperature": c.temperature, "top_p": c.top_p, "top_k": c.top_k,
                "chat_template_kwargs": {"enable_thinking": c.thinking}}
        if c.max_tokens > 0:
            body["max_tokens"] = c.max_tokens
        request = urllib.request.Request(f"{c.base_url}/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=c.request_timeout_s) as response:
            answer = json.loads(response.read())
        message = answer["choices"][0]["message"]
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        content = message.get("content") or ""
        calls = message.get("tool_calls") or []
        if not calls:  # the parser missed a call written as markup: recover it
            calls = [{"id": f"recovered{i}", "type": "function",
                      "function": {"name": "python", "arguments": json.dumps({"code": code})}}
                     for i, code in enumerate(TOOL_MARKUP.findall(content + reasoning))]
            content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()
        return {"content": content, "reasoning": reasoning, "tool_calls": calls,
                "usage": answer.get("usage", {})}


# ── the conversation ────────────────────────────────────────────────────────


@dataclass(slots=True)
class Conversation:
    """Messages sent to the model: the system prompt, recent turns, and the model's notes."""
    budget_tokens: int
    messages: list[dict[str, Any]] = field(default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}])
    notes: dict[str, str] = field(default_factory=dict)

    def add(self, message: dict[str, Any]) -> None:
        if message["role"] == "user":  # only the newest screen is kept (image or letter grid); older ones go
            for old in self.messages:
                if old["role"] == "user" and isinstance(old.get("content"), list):
                    old["content"] = " ".join(p.get("text", "") for p in old["content"]
                                              if p["type"] == "text" and not p["text"].startswith(SCREEN_TEXT_HEADER))
        self.messages.append(message)
        self._fit()

    def learn_notes(self, text: str) -> None:
        """Keep the model's `World model: ...` style lines (the newest version of each)."""
        any_key = "|".join(NOTE_KEYS)
        for key in NOTE_KEYS:
            m = re.search(rf"^\s*[*#-]*\s*{re.escape(key)}\s*[:：]\s*(.+?)(?=\n\s*[*#-]*\s*(?:{any_key})\s*[:：]|\Z)",
                          text, re.S | re.M | re.I)
            if m:
                self.notes[key] = " ".join(m.group(1).split())[:600]

    def notes_text(self) -> str:
        if not self.notes:
            return ""
        lines = [f"- {k}: {self.notes[k]}" for k in NOTE_KEYS if k in self.notes]
        return "Your world model so far (revise it when the evidence disagrees):\n" + "\n".join(lines)

    def _fit(self) -> None:
        """Drop whole old turns (a user message and its replies) until the estimate fits; if one turn alone is too
        big, drop its oldest tool exchanges. The system prompt and the newest user message always stay."""
        while self._size() > self.budget_tokens:
            users = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
            if len(users) >= 2:
                del self.messages[1:users[1]]
                continue
            first = users[0] + 1 if users else 1
            if first >= len(self.messages) - 2:
                break  # only the newest exchange is left: send it as it is
            end = first + 1  # an assistant message and the tool results that answer it
            while end < len(self.messages) and self.messages[end]["role"] == "tool":
                end += 1
            del self.messages[first:end]

    def _size(self) -> int:
        """Rough token count: characters / 3, images counted by their text only (base64 is not tokens)."""
        chars = 0
        for m in self.messages:
            content = m.get("content")
            if isinstance(content, list):
                chars += sum(len(p.get("text", "")) for p in content) + 1200
            elif content:
                chars += len(content)
            chars += len(m.get("reasoning_content") or "") + len(json.dumps(m.get("tool_calls") or ""))
        return chars // 3 + 400


# ── the player ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Agent:
    """Plays one game: turns until the game is won, the time is up or the action cap is reached."""
    game: Game
    llm: LLM
    config: Config
    deadline: float
    max_actions: int | None = None
    conversation: Conversation = field(init=False)
    last_result: dict[str, Any] = field(default_factory=dict)
    last_summary: str = "No action has been taken yet."
    turn: int = 0
    tokens: int = 0
    trace: Any = None
    saved: dict[str, Any] = field(default_factory=dict)  # the model's `memory` dict, kept between python calls

    def __post_init__(self) -> None:
        self.conversation = Conversation(int(self.config.context_tokens * 0.8))
        if self.config.trace_dir:
            Path(self.config.trace_dir).mkdir(parents=True, exist_ok=True)
            self.trace = open(Path(self.config.trace_dir) / f"{self.game.game_id}.jsonl", "w")

    def done(self) -> bool:
        return (self.game.state == "WIN" or time.time() > self.deadline
                or (self.max_actions is not None and self.game.actions_taken >= self.max_actions))

    def play(self) -> None:
        self.game.start()
        while not self.done():
            if self.game.state == "GAME_OVER":
                self.game.reset()
                self.last_summary = "You lost: the level restarted (RESET)."
                continue
            try:
                self.play_turn()
            except Exception as e:  # a failed request or tool call must not end the game
                self._log({"turn": self.turn, "error": f"{type(e).__name__}: {e}"[:500]})
                time.sleep(5)
        if self.trace is not None:
            self.trace.close()

    def play_turn(self) -> None:
        self.turn += 1
        self.conversation.add({"role": "user", "content": self._user_content()})
        for _call in range(self.config.max_tool_calls):
            if self.done():
                return
            reply = self.llm.chat(self.conversation.messages)
            self.tokens += int(reply["usage"].get("completion_tokens", 0) or 0)
            if reply["content"]:
                self.conversation.learn_notes(reply["content"])
            assistant: dict[str, Any] = {"role": "assistant", "content": reply["content"] or None}
            if reply["reasoning"]:
                assistant["reasoning_content"] = reply["reasoning"]
            if not reply["tool_calls"]:
                self.conversation.add(assistant)
                self.conversation.add({"role": "user", "content": (
                    "You have not acted yet. Update your world model lines, then call the `python` tool: inspect the "
                    "frames, decide, and call action([...]) from your code.")})
                continue
            assistant["tool_calls"] = reply["tool_calls"]
            self.conversation.add(assistant)
            acted = False
            for call in reply["tool_calls"]:
                output = self._run_code(call)
                acted = acted or bool(output["actions"])
                self.conversation.add({"role": "tool", "tool_call_id": call.get("id", "call"),
                                       "content": self._tool_text(output)})
                self._log({"turn": self.turn, "notes": dict(self.conversation.notes), "code": output["code"][:4000],
                           "stdout": (output.get("stdout") or "")[:2000], "error": output.get("error"),
                           "actions": [a.get("action") for a in output["actions"]], "levels": self.game.levels,
                           "reasoning": reply["reasoning"][-1500:]})
            if acted:
                return

    def _run_code(self, call: dict[str, Any]) -> dict[str, Any]:
        try:
            code = json.loads(call["function"]["arguments"] or "{}").get("code", "")
        except (ValueError, KeyError, AttributeError):
            return {"code": "", "error": "the tool arguments were not valid JSON with a `code` string", "actions": []}
        stop = {"hit": False}
        ran: list[dict[str, Any]] = []  # every single action this call ran, in order
        sent = {"frames": len(self.game.frames)}  # frames the sandbox has; action replies carry only newer ones

        def act(actions: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            results = []
            for a in actions:
                if stop["hit"] or self.done():
                    break
                r = self.game.do(a)
                results.append(r)
                ran.append(r)
                if r.get("level_completed") or r.get("game_over") or r.get("run_complete") or not r["executed"]:
                    stop["hit"] = True
            result = results[-1] if results else {"executed": False, "error": "stopped: level ended or time is up"}
            result = dict(result, executed_count=sum(1 for r in results if r.get("executed")),
                          requested_count=len(actions))
            self.last_result = result
            delta = self.game.sandbox_state(result, since=sent["frames"])
            sent["frames"] = len(self.game.frames)
            return result, delta

        state = dict(self.game.sandbox_state(self.last_result), memory=self.saved)
        output = Sandbox.run(code, state, act, self.config.tool_timeout_s)
        output["code"], output["actions"] = code, ran
        if isinstance(output.get("memory"), dict):
            self.saved = output["memory"]
        if ran:
            executed = [a.get("action", "?") for a in ran if a.get("executed")]
            last = ran[-1]
            completed = any(a.get("level_completed") for a in ran)
            self.last_summary = (f"Your code executed {len(executed)} action(s): {', '.join(executed[-12:])}. "
                                 + ("A LEVEL WAS COMPLETED. " if completed else "")
                                 + ("GAME OVER: the level restarts. " if last.get("game_over") else ""))
        return output

    def _tool_text(self, output: dict[str, Any]) -> str:
        parts = []
        if output.get("stdout"):
            parts.append(output["stdout"])
        if output.get("result") is not None:
            parts.append(f"result = {output['result']}")
        if output.get("error"):
            parts.append(f"ERROR: {output['error']}")
        if output["actions"]:
            parts.append(f"actions run: {json.dumps(output['actions'][-3:])}")
        text = "\n".join(parts) or "(no output)"
        limit = self.config.tool_output_chars
        return text if len(text) <= limit else text[:limit] + f"\n... (cut: {len(text) - limit} more characters)"

    def _user_content(self) -> Any:
        g = self.game
        text = "\n".join(filter(None, [
            self.last_summary,
            f"Now: level {g.levels + 1}, {g.actions_taken} actions so far, state {g.state}. "
            f"Valid actions: {', '.join(g.valid_names())}.",
            self._memory_line(),
            self.conversation.notes_text(),
            "Look at what changed (changes()), update your world model lines, then act through the `python` tool.",
        ]))
        screen = self.config.screen
        if screen not in ("image", "text", "both", "none"):
            raise ValueError(f"AGENT_SCREEN must be image, text, both or none, not {screen!r}")
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if screen in ("text", "both"):
            parts.append({"type": "text", "text": letter_grid(g.frames[-1][1])})
        if screen in ("image", "both"):
            png = png_of(g.frames[-1][1], self.config.image_upscale)
            parts.append({"type": "text", "text": "The current screen as an image:"})
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
        return parts if len(parts) > 1 else text

    def _memory_line(self) -> str:
        """One line from the memory: what was tried on this screen, the step counter, how long lives last."""
        g = self.game
        if not g.frames:
            return ""
        m = g.memory.summary(g.levels + 1, np.array(g.frames[-1][1]))
        tried = ", ".join(f"{a}: {o}" for a, o in list(m["tried_here"].items())[:12]) or "nothing yet"
        lives = f"; earlier lives lasted {m['earlier_lives_lasted']} actions" if m["earlier_lives_lasted"] else ""
        return (f"Already tried on this exact screen: {tried}. Step counter: {m['step_counter']}. "
                f"This life: {m['actions_this_life']} actions{lives}.")

    def _log(self, record: dict[str, Any]) -> None:
        if self.trace is not None:
            self.trace.write(json.dumps(record) + "\n")
            self.trace.flush()


def letter_grid(grid: list[list[int]]) -> str:
    """The screen as text: one letter per cell (the same letters as `.ascii`), row numbers on the left and column
    numbers on top, so a cell can be read off as (row, col)."""
    width = len(grid[0]) if grid else 0
    tens = "".join(str(c // 10) if c % 10 == 0 else " " for c in range(width))
    units = "".join(str(c % 10) for c in range(width))
    rows = [f"{r:2d} " + "".join(COLOR_CHARS[v] for v in row) for r, row in enumerate(grid)]
    return f"{SCREEN_TEXT_HEADER} (rows 0-{len(grid) - 1}, cols 0-{width - 1}):\n   {tens}\n   {units}\n" \
        + "\n".join(rows)


def png_of(grid: list[list[int]], scale: int) -> bytes:
    """The screen as a PNG (ARC palette), no image library needed."""
    colours = np.array(PALETTE, dtype=np.uint8)[np.clip(np.array(grid), 0, 15)]
    image = colours.repeat(scale, axis=0).repeat(scale, axis=1)
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ── running games ───────────────────────────────────────────────────────────


def play_games(arcade: Any, game_ids: list[str], minutes: float, concurrency: int, config: Config,
               max_actions: int | None = None) -> dict[str, dict[str, Any]]:
    """Plays the games in parallel on one scorecard; returns the official score, levels and actions per game."""
    card = arcade.open_scorecard(tags=["agent"])
    llm = LLM(config)

    def one(game_id: str) -> None:
        agent = Agent(Game(arcade.make(game_id, scorecard_id=card), game_id), llm, config,
                    time.time() + minutes * 60, max_actions)
        agent.play()
        print(f"[{game_id}] levels {agent.game.levels}, actions {agent.game.actions_taken}, turns {agent.turn}, "
              f"generated tokens {agent.tokens}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, game_ids))
    scorecard = arcade.close_scorecard(card)
    results = {}
    for e in scorecard.environments:
        results[e.id] = {"score": float(e.score), "levels": int(e.levels_completed), "actions": int(e.actions)}
    return results


def open_arcade(mode: str, environments_dir: str = "environment_files") -> Any:
    """The games: local files (offline), or Kaggle's game server during a scored run (competition: one scorecard,
    each game once; the server's address comes from ARC_BASE_URL)."""
    import logging

    import arc_agi
    logging.disable(logging.WARNING)
    if mode == "competition":
        return arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION,
                              arc_base_url=os.environ.get("ARC_BASE_URL", "http://gateway:8001/"),
                              arc_api_key=os.environ.get("ARC_API_KEY", ""), environments_dir="")
    return arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=environments_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="ar25", help="'all' or comma-separated short ids")
    ap.add_argument("--minutes", type=float, default=10.0, help="time per game")
    ap.add_argument("--max-actions", type=int, default=0, help="action cap per game (0 = none)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--mode", choices=["offline", "competition"], default="offline")
    ap.add_argument("--environments", default="environment_files")
    ap.add_argument("--out", default="", help="write the per-game results as JSON here")
    args = ap.parse_args()
    arcade = open_arcade(args.mode, args.environments)
    ids = sorted(e.game_id for e in arcade.available_environments)
    if args.games != "all":
        ids = [i for i in ids if i.split("-")[0] in args.games.split(",")]
    started = time.time()
    results = play_games(arcade, ids, args.minutes, args.concurrency, Config(), args.max_actions or None)
    for game_id, r in sorted(results.items()):
        print(f"{game_id}: score {r['score']:.2f}, levels {r['levels']}, actions {r['actions']}")
    mean = sum(r["score"] for r in results.values()) / max(1, len(results))
    print(f"mean score {mean:.2f} over {len(results)} games, total levels "
          f"{sum(r['levels'] for r in results.values())}, {time.time() - started:.0f} s")
    if args.out:
        Path(args.out).write_text(json.dumps({"mean_score": mean, "games": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
