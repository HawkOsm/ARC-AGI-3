"""ASK — an optional advisor: a local open-weight LLM behind an OpenAI-compatible API (vLLM).

The brain stays in charge. When the advisor is on (environment variable ARC3_LLM_URL is set)
and the current level has no known solution yet, the brain shows the model the screen, what
the World believes, and what its recent steps did, and asks for the next few actions. The
brain plays that plan one action at a time, still learning from every step, and drops the
plan as soon as the game surprises it (a death, a new level, an action that changed nothing).
No answer, a slow answer or an unreadable answer -> the brain decides exactly as before.
Nothing here is game-specific.

    advisor()                              the shared client for all game threads (None when off)
    Advisor.propose(prompt, available)     -> Advice(plan=[Action, ...], theory) (plan may be empty)
    PromptWriter(world, memory, stats)
      .describe(grid, available)           the prompt: beliefs, recent steps, the screen
      .describe_step(action, before, after, outcome)   one line of history
    parse_plan(text, available)            the model's answer -> [Action, ...]
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from .scene import LevelScene, Observation, SceneObject
from .world import Action, Goal, WorkingMemory, World
from .settings import (
    LLM_FAILS_BEFORE_PAUSE,
    LLM_HISTORY,
    LLM_MAX_TOKENS,
    LLM_MAX_CONCURRENT,
    LLM_MAX_TOKENS_THINKING,
    LLM_MODEL,
    LLM_PAUSE_S,
    LLM_PLAN_MAX,
    LLM_SAMPLING,
    LLM_SAMPLING_THINKING,
    LLM_STRUCTURED_OUTPUT,
    LLM_THINKING,
    LLM_TIMEOUT_S,
    LLM_TIMEOUT_THINKING_S,
)

SYSTEM_PROMPT = (
    "You are playing an unknown turn-based puzzle game on a 64x64 grid. Nobody tells you the rules or the "
    "goal: you find them out by acting and watching what changes.\n\n"
    "- Each cell holds a colour 0-15, written as one hex digit 0-f. Row y goes top (0) to bottom (63), "
    "column x left (0) to right (63).\n"
    "- ACTION1..ACTION5 are simple inputs (often ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right, "
    "ACTION5=use/select, but check the evidence). ACTION6 x y clicks cell (x, y). ACTION7 is usually undo. "
    "Only the listed available actions can be used.\n"
    "- A level is won by reaching some goal state; the game then shows the next level. Many games have a "
    "step counter or lives shown on screen; running out loses the life and restarts the level.\n"
    "- Every action counts against you, including ones that change nothing: finish each level in as FEW "
    "actions as possible.\n\n"
    "Answer with exactly one JSON object and nothing else:\n"
    '{"theory": "<one or two sentences: what the game is and what you are trying>", '
    '"plan": ["ACTION4", "ACTION4", "ACTION6 12 30"]}\n'
    "The plan holds 1 to 8 actions to take next, in order."
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "theory": {"type": "string", "maxLength": 400},
        "plan": {"type": "array", "minItems": 1, "maxItems": LLM_PLAN_MAX,
                 "items": {"type": "string", "pattern": "^ACTION[1-7]( [0-9]{1,2} [0-9]{1,2})?$"}},
    },
    "required": ["theory", "plan"],
    "additionalProperties": False,
}

OBJECT_SYSTEM_PROMPT = (
    "You help an agent play an unknown turn-based puzzle game on a 64x64 grid. Nobody knows the rules or the goal. "
    "The agent's eyes have split the screen into numbered objects (ids stay the same during a level), noticed some "
    "relations between them, and recorded exactly what each action did to which object. Your job is to explain the "
    "game in terms of these objects and to propose ONE experiment that tests your explanation.\n\n"
    "- ACTION1..ACTION5 are simple inputs (often up, down, left, right, use), ACTION6 x y clicks a cell, ACTION7 is "
    "often undo. Only the listed actions exist in this game.\n"
    "- Every action counts against the player, so the experiment must be informative.\n"
    "- Base every claim on the evidence given. If an earlier prediction of yours was wrong, change your "
    "explanation.\n\n"
    "Answer with exactly one JSON object:\n"
    '{"rule": "<how the game works, in terms of object ids>", "controlled": [ids you believe the player moves], '
    '"goal": "<what must become true to finish the level, in object ids, e.g. #4 covers #5>", '
    '"roles": [{"id": 4, "role": "player"}, ...], '
    '"test": {"action": "ACTION4", "object": 4, "change": "moved", "direction": "left"}}\n'
    "test.change is what you expect to happen to test.object: moved (give the direction), appeared, disappeared, "
    "recoloured, or nothing."
)

ROLES = ["player", "goal", "mirror-copy", "mirror-line", "wall", "floor", "pushable", "collectible", "hazard",
         "switch", "indicator", "background", "decoration", "unknown"]
OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "rule": {"type": "string", "maxLength": 400},
        "controlled": {"type": "array", "maxItems": 8, "items": {"type": "integer"}},
        "goal": {"type": "string", "maxLength": 250},
        "roles": {"type": "array", "maxItems": 16, "items": {
            "type": "object", "properties": {"id": {"type": "integer"}, "role": {"type": "string", "enum": ROLES}},
            "required": ["id", "role"], "additionalProperties": False}},
        "test": {"type": "object", "properties": {
            "action": {"type": "string", "pattern": "^ACTION[1-7]( [0-9]{1,2} [0-9]{1,2})?$"},
            "object": {"type": "integer"},
            "change": {"type": "string", "enum": ["moved", "appeared", "disappeared", "recoloured", "nothing"]},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right", "none"]}},
            "required": ["action", "object", "change", "direction"], "additionalProperties": False},
    },
    "required": ["rule", "controlled", "goal", "roles", "test"],
    "additionalProperties": False,
}


class Hypothesis(NamedTuple):
    """The advisor's explanation of the game in terms of scene object ids, plus one experiment."""
    rule: str
    controlled: list[int]
    goal: str
    roles: dict[int, str]
    test_action: Action | None
    test_object: int
    test_change: str
    test_direction: str

    def prediction(self) -> str:
        if self.test_change == "nothing":
            return "nothing changes"
        return f"#{self.test_object} {self.test_change}" + (
            f" {self.test_direction}" if self.test_change == "moved" and self.test_direction != "none" else "")


def parse_hypothesis(text: str, available: list[int]) -> Hypothesis | None:
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        test = data["test"]
        tests = parse_plan(json.dumps({"plan": [test["action"]]}), available)
        return Hypothesis(str(data["rule"]), [int(i) for i in data["controlled"]], str(data["goal"]),
                          {int(r["id"]): str(r["role"]) for r in data["roles"]}, tests[0] if tests else None,
                          int(test["object"]), str(test["change"]), str(test["direction"]))
    except (ValueError, KeyError, TypeError):
        return None


class Lab:
    """Hypothesis mode's notebook: the explanation being tested, and how earlier tests turned out."""

    def __init__(self) -> None:
        self.pending: Hypothesis | None = None  # its test action was just sent; checked on the next screen
        self.latest: Hypothesis | None = None
        self.history: list[str] = []
        self.last_verdict = ""
        self.verdicts = 0  # tests judged so far (the history itself is capped)
        self.next_ask_at = 0

    def record(self, verdict: str) -> None:
        h = self.pending
        if h is None:
            return
        name = f"ACTION6 {h.test_action[1]} {h.test_action[2]}" if h.test_action and h.test_action[0] == 6 \
            else f"ACTION{h.test_action[0]}" if h.test_action else "?"
        self.history.append(f"you said: {h.rule} Goal: {h.goal} -> test {name}: {verdict}")
        self.history = self.history[-LLM_HISTORY:]
        self.last_verdict = verdict
        self.verdicts += 1
        self.pending = None

    @staticmethod
    def check(h: Hypothesis, observation: Observation) -> str:
        """Did the test action do what the hypothesis predicted?"""
        if h.test_change == "nothing":
            ok = not observation.changes
        else:
            mine = [c for c in observation.changes if c.object_id == h.test_object]
            if h.test_change == "moved":
                ok = any(c.kind == "moved" and Lab._direction_ok(c.dx, c.dy, h.test_direction) for c in mine)
            else:
                ok = any(c.kind == h.test_change for c in mine)
        return f"{'CORRECT' if ok else 'WRONG'} (predicted {h.prediction()}; actually {observation.effects()})"

    @staticmethod
    def _direction_ok(dx: int, dy: int, direction: str) -> bool:
        return {"up": dy < 0, "down": dy > 0, "left": dx < 0, "right": dx > 0}.get(direction, True)


_ACTION_RE = re.compile(r"ACTION([1-7])(?:\s+(\d{1,2})\s+(\d{1,2}))?")


class Advice(NamedTuple):
    """One answer: the actions to take and the model's stated theory of the game."""
    plan: list[Action]
    theory: str


NO_ADVICE = Advice([], "")


class Advisor:
    """One client shared by every game thread. Failing requests pause it for everyone."""

    def __init__(self, url: str, model: str) -> None:
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        env = os.environ.get
        self.thinking = env("ARC3_LLM_THINKING", "1" if LLM_THINKING else "0") == "1"
        self.max_tokens = int(env("ARC3_LLM_MAX_TOKENS", LLM_MAX_TOKENS_THINKING if self.thinking else LLM_MAX_TOKENS))
        self.timeout = float(env("ARC3_LLM_TIMEOUT_S", LLM_TIMEOUT_THINKING_S if self.thinking else LLM_TIMEOUT_S))
        self.sampling = LLM_SAMPLING_THINKING if self.thinking else LLM_SAMPLING
        self.structured = env("ARC3_LLM_STRUCTURED", "1" if LLM_STRUCTURED_OUTPUT else "0") == "1"
        self._in_flight = threading.BoundedSemaphore(int(env("ARC3_LLM_MAX_CONCURRENT", LLM_MAX_CONCURRENT)))
        self._lock = threading.Lock()
        self._fails_in_a_row = 0
        self._paused_until = 0.0

    def available(self) -> bool:
        return time.time() >= self._paused_until

    def propose(self, prompt: str, available: list[int], tag: str = "") -> Advice:
        """Ask for a plan; `tag` (the game id) only labels the request log."""
        if not self.available():
            return NO_ADVICE
        text = self._ask(prompt, tag)
        if not text:
            return NO_ADVICE
        return Advice(parse_plan(text, available), parse_theory(text))

    def hypothesise(self, question: str, available: list[int], tag: str = "") -> Hypothesis | None:
        """Ask the object question; None if there is no usable answer."""
        if not self.available():
            return None
        text = self._ask(question, tag, OBJECT_SYSTEM_PROMPT, OBJECT_SCHEMA)
        return parse_hypothesis(text, available) if text else None

    def _ask(self, prompt: str, tag: str = "", system: str | None = None, schema: dict | None = None) -> str | None:
        with self._in_flight:  # wait our turn; the timeout below starts when the request is sent
            return self._send(prompt, tag, system or SYSTEM_PROMPT, schema or PLAN_SCHEMA)

    def _send(self, prompt: str, tag: str, system: str, schema: dict) -> str | None:
        started = time.time()
        request_body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens, **self.sampling,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if self.structured:
            request_body["response_format"] = {"type": "json_schema",
                                               "json_schema": {"name": "answer", "schema": schema}}
        body = json.dumps(request_body).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                answer = json.loads(response.read())
            text = answer["choices"][0]["message"].get("content") or ""
        except Exception as e:
            self._failed()
            self._log({"game": tag, "seconds": round(time.time() - started, 1), "error": repr(e)[:300]})
            return None
        with self._lock:
            self._fails_in_a_row = 0
        self._log({"game": tag, "seconds": round(time.time() - started, 1), "usage": answer.get("usage"),
                   "finish": answer["choices"][0].get("finish_reason"), "answer": text[-1500:]})
        return text

    def _log(self, record: dict) -> None:
        """Diagnostics only: one JSON line per request when ARC3_LLM_LOG names a file."""
        path = os.environ.get("ARC3_LLM_LOG")
        if not path:
            return
        record["t"] = round(time.time(), 1)
        try:
            with self._lock, open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _failed(self) -> None:
        with self._lock:
            self._fails_in_a_row += 1
            if self._fails_in_a_row >= LLM_FAILS_BEFORE_PAUSE:
                self._paused_until = time.time() + LLM_PAUSE_S
                self._fails_in_a_row = 0


_ADVISOR: Advisor | None = None
_ADVISOR_LOCK = threading.Lock()


def advisor() -> Advisor | None:
    global _ADVISOR
    url = os.environ.get("ARC3_LLM_URL")
    if not url:
        return None
    with _ADVISOR_LOCK:
        if _ADVISOR is None:
            _ADVISOR = Advisor(url, os.environ.get("ARC3_LLM_MODEL", LLM_MODEL))
    return _ADVISOR


# ── the answer ──────────────────────────────────────────────────────────────


def parse_theory(text: str) -> str:
    """The "theory" field of the answer, or "" if there is none."""
    match = re.search(r'"theory"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match is None:
        return ""
    try:
        return str(json.loads('"' + match.group(1) + '"'))
    except ValueError:
        return match.group(1)


def parse_plan(text: str, available: list[int]) -> list[Action]:
    """Actions from the last {"plan": [...]} in the answer; illegal ones are dropped. Never RESET:
    a RESET with no steps taken restarts the whole game, so only the brain may reset."""
    steps = None
    for match in reversed(list(re.finditer(r"\{[^{}]*\"plan\"[^{}]*\}", text))):
        try:
            steps = json.loads(match.group(0))["plan"]
            break
        except (ValueError, KeyError):
            continue
    if not isinstance(steps, list):
        return []
    plan = []
    for step in steps[:LLM_PLAN_MAX]:
        m = _ACTION_RE.fullmatch(str(step).strip().upper().replace(",", " ").replace("  ", " "))
        if m is None:
            continue
        action_id = int(m.group(1))
        if action_id not in available:
            continue
        if action_id == 6:
            if m.group(2) is None:
                continue
            x, y = int(m.group(2)), int(m.group(3))
            if not (0 <= x < 64 and 0 <= y < 64):
                continue
            plan.append(Action(6, x, y))
        else:
            plan.append(Action(action_id, -1, -1))
    return plan


# ── the question ────────────────────────────────────────────────────────────


class PromptWriter:
    """Writes the question for the advisor from the beliefs and short-term memory (read-only)."""

    def __init__(self, world: World, memory: WorkingMemory, stats: defaultdict[str, int]) -> None:
        self.world = world
        self.memory = memory
        self.stats = stats

    def describe(self, grid: np.ndarray, available: list[int]) -> str:
        world, m = self.world, self.memory
        level = m.level if m.level is not None else 0
        level_map = world.levels.get(level)
        parts = [f"Level {level + 1} (levels completed so far: {level}). "
                 f"Actions taken in this life: {m.life_length}. Lives lost so far: {self.stats['deaths']}."]
        budget = world.rules.steps_per_life.get(level)
        if budget:
            parts.append(f"A life on this level seems to last about {budget} actions.")
        parts.append("Available actions: " + ", ".join(f"ACTION{a}" for a in sorted(available) if a != 0) + ".")
        parts += self._what_actions_do(available)
        colour = world.avatar.colour
        if colour is not None:
            at = world.avatar.position(world.counters.hide(grid))
            where = f" (centre near x={at[1] // 4}, y={at[0] // 4})" if at else ""
            parts.append(f"You probably control the object of colour {colour:x}{where}.")
        roles = {c: r.most_common(1)[0][0] for c, r in world.colour_roles.evidence.items() if r}
        if roles:
            parts.append("Touching colours seemed to: "
                         + ", ".join(f"{c:x}={r.value}" for c, r in sorted(roles.items())) + ".")
        won = [g for g in world.goals if g.source.startswith("won")]
        guesses = [g for g in world.goals if not g.source.startswith("won")]
        if won:  # only real level completions are presented as evidence
            parts.append("Earlier level completions came with: " + "; ".join(self._goal_text(g) for g in won[:4]) + ".")
        if guesses:
            parts.append("Unconfirmed guess from the first look (one shape appears in several colours): maybe "
                         + "; ".join(self._goal_text(g) for g in guesses[:2]) + ". No level has been won yet"
                         if not won else "Unconfirmed guess: " + "; ".join(self._goal_text(g) for g in guesses[:2]))
        if level_map is not None:
            parts.append(f"Distinct screens seen on this level: {len(level_map.screens)}.")
        if m.recent:
            parts.append("Your most recent steps, oldest first:\n" + "\n".join(f"  {s}" for s in m.recent))
        parts.append("Current screen (row number, then 64 hex digits for x=0..63):\n" + self.render_grid(grid))
        return "\n".join(parts)

    def object_question(self, scene: LevelScene, relations: list[str], observations: list[Observation],
                        history: list[str], available: list[int]) -> str:
        """The question in terms of objects: what is on screen, what the brain noticed, what actions did."""
        m = self.memory
        level = m.level if m.level is not None else 0
        parts = [f"Level {level + 1}. Lives lost so far: {self.stats['deaths']}. "
                 f"Actions in this game: " + ", ".join(f"ACTION{a}" for a in sorted(available) if a != 0) + "."]
        parts.append("Objects on screen now:\n" + "\n".join(f"  {line}" for line in self._object_lines(scene)))
        if relations:
            parts.append("Relations the agent noticed:\n" + "\n".join(f"  {r}" for r in relations))
        if observations:
            parts.append("What the most recent actions did (oldest first):\n"
                         + "\n".join(f"  {o.describe()}" for o in observations))
        else:
            parts.append("No action has been taken on this level yet.")
        if history:
            parts.append("Your earlier explanations and how their tests turned out:\n"
                         + "\n".join(f"  {h}" for h in history))
        return "\n".join(parts)

    @staticmethod
    def _object_lines(scene: LevelScene) -> list[str]:
        """One line per object; composites (parts that always move together) and split objects appear once."""
        first: dict[int, SceneObject] = {}
        pieces: dict[int, int] = {}
        for o in scene.objects:
            root = scene.root(o.id)
            pieces[o.id] = pieces.get(o.id, 0) + 1
            if root == o.id and o.id not in first:
                first[o.id] = o
        lines: list[str] = []
        kinds: dict[tuple, list[SceneObject]] = {}
        for o in first.values():
            if o.background:
                lines.append(o.describe())
            else:
                kinds.setdefault((o.colour, o.shape, not scene.members(o.id)), []).append(o)
        for group in sorted(kinds.values(), key=lambda g: -g[0].size)[:28]:
            if len(group) >= 3:  # many identical objects: one line
                where = " ".join(f"(x{o.bbox[1]},y{o.bbox[0]})" for o in group[:12])
                what = group[0].describe().split(", ", 1)[1].split(" box at")[0]
                ids = " ".join(f"#{o.id}" for o in group)
                lines.append(f"{ids}: {len(group)} identical {what} objects at {where}")
                continue
            for o in group:
                line = o.describe()
                if pieces.get(o.id, 1) > 1:
                    line += f" (now in {pieces[o.id]} pieces: partly covered)"
                parts = scene.members(o.id)
                if parts:
                    line += " + parts " + " ".join(f"#{i}" for i in parts) + " (they always move with it)"
                lines.append(line)
        if len(kinds) > 28:
            lines.append("... and more small objects not listed")
        return lines

    @staticmethod
    def render_grid(grid: np.ndarray) -> str:
        return "\n".join(f"{y:02d} " + "".join(f"{v:x}" for v in row) for y, row in enumerate(grid.tolist()))

    @staticmethod
    def describe_step(action: Action, before: np.ndarray | None, after: np.ndarray | None,
                      outcome: str | None) -> str:
        """One line of history: what an action did."""
        name = f"ACTION6 {action[1]} {action[2]}" if action[0] == 6 else f"ACTION{action[0]}"
        if outcome is not None:
            return f"{name} -> {outcome}"
        if before is None or after is None:
            return f"{name} -> ?"
        changed = (before != after) & (before != 255) & (after != 255)
        n = int(changed.sum())
        if n == 0:
            return f"{name} -> nothing changed"
        ys, xs = np.nonzero(changed)
        return f"{name} -> {n} cells changed within x={xs.min()}..{xs.max()}, y={ys.min()}..{ys.max()}"

    def _what_actions_do(self, available: list[int]) -> list[str]:
        lines = []
        for a in sorted(available):
            if a == 0:
                continue
            tally = self.world.actions.changed_screen.get(a)
            if tally is None or tally.tries == 0:
                continue
            text = f"ACTION{a}: changed the screen {tally.hits} of {tally.tries} times"
            steps = self.world.avatar.steps.get(a)
            if steps:
                (dy, dx), _ = steps.most_common(1)[0]
                text += f"; moves your object by dx={dx:+d}, dy={dy:+d}"
            lines.append(text + ".")
        return lines

    @staticmethod
    def _goal_text(goal: Goal) -> str:
        kind, *colours = goal.feature
        change = "more" if goal.direction > 0 else "fewer"
        if kind == "count":
            return f"{change} cells of colour {colours[0]:x}"
        if kind == "touch":
            return f"{change} places where colours {colours[0]:x} and {colours[1]:x} touch"
        return f"your object touching {change} cells of colour {colours[0]:x}"
