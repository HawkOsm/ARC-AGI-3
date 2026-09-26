"""The LLM advisor (agent/brain/llm.py): parsing, request bodies, failure pause, concurrency limit, logging.
No network: urllib.request.urlopen is replaced by a fake server."""
import json
import threading
import time
import urllib.request
from collections import defaultdict, deque

import numpy as np
import pytest

from agent.brain import llm
from agent.brain.llm import Advisor, PromptWriter, parse_plan
from agent.brain.world import Action, WorkingMemory, World


def test_parse_plan_reads_the_json_plan():
    text = 'thinking... {"theory": "move right", "plan": ["ACTION4", "ACTION6 12 30", "ACTION1"]}'
    assert parse_plan(text, [1, 4, 6]) == [Action(4), Action(6, 12, 30), Action(1)]


def test_parse_plan_drops_illegal_and_out_of_range_steps_and_never_resets():
    text = '{"plan": ["ACTION2", "ACTION6 70 3", "RESET", "ACTION6", "banana", "ACTION1"]}'
    assert parse_plan(text, [1, 6]) == [Action(1)]


def test_parse_theory_reads_escaped_text():
    assert llm.parse_theory('{"theory": "a \\"box\\" moves", "plan": []}') == 'a "box" moves'
    assert llm.parse_theory("no json here") == ""


def test_parse_plan_uses_the_last_plan_and_survives_garbage():
    assert parse_plan('{"plan": ["ACTION1"]} then {"plan": ["ACTION2"]}', [1, 2]) == [Action(2)]
    assert parse_plan("I think we should move up. {plan: ACTION1", [1]) == []
    assert parse_plan("", [1]) == []


class FakeServer:
    """Stands in for urllib.request.urlopen; records request bodies."""

    def __init__(self, reply='{"theory": "t", "plan": ["ACTION1"]}', fail=False, delay=0.0):
        self.reply, self.fail, self.delay = reply, fail, delay
        self.bodies: list[dict] = []
        self.in_flight = self.max_in_flight = 0
        self.lock = threading.Lock()

    def __call__(self, request, timeout):
        with self.lock:
            self.bodies.append(json.loads(request.data))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            if self.fail:
                raise TimeoutError("timed out")
            return _Response(self.reply)
        finally:
            with self.lock:
                self.in_flight -= 1


class _Response:
    def __init__(self, content):
        self.data = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                "usage": {"completion_tokens": 5}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.data


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    for var in ("ARC3_LLM_THINKING", "ARC3_LLM_MAX_TOKENS", "ARC3_LLM_TIMEOUT_S", "ARC3_LLM_STRUCTURED",
                "ARC3_LLM_MAX_CONCURRENT", "ARC3_LLM_LOG"):
        monkeypatch.delenv(var, raising=False)
    return fake


def test_default_request_is_non_thinking_with_json_schema(server):
    advice = Advisor("http://x/v1", "qwen").propose("Available actions: ACTION1.", [1])
    assert advice.plan == [Action(1)] and advice.theory == "t"
    body = server.bodies[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == llm.LLM_MAX_TOKENS
    assert body["temperature"] == 0.7 and body["presence_penalty"] == 1.5
    assert body["response_format"]["json_schema"]["schema"] == llm.PLAN_SCHEMA


def test_thinking_mode_switches_sampling_and_budget(server, monkeypatch):
    monkeypatch.setenv("ARC3_LLM_THINKING", "1")
    monkeypatch.setenv("ARC3_LLM_STRUCTURED", "0")
    Advisor("http://x/v1", "qwen").propose("p", [1])
    body = server.bodies[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["temperature"] == 1.0 and body["max_tokens"] == llm.LLM_MAX_TOKENS_THINKING
    assert "response_format" not in body


def test_three_failures_pause_the_advisor(server):
    server.fail = True
    adv = Advisor("http://x/v1", "qwen")
    for _ in range(llm.LLM_FAILS_BEFORE_PAUSE):
        assert adv.propose("p", [1]).plan == []
    assert not adv.available()
    asked = len(server.bodies)
    assert adv.propose("p", [1]).plan == []  # paused: no request is sent
    assert len(server.bodies) == asked


def test_concurrency_limit_queues_requests_on_our_side(server, monkeypatch):
    monkeypatch.setenv("ARC3_LLM_MAX_CONCURRENT", "2")
    server.delay = 0.05
    adv = Advisor("http://x/v1", "qwen")
    threads = [threading.Thread(target=adv.propose, args=("p", [1])) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(server.bodies) == 8
    assert server.max_in_flight == 2


def test_request_log_records_timing_and_answers(server, monkeypatch, tmp_path):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("ARC3_LLM_LOG", str(log))
    Advisor("http://x/v1", "qwen").propose("p", [1])
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["finish"] == "stop" and "seconds" in record and "ACTION1" in record["answer"]


def test_prompt_contains_actions_history_and_the_whole_screen():
    world = World()
    memory = WorkingMemory(level=0, recent=deque(["ACTION1 -> nothing changed"], maxlen=16))
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[10, 20] = 11
    text = PromptWriter(world, memory, defaultdict(int)).describe(grid, [1, 2, 6])
    assert "Available actions: ACTION1, ACTION2, ACTION6." in text
    assert "ACTION1 -> nothing changed" in text
    rows = [line for line in text.splitlines() if len(line) == 67 and line[2] == " "]
    assert len(rows) == 64 and rows[10][3 + 20] == "b"


def test_describe_step_summarises_what_changed():
    before = np.zeros((4, 4), dtype=np.uint8)
    after = before.copy()
    after[1, 2] = 5
    assert PromptWriter.describe_step(Action(4), before, before, None) == "ACTION4 -> nothing changed"
    assert PromptWriter.describe_step(Action(6, 2, 1), before, after, None) == \
        "ACTION6 2 1 -> 1 cells changed within x=2..2, y=1..1"
    assert PromptWriter.describe_step(Action(1), None, None, "LEVEL COMPLETE") == "ACTION1 -> LEVEL COMPLETE"


def test_prompt_never_calls_a_first_look_guess_a_win():
    from agent.brain.world import Goal
    world = World()
    world.goals.append(Goal(feature=("touch", 4, 11), direction=1, support=0, against=0,
                            source="first look level 0: same shape", prior=0.45))
    memory = WorkingMemory(level=0, recent=deque(maxlen=16))
    text = PromptWriter(world, memory, defaultdict(int)).describe(np.zeros((64, 64), dtype=np.uint8), [1])
    assert "Earlier level completions" not in text and "wins came with" not in text
    assert "Unconfirmed guess" in text and "No level has been won yet" in text
