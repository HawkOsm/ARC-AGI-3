"""agent/model.py: the sandbox, the conversation, the game wrapper, and one whole turn with a fake model."""
import json
import struct

import pytest

from agent.model import LLM, Agent, Config, Conversation, Game, Sandbox, png_of


def _state(*frames, result=None):
    """Sandbox state from (action, grid) pairs, oldest first."""
    return {"history": [{"action": a, "frame": {"grid": g, "level": 1, "step": i}} for i, (a, g) in enumerate(frames)],
            "last_action_result": result or {}, "valid_actions": ["UP", "MOUSE"]}


def test_sandbox_sees_the_frame_calls_actions_and_blocks_imports():
    grid = [[0] * 4 for _ in range(4)]
    grid[1][1] = grid[1][2] = 9  # a blue 1x2 object inside white
    calls = []

    def act(actions):
        calls.append(actions)
        moved = [row[:] for row in grid]
        moved[1][1], moved[1][3] = 0, 9
        return {"executed": True, "board_changed": True}, _state(("", grid), ("RIGHT", moved),
                                                                  result={"board_changed": True})

    code = ("nodes = current_frame.segmentation['nodes']\n"
            "print([(n['color'], n['size'], n['bbox'], n['pixels'][:1]) for n in nodes])\n"
            "r = action(['RIGHT'])\n"
            "print(r['board_changed'], current_frame.ascii.splitlines()[1], previous_frame is not None)\n"
            "result = len(history)")
    out = Sandbox.run(code, _state(("", grid)), act, 10)
    assert out["error"] is None, out
    assert "('b', 2, [1, 1, 1, 2], [[1, 1]])" in out["stdout"]
    assert "True WWbb True" in out["stdout"]  # after action(): the refreshed frame, and a previous frame exists
    assert calls == [["RIGHT"]] and out["result"] == "2"
    blocked = Sandbox.run("import os", _state(("", grid)), act, 10)
    assert "not allowed" in blocked["error"]
    slow = Sandbox.run("while True: pass", _state(("", grid)), act, 1)
    assert "time limit" in slow["error"]


def test_conversation_keeps_notes_newest_image_and_fits_its_budget():
    c = Conversation(budget_tokens=3000)
    c.learn_notes("Some thinking.\nWorld model: a blue block moves right.\nPlan: click the red button.\n")
    assert c.notes == {"World model": "a blue block moves right.", "Plan": "click the red button."}
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    for turn in range(30):
        c.add({"role": "user", "content": [{"type": "text", "text": f"turn {turn} " + "x" * 300}, image]})
        c.add({"role": "assistant", "content": "ok " * 100})
    users = [m for m in c.messages if m["role"] == "user"]
    assert c.messages[0]["role"] == "system" and c._size() <= 3000
    assert isinstance(users[-1]["content"], list) and all(isinstance(u["content"], str) for u in users[:-1])
    assert "turn 29" in users[-1]["content"][0]["text"]


def test_png_is_valid():
    png = png_of([[0, 8], [9, 5]], 3)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (6, 6)


@pytest.fixture
def ls20():
    import arc_agi
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir="environment_files")
    return Game(arc.make("ls20", scorecard_id=arc.open_scorecard(tags=["test"])), "ls20")


def test_game_runs_actions_in_the_models_names(ls20):
    ls20.start()
    assert ls20.valid_names() and len(ls20.frames) == 1
    r = ls20.do("UP")
    assert r["executed"] and ls20.actions_taken == 1 and ls20.frames[-1][0] == "UP"
    assert not ls20.do({"action": "MOUSE", "row": 3, "col": 4})["executed"]  # ls20 has no clicks
    assert not ls20.do("JUMP")["executed"]


def test_one_turn_with_a_fake_model_plays_the_models_code(ls20, monkeypatch):
    code = "print(len(current_frame.segmentation['nodes']))\naction(['UP', 'DOWN'])"
    replies = iter([
        {"content": "World model: a maze.\nPlan: probe keys.", "reasoning": "think",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "python", "arguments": json.dumps({"code": code})}}], "usage": {}},
    ])
    monkeypatch.setattr(LLM, "chat", lambda self, messages: next(replies))
    agent = Agent(ls20, LLM(Config()), Config(image_upscale=2), deadline=1e18)
    ls20.start()
    agent.play_turn()
    assert ls20.actions_taken == 2 and [f[0] for f in ls20.frames[1:]] == ["UP", "DOWN"]
    assert agent.conversation.notes["Plan"] == "probe keys."
    last = agent.conversation.messages[-1]
    assert last["role"] == "tool" and "actions run" in last["content"]
    assert "executed 2 action(s): UP, DOWN" in agent.last_summary


def test_screen_setting_puts_letters_or_an_image_in_the_message_and_old_screens_are_dropped(ls20):
    from agent.model import SCREEN_TEXT_HEADER
    ls20.start()
    for screen, kinds in (("text", ["text", "text"]), ("image", ["text", "text", "image_url"]),
                          ("both", ["text", "text", "text", "image_url"]), ("none", None)):
        content = Agent(ls20, LLM(Config()), Config(screen=screen), deadline=1e18)._user_content()
        assert (content if isinstance(content, str) else [p["type"] for p in content]) == (kinds or content)
    c = Conversation(budget_tokens=100_000)
    agent = Agent(ls20, LLM(Config()), Config(screen="text"), deadline=1e18)
    c.add({"role": "user", "content": agent._user_content()})
    c.add({"role": "user", "content": agent._user_content()})
    assert SCREEN_TEXT_HEADER not in c.messages[1]["content"]  # the older message lost its letter grid
    assert any(p["text"].startswith(SCREEN_TEXT_HEADER) for p in c.messages[2]["content"])


def test_sandbox_helpers_rows_entry_frames_and_no_files():
    grid = [[0, 8], [9, 5]]
    code = ("print(current_frame.rows[1][0], current_frame.at(0, 1), history[-1].ascii.splitlines()[0])\n"
            "print(Counter(current_frame.rows[0]), len(deque([1, 2])), int(math.sqrt(16)))\n"
            "import hashlib\nprint(hashlib.md5(b'x').hexdigest()[:4])\n"
            "open('f_snap.txt', 'w')")
    out = Sandbox.run(code, _state(("", grid)), lambda a: ({}, _state(("", grid))), 10)
    assert "b R WR" in out["stdout"] and "Counter({'W': 1, 'R': 1}) 2 4" in out["stdout"]
    assert "9dd4" in out["stdout"] and "PermissionError" in out["error"]
