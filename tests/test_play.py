"""Play mode (agent/brain/player.py, render.py): the LLM drives, the code keeps only Reki's helpers.
No network: the fake server from test_llm stands in for vLLM."""
import json
import struct
import urllib.request
import zlib

import numpy as np

from agent.brain import llm
from agent.brain.brain import Brain
from agent.brain.player import ClickHelper, Player, parse_play
from agent.brain.render import MARGIN, PALETTE, FramePainter
from agent.brain.world import Action
from test_llm import FakeServer


def test_png_is_valid_and_shows_the_screen_with_a_ruler():
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[10, 20] = 8  # one red cell
    png = FramePainter.png(grid, scale=4)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (MARGIN + 256, MARGIN + 256)
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    rows = zlib.decompress(idat)
    stride = 1 + width * 3
    pixel = rows[(MARGIN + 10 * 4 + 1) * stride + 1 + (MARGIN + 20 * 4 + 1) * 3:][:3]
    assert tuple(pixel) == tuple(PALETTE[8])
    ruler = FramePainter.pixels(grid, 4)[:MARGIN, MARGIN:]
    assert (ruler == 40).any()  # digits drawn in the top margin


def test_parse_play_reads_actions_and_drops_illegal_ones():
    text = '{"changed": "c", "plan": "p", "notes": "n", "actions": ["ACTION1", "ACTION6 3 4", "ACTION5"]}'
    answer = parse_play(text, [1, 6])
    assert answer is not None and answer.actions == [Action(1), Action(6, 3, 4)] and answer.notes == "n"
    assert parse_play("garbage", [1]) is None


def test_clicks_snap_onto_objects_and_dead_kinds_are_refused():
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[30:33, 30:33] = 9  # a blue 3x3 button on a white screen
    assert ClickHelper.snap(grid, 29, 31) == (30, 31)  # one cell off: lands on the button
    assert ClickHelper.snap(grid, 5, 5) == (5, 5)  # nothing near: stays
    player = Player()
    assert player.legal(Action(6, 29, 31), grid, [6]) == Action(6, 30, 31)
    player.observe(grid, Action(6, 31, 31), grid, changed=False)  # clicked the button: nothing happened
    assert player.legal(Action(6, 31, 31), grid, [6]) is None  # dead signature: not clicked again
    assert player.legal(Action(1), grid, [6]) is None  # not offered in this game


def test_the_brain_plays_the_models_actions(monkeypatch):
    """A real game with a fake model that always answers 'ACTION1, ACTION2': the brain sends exactly those, asks
    with images, and asks again when the queue is empty."""
    import arc_agi
    from arcengine import GameAction
    reply = json.dumps({"changed": "c", "plan": "go", "notes": "n", "actions": ["ACTION1", "ACTION2"]})
    fake = FakeServer(reply=reply)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("ARC3_LLM_URL", "http://fake/v1")
    monkeypatch.setenv("ARC3_LLM_MODE", "play")
    monkeypatch.setattr(llm, "_ADVISOR", None)
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("ls20", scorecard_id=arc.open_scorecard(tags=["test"]))
    frame = env.reset()
    brain = Brain(seed=0, name="ls20")
    sent = []
    for _ in range(8):
        action = brain.next_action(frame.state.value, frame.frame, frame.levels_completed,
                                   list(frame.available_actions or []), bool(frame.full_reset))
        sent.append(action)
        frame = env.step(GameAction.from_id(action[0]), data={})
    moves = [a for a in sent if a[0] != 0]
    assert moves[:4] == [Action(1), Action(2), Action(1), Action(2)]
    content = fake.bodies[-1]["messages"][1]["content"]
    assert any(part["type"] == "image_url" for part in content)
    assert fake.bodies[-1]["response_format"]["json_schema"]["schema"]["required"] == ["changed", "plan", "notes",
                                                                                       "actions"]
    monkeypatch.setattr(llm, "_ADVISOR", None)
