"""End to end: the Brain plays real public games offline (environment_files/, from `make setup`)."""
import os
from pathlib import Path

import numpy as np
import pytest

from agent.brain.brain import Brain
from agent.brain.world import RESET_ACTION, GameMode

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "environment_files" / "ls20").exists(),
                                reason="public games not downloaded (make setup / make verify-local)")


def _play(game: str, steps: int, monkeypatch) -> Brain:
    monkeypatch.delenv("ARC3_LLM_URL", raising=False)  # the plain brain; no network
    import arc_agi
    from arcengine import GameAction

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                         environments_dir=str(ROOT / "environment_files"))
    card = arc.open_scorecard(tags=["pytest"])
    env = arc.make(game, scorecard_id=card)
    frame = env.reset()
    brain = Brain(seed=1)
    for _ in range(steps):
        action_id, x, y = brain.next_action(frame.state.value, frame.frame, frame.levels_completed,
                                            list(frame.available_actions or []), bool(frame.full_reset))
        assert action_id == 0 or action_id in frame.available_actions or not frame.available_actions
        data = {"x": x, "y": y} if action_id == 6 else {}
        frame = env.step(GameAction.from_id(action_id), data=data)
    arc.close_scorecard(card)
    return brain


def test_brain_plays_a_key_game(monkeypatch):
    brain = _play("ls20", 120, monkeypatch)
    world = brain.world
    assert world.mode == GameMode.MOVER
    assert world.avatar.colour is not None  # it found what it controls
    assert world.counters.ignore.any()  # and the step counter to ignore
    assert brain.stats["screens"] > 10


def test_brain_plays_a_click_game(monkeypatch):
    brain = _play("vc33", 80, monkeypatch)
    assert brain.world.mode == GameMode.CLICKER
    assert brain.stats["screens"] > 1
    assert brain.memory.last_action != RESET_ACTION or brain.stats["deaths"] > 0


def test_brain_never_calls_the_llm_when_off(monkeypatch):
    import urllib.request

    def no_network(*a, **k):
        raise AssertionError("advisor off must not send requests")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    brain = _play("ls20", 30, monkeypatch)
    assert brain.stats["llm_questions"] == 0
    assert os.environ.get("ARC3_LLM_URL") is None
    assert isinstance(brain.world.counters.ignore, np.ndarray)
